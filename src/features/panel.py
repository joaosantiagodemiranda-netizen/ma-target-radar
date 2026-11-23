"""Step 7 - Firm-year panel with point-in-time features and the takeover label.

One observation = one 10-K filed in OBS_FIRST_YEAR..OBS_LAST_YEAR. The prediction
date t is the 10-K filing date. Every feature uses only information public at t:

  Fundamentals   values reported inside that 10-K (see xbrl.py)
  SIC activity   share of same-SIC3 peers with a deal filing in the 24m before t
  Text activity  same, but peers = 50 nearest neighbours by TF-IDF / embeddings
  Target-like    similarity to firms that were *verifiably* acquired before t
                 (event + 18m completion window < 1 Jan of the filing year)
  Fluidity       1 - cos(own description at t, own description one year earlier)
  Signals        counts of 'strategic alternatives', activist, going-concern... text

Label  y = 1 if a completed-takeover event occurs in (t + GUARD_DAYS, t + 365d].
Dropped: deal filings within GUARD_DAYS of t (deal likely already announced) and
10-Ks whose own text discloses a signed merger agreement (pending_deal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

FUND_FEATURES = [
    "log_assets", "log_float", "roa", "op_margin", "loss", "sales_growth", "asset_growth",
    "leverage", "debt_assets", "cash_assets", "current_ratio", "cfo_assets", "capex_assets",
    "rnd_assets", "goodwill_assets", "div_payer", "mtb", "grd",
]
SIC_FEATURES = ["sic_deal_activity", "sic_n_peers", "own_prior_deal"]
SIGNAL_FEATURES = ["sig_strategic_alt", "sig_activist", "sig_going_concern",
                   "sig_change_control", "sig_consolidation"]
TFIDF_FEATURES = ["tfidf_deal_activity", "tfidf_deal_activity_w", "tfidf_crowd"]
EMB_FEATURES = ["emb_deal_activity", "emb_deal_activity_w", "emb_crowd",
                "emb_target_like", "emb_fluidity"]
WINDOW = 730  # deal-activity look-back (days)


# ----------------------------------------------------------------------------- helpers
def _safe_div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def observations(filings: pd.DataFrame) -> pd.DataFrame:
    ann = filings[filings["form"].isin(config.ANNUAL_FORMS)].copy()
    ann["fyear"] = ann["date"].dt.year
    ann = ann[ann["fyear"].between(config.OBS_FIRST_YEAR, config.OBS_LAST_YEAR)]
    ann = ann.sort_values("date").drop_duplicates(["cik", "fyear"])
    return ann[["cik", "fyear", "date", "accession"]].reset_index(drop=True)


def attach_docs(obs: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    docs = docs.reset_index(drop=True).assign(doc_row=lambda d: np.arange(len(d)))
    keep = ["doc_row", "n_words", "pending_deal"] + [c for c in docs if c.startswith("sig_")]
    by_acc = obs.merge(docs.dropna(subset=["accession"])[["accession"] + keep],
                       on="accession", how="left")
    miss = by_acc["doc_row"].isna()
    if miss.any():  # fall back on (cik, filing year)
        d2 = docs.drop_duplicates(["cik", "year"]).rename(columns={"year": "fyear"})
        fb = by_acc.loc[miss, obs.columns].merge(d2[["cik", "fyear"] + keep],
                                                 on=["cik", "fyear"], how="left")
        by_acc.loc[miss, keep] = fb[keep].to_numpy()
    by_acc["doc_row"] = by_acc["doc_row"].astype("Int64")
    return by_acc


def attach_fundamentals(obs: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    fund = fund.drop(columns=["cik", "filed"], errors="ignore")
    return obs.merge(fund, on="accession", how="left")


def fundamental_features(p: pd.DataFrame) -> pd.DataFrame:
    A = p["assets_cur"]
    p["log_assets"] = np.log(A.where(A > 0))
    p["log_float"] = np.log(p["public_float"].where(p["public_float"] > 0))
    p["roa"] = _safe_div(p["net_income_cur"], A)
    p["op_margin"] = _safe_div(p["op_income_cur"], p["revenue_cur"])
    p["loss"] = (p["net_income_cur"] < 0).astype(float).where(p["net_income_cur"].notna())
    p["sales_growth"] = _safe_div(p["revenue_cur"], p["revenue_prev"]) - 1
    p["asset_growth"] = _safe_div(A, p["assets_prev"]) - 1
    p["leverage"] = _safe_div(p["liabilities_cur"], A)
    p["debt_assets"] = _safe_div(p["lt_debt_cur"].fillna(0), A)
    p["cash_assets"] = _safe_div(p["cash_cur"], A)
    p["current_ratio"] = _safe_div(p["cur_assets_cur"], p["cur_liab_cur"])
    p["cfo_assets"] = _safe_div(p["cfo_cur"], A)
    p["capex_assets"] = _safe_div(p["capex_cur"].fillna(0), A)
    p["rnd_assets"] = _safe_div(p["rnd_cur"].fillna(0), A)
    p["goodwill_assets"] = _safe_div(p["goodwill_cur"].fillna(0), A)
    p["div_payer"] = (p["dividends_cur"].fillna(0) > 0).astype(float)
    p["mtb"] = _safe_div(p["public_float"], p["equity_cur"].where(p["equity_cur"] > 0))
    # Palepu (1986) growth-resource imbalance, cut-offs = medians of the *previous* year
    med = (p.groupby("fyear")[["sales_growth", "cash_assets", "leverage"]].median()
           .shift(1).reindex(p["fyear"]).to_numpy())
    g, liq, lev = (p[["sales_growth", "cash_assets", "leverage"]].to_numpy().T)
    p["grd"] = (((g < med[:, 0]) & (liq > med[:, 1]) & (lev < med[:, 2])) |
                ((g > med[:, 0]) & (liq < med[:, 1]) & (lev > med[:, 2]))).astype(float)
    return p


def label(p: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    def next_event(ev):
        e = ev[["cik", "event_date"]].sort_values("event_date")
        m = pd.merge_asof(p[["cik", "date"]].reset_index().sort_values("date"), e,
                          left_on="date", right_on="event_date", by="cik",
                          direction="forward", allow_exact_matches=True)
        return m.set_index("index").sort_index()["event_date"]
    done = next_event(events[events["completed"]])
    anyev = next_event(events)
    d_done = (done - p["date"]).dt.days
    d_any = (anyev - p["date"]).dt.days
    p["days_to_event"] = d_done
    p["y"] = (d_done > config.GUARD_DAYS) & (d_done <= config.GUARD_DAYS + config.LABEL_HORIZON_DAYS)
    p["y"] = p["y"].astype(int)
    p["guard_drop"] = d_any.between(0, config.GUARD_DAYS)
    return p


def deal_activity(pairs: pd.DataFrame, ev_dates: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """pairs: obs_id, date, nbr_cik, sim -> share of neighbours with a deal in [t-730, t)."""
    m = pairs.merge(ev_dates.rename(columns={"cik": "nbr_cik"}), on="nbr_cik", how="left")
    m["hit"] = ((m["event_date"] < m["date"]) &
                (m["event_date"] >= m["date"] - pd.Timedelta(days=WINDOW))).astype(float)
    per_pair = m.groupby(["obs_id", "nbr_cik"], sort=False).agg(hit=("hit", "max"), sim=("sim", "first"))
    per_pair["w"] = per_pair["hit"] * per_pair["sim"].clip(lower=0)
    g = per_pair.groupby(level=0)
    out = pd.DataFrame({
        f"{prefix}_deal_activity": g["hit"].mean(),
        f"{prefix}_deal_activity_w": g["w"].sum() / g["sim"].apply(lambda s: s.clip(lower=0).sum()).clip(lower=1e-9),
    })
    return out


def sic_features(p: pd.DataFrame, ev_dates: pd.DataFrame) -> pd.DataFrame:
    base = p[["obs_id", "cik", "fyear", "date", "sic3"]]
    pairs = base.merge(base[["fyear", "sic3", "cik"]].rename(columns={"cik": "nbr_cik"}),
                       on=["fyear", "sic3"])
    pairs = pairs[pairs["cik"] != pairs["nbr_cik"]].assign(sim=1.0)
    act = deal_activity(pairs[["obs_id", "date", "nbr_cik", "sim"]], ev_dates, "sic")
    act["sic_n_peers"] = pairs.groupby("obs_id").size()
    own = base.merge(ev_dates, on="cik", how="left")
    own["hit"] = ((own["event_date"] < own["date"]) &
                  (own["event_date"] >= own["date"] - pd.Timedelta(days=WINDOW)))
    act["own_prior_deal"] = own.groupby("obs_id")["hit"].max().astype(float)
    return act[["sic_deal_activity", "sic_n_peers", "own_prior_deal"]]


def tfidf_features(p, nb, ev_dates):
    pairs = p[["obs_id", "cik", "fyear", "date"]].merge(
        nb.rename(columns={"year": "fyear"}), on=["cik", "fyear"])
    act = deal_activity(pairs[["obs_id", "date", "nbr_cik", "sim"]], ev_dates, "tfidf")
    act["tfidf_crowd"] = pairs.sort_values("sim", ascending=False).groupby("obs_id")["sim"].apply(
        lambda s: s.head(10).mean())
    return act


def emb_features(p, docs, E, events, ev_dates, k=config.N_NEIGHBORS):
    E = E.astype(np.float32)
    docs = docs.reset_index(drop=True)
    has = p["doc_row"].notna()
    frames = []
    # (a) nearest neighbours among all 10-Ks of the same filing year
    for y, g in p[has].groupby("fyear"):
        pool = np.where(docs["year"].to_numpy() == y)[0]
        S = E[g["doc_row"].astype(int).to_numpy()] @ E[pool].T
        own = docs["cik"].to_numpy()[pool]
        S[own[None, :] == g["cik"].to_numpy()[:, None]] = -1          # exclude self
        top = np.argpartition(-S, k, axis=1)[:, :k]
        sims = np.take_along_axis(S, top, axis=1)
        frames.append(pd.DataFrame({
            "obs_id": np.repeat(g["obs_id"].to_numpy(), k),
            "date": np.repeat(g["date"].to_numpy(), k),
            "nbr_cik": own[top].ravel(), "sim": sims.ravel()}))
    pairs = pd.concat(frames, ignore_index=True)
    act = deal_activity(pairs, ev_dates, "emb")
    act["emb_crowd"] = pairs.sort_values("sim", ascending=False).groupby("obs_id")["sim"].apply(
        lambda s: s.head(10).mean())

    # (b) similarity to verifiably-completed past targets (their last pre-deal 10-K)
    done = events[events["completed"]][["cik", "event_date"]]
    d = docs[["cik", "year"]].assign(doc_row=np.arange(len(docs)))
    t = done.merge(d, on="cik")
    t = t[t["year"] < t["event_date"].dt.year].sort_values("year").drop_duplicates("cik", keep="last")
    tl = {}
    for y, g in p[has].groupby("fyear"):
        cutoff = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=config.DELIST_WINDOW_DAYS)
        tt = t[t["event_date"] < cutoff]
        if len(tt) < 5:
            continue
        S = E[g["doc_row"].astype(int).to_numpy()] @ E[tt["doc_row"].to_numpy()].T
        S[tt["cik"].to_numpy()[None, :] == g["cik"].to_numpy()[:, None]] = -1
        top5 = -np.sort(-S, axis=1)[:, :5].mean(axis=1)
        tl.update(dict(zip(g["obs_id"], top5)))
    act["emb_target_like"] = pd.Series(tl)

    # (c) fluidity: change of own description vs previous year
    prev = d.assign(year=d["year"] + 1).drop_duplicates(["cik", "year"]).rename(
        columns={"doc_row": "prev_row", "year": "fyear"})
    q = p[has][["obs_id", "cik", "fyear", "doc_row"]].merge(prev, on=["cik", "fyear"])
    flu = 1 - np.einsum("ij,ij->i", E[q["doc_row"].astype(int)], E[q["prev_row"]])
    act["emb_fluidity"] = pd.Series(flu, index=q["obs_id"].to_numpy())
    return act


# ----------------------------------------------------------------------------- build
def build(save: bool = True) -> pd.DataFrame:
    P = config.PROCESSED
    filings = pd.read_parquet(P / "filings.parquet")
    events = pd.read_parquet(P / "events.parquet")
    fund = pd.read_parquet(P / "fundamentals.parquet")
    comp = pd.read_parquet(P / "companies.parquet")
    docs = pd.read_parquet(P / "doc_index.parquet")
    nb = pd.read_parquet(P / "tfidf_neighbors.parquet")
    E = np.load(P / "embeddings.npy")

    attrition = []
    p = observations(filings); attrition.append(("10-K filings 2010-2020", len(p)))
    p = p.merge(comp[["cik", "name", "sic"]], on="cik", how="left")
    excl = np.zeros(len(p), bool)
    for lo, hi in config.EXCLUDED_SIC_RANGES:
        excl |= p["sic"].between(lo, hi).to_numpy()
    p = p[~excl & p["sic"].notna()]; attrition.append(("excl. financials / missing SIC", len(p)))
    p = attach_fundamentals(p, fund)
    p = p[p["assets_cur"] >= config.MIN_ASSETS_USD]; attrition.append(("XBRL match, assets >= $10m", len(p)))
    p = attach_docs(p, docs)
    p = p[p["doc_row"].notna()]; attrition.append(("10-K text available", len(p)))
    p = p.reset_index(drop=True)
    p["obs_id"] = np.arange(len(p))
    p["sic3"] = (p["sic"] // 10).astype(int)

    p = fundamental_features(p)
    p = label(p, events)
    ev_dates = events[["cik", "event_date"]]
    p = p.join(sic_features(p, ev_dates), on="obs_id")
    p = p.join(tfidf_features(p, nb, ev_dates), on="obs_id")
    p = p.join(emb_features(p, docs, E, events, ev_dates), on="obs_id")

    drop = p["guard_drop"] | p["pending_deal"].astype(bool)
    p = p[~drop]; attrition.append(("excl. deals already in play (guard + pending text)", len(p)))
    p = p.reset_index(drop=True)
    att = pd.DataFrame(attrition, columns=["step", "obs"])
    att["targets"] = np.nan
    att.loc[att.index[-1], "targets"] = p["y"].sum()
    print(att.to_string(index=False))
    print(p.groupby("fyear")["y"].agg(["size", "sum", "mean"]).to_string())
    if save:
        p.to_parquet(P / "panel.parquet", index=False)
        att.to_csv(config.TABLES / "sample_attrition.csv", index=False)
    return p


if __name__ == "__main__":
    build()
