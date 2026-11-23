"""Step 9 - Statistical and economic evaluation of the out-of-sample predictions.

Statistical  AUC, average precision (PR-AUC), Brier score, lift in the top decile,
             precision@K, block-bootstrap (by year) CIs for model differences.
Economic     'Coverage economics' for an M&A advisory team that can actively
             cover K companies per year: realised sell-side targets in the list,
             their aggregate deal value, and the implied advisory fee pool,
             vs. three practitioner baselines:
               - random list (expected value)
               - largest companies first (how coverage is usually allocated)
               - hottest SIC industry first (merger-wave rule of thumb)
Robustness   sub-periods, firm-size terciles, excluding going-private deals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src import config

K_LIST = (25, 50, 100)
MAIN = "M5_gbm_emb"


def fee_rate(deal_value: np.ndarray) -> np.ndarray:
    """Stylised sell-side fee schedule (declining % with size). Only ratios between
    strategies are reported in the paper, so the level is not critical."""
    v = np.asarray(deal_value, float)
    return np.where(v < 500e6, 0.0125, np.where(v < 2e9, 0.0075, 0.004))


def stats(df: pd.DataFrame) -> dict:
    y, p = df["y"].to_numpy(), df["p"].to_numpy()
    out = {"n": len(y), "targets": int(y.sum()), "base_rate": y.mean()}
    if y.sum() and (1 - y).sum() and np.unique(p).size > 1:
        out.update(auc=roc_auc_score(y, p), ap=average_precision_score(y, p))
    else:
        out.update(auc=0.5, ap=y.mean())
    out["brier"] = brier_score_loss(y, p)
    if np.unique(p).size == 1:          # constant score: a top-decile list is a random list
        out["lift_top10"] = 1.0
    else:
        top = df.nlargest(max(1, len(df) // 10), "p")
        out["lift_top10"] = top["y"].mean() / max(y.mean(), 1e-9)
    return out


def precision_at_k(df: pd.DataFrame, k: int) -> float:
    def one(g):
        return g["y"].mean() if g["p"].nunique() == 1 else g.nlargest(k, "p")["y"].mean()
    return df.groupby("fyear").apply(one).mean()


def within_year(g: pd.DataFrame) -> dict:
    """Ranking metrics computed within each test year, averaged with weights = targets.
    (Pooling across years would reward models for tracking the aggregate M&A cycle,
    which is not the cross-sectional task a coverage team faces.)"""
    per = [(stats(h), h["y"].sum()) for _, h in g.groupby("fyear") if h["y"].sum() > 0]
    w = np.array([n for _, n in per], float)
    return {k: float(np.average([s[k] for s, _ in per], weights=w)) for k in ("auc", "ap", "lift_top10")}


def summary_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in pred.groupby("model"):
        r = {"model": m, **stats(g), **within_year(g)}
        for k in K_LIST:
            r[f"prec@{k}"] = precision_at_k(g, k)
        rows.append(r)
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def by_year(pred: pd.DataFrame) -> pd.DataFrame:
    return (pred.groupby(["model", "fyear"]).apply(lambda g: pd.Series(stats(g)))
            .reset_index())


def bootstrap_diff(pred: pd.DataFrame, a: str, b: str, metric="ap", n=1000, seed=0) -> dict:
    """Year-block bootstrap of metric(a) - metric(b) on identical observations."""
    rng = np.random.default_rng(seed)
    wide = pred[pred.model.isin([a, b])].pivot_table(index=["obs_id", "fyear", "y"], columns="model",
                                                      values="p").reset_index()
    years = wide["fyear"].unique()
    groups = {y: wide[wide.fyear == y] for y in years}
    f = average_precision_score if metric == "ap" else roc_auc_score

    def wmetric(frames, col):
        vals = [(f(h.y, h[col]), h.y.sum()) for h in frames if 0 < h.y.sum() < len(h)]
        return np.average([v for v, _ in vals], weights=[n for _, n in vals])

    point = wmetric(groups.values(), a) - wmetric(groups.values(), b)
    diffs = []
    for _ in range(n):
        fr = [groups[y] for y in rng.choice(years, len(years))]
        if sum(h.y.sum() for h in fr) == 0:
            continue
        diffs.append(wmetric(fr, a) - wmetric(fr, b))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"comparison": f"{a} - {b}", "metric": metric, "diff": point, "ci_low": lo, "ci_high": hi,
            "p_le_0": float(np.mean(np.asarray(diffs) <= 0))}


def coverage_economics(pred: pd.DataFrame, panel: pd.DataFrame, k: int = 50,
                       models=("M1_logit_palepu", "M3_gbm_sic", MAIN)) -> pd.DataFrame:
    """Targets, deal value and fee pool captured by a K-name coverage list per year."""
    info = panel[["obs_id", "public_float", "assets_cur", "sic_deal_activity"]].copy()
    info["deal_value"] = info["public_float"].fillna(info["assets_cur"])
    info["fee"] = info["deal_value"] * fee_rate(info["deal_value"])
    base = pred[pred.model == MAIN][["obs_id", "fyear", "y"]].merge(info, on="obs_id")

    rules = {m: pred[pred.model == m][["obs_id", "p"]] for m in models}
    rules["Largest firms first"] = base[["obs_id"]].assign(p=base["deal_value"])
    rules["Hottest SIC industry"] = base[["obs_id"]].assign(p=base["sic_deal_activity"] + 1e-9 * base["deal_value"])

    rows = []
    for name, sc in rules.items():
        d = base.merge(sc, on="obs_id")
        pick = d.groupby("fyear", group_keys=False).apply(lambda g: g.nlargest(k, "p"))
        hit = pick[pick.y == 1]
        rows.append({"strategy": name, "K": k, "targets_hit": len(hit),
                     "hit_rate": len(hit) / len(pick),
                     "deal_value_bn": hit["deal_value"].sum() / 1e9,
                     "fee_pool_mn": hit["fee"].sum() / 1e6})
    # random list: expected value
    yrs = base.groupby("fyear")
    exp_hits = sum(k * g.y.mean() for _, g in yrs)
    exp_val = sum(k * (g.y * g.deal_value).sum() / len(g) for _, g in yrs)
    exp_fee = sum(k * (g.y * g.fee).sum() / len(g) for _, g in yrs)
    rows.append({"strategy": "Random list (expected)", "K": k, "targets_hit": exp_hits,
                 "hit_rate": exp_hits / (k * base.fyear.nunique()),
                 "deal_value_bn": exp_val / 1e9, "fee_pool_mn": exp_fee / 1e6})
    out = pd.DataFrame(rows)
    rnd = out.loc[out.strategy == "Random list (expected)"].iloc[0]
    out["hits_x_random"] = out["targets_hit"] / rnd["targets_hit"]
    out["fees_x_random"] = out["fee_pool_mn"] / rnd["fee_pool_mn"]
    return out


def robustness(pred: pd.DataFrame, panel: pd.DataFrame, events: pd.DataFrame,
               models=("M1_logit_palepu", "M3_gbm_sic", "M4_gbm_tfidf", MAIN)) -> pd.DataFrame:
    d = pred[pred.model.isin(models)].merge(
        panel[["obs_id", "cik", "date", "log_assets", "days_to_event"]], on="obs_id")
    d["size_tercile"] = d.groupby(["model", "fyear"])["log_assets"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["small", "mid", "large"]))
    # going-private (13E-3) targets: flag and drop for one robustness cut
    gp = events[events["forms"].str.contains("SC 13E3")][["cik", "event_date"]]
    d = d.merge(gp, on="cik", how="left")
    d["going_private"] = (d["y"] == 1) & d["event_date"].notna() & \
        ((d["event_date"] - d["date"]).dt.days.between(0, 500))
    cuts = {
        "Full sample": d,
        "2014-2016": d[d.fyear <= 2016],
        "2017-2020": d[d.fyear >= 2017],
        "Small firms": d[d.size_tercile == "small"],
        "Mid firms": d[d.size_tercile == "mid"],
        "Large firms": d[d.size_tercile == "large"],
        "Excl. going-private": d[~d.going_private],
    }
    rows = []
    for cut, g in cuts.items():
        for m, h in g.groupby("model"):
            s = {**stats(h), **within_year(h)}
            rows.append({"cut": cut, "model": m, "n": s["n"], "targets": s["targets"],
                         "auc": s["auc"], "ap": s["ap"], "lift_top10": s["lift_top10"]})
    return pd.DataFrame(rows)


def run() -> dict[str, pd.DataFrame]:
    P, T = config.PROCESSED, config.TABLES
    pred = pd.read_parquet(P / "oos_predictions.parquet")
    panel = pd.read_parquet(P / "panel.parquet")
    events = pd.read_parquet(P / "events.parquet")
    out = {
        "summary": summary_table(pred),
        "by_year": by_year(pred),
        "bootstrap": pd.DataFrame([
            bootstrap_diff(pred, MAIN, "M1_logit_palepu"),
            bootstrap_diff(pred, MAIN, "M3_gbm_sic"),
            bootstrap_diff(pred, "M4_gbm_tfidf", "M3_gbm_sic"),
            bootstrap_diff(pred, MAIN, "M4_gbm_tfidf"),
            bootstrap_diff(pred, MAIN, "M6_logit_all"),
        ]),
        "coverage_k50": coverage_economics(pred, panel, 50),
        "coverage_k100": coverage_economics(pred, panel, 100),
        "robustness": robustness(pred, panel, events),
    }
    for k, v in out.items():
        v.to_csv(T / f"{k}.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        for k in ("summary", "bootstrap", "coverage_k50"):
            print(f"\n--- {k} ---\n{out[k].to_string(index=False)}")
    return out


if __name__ == "__main__":
    run()
