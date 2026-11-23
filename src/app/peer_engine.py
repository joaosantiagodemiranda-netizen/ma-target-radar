"""Peer & buyer engine: description (Portuguese or English) -> comparable companies,
likely strategic buyers and their current takeover score.

Uses the same multilingual embedding space as the prediction model, so a
Portuguese teaser for a Brazilian company is matched directly against US 10-K
business descriptions (cross-lingual retrieval; no translation step).

    python -m src.app.peer_engine "Operadora de rodovias com concessões em São Paulo..." --size 800e6
    python -m src.app.peer_engine --cik 320193

Buyers = text peers at least `buyer_multiple` x the target's size (strategic
acquirers are usually much larger), ranked by similarity x log size, with a flag
for firms observed as acquirers in tender offers / going-private deals.
"""
from __future__ import annotations

import argparse
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config


@lru_cache(1)
def universe(year: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    P = config.PROCESSED
    docs = pd.read_parquet(P / "doc_index.parquet").reset_index(drop=True)
    docs["doc_row"] = np.arange(len(docs))
    E = np.load(P / "embeddings.npy").astype(np.float32)
    year = year or int(docs["year"].max())
    latest = (docs[docs["year"] <= year].sort_values("year")
              .drop_duplicates("cik", keep="last"))
    # only firms still filing within 2 years of the reference year
    latest = latest[latest["year"] >= year - 1]
    comp = pd.read_parquet(P / "companies.parquet")[["cik", "name", "sic", "sic_desc", "tickers"]]
    fund = (pd.read_parquet(P / "fundamentals.parquet")
            .sort_values("filed").drop_duplicates("cik", keep="last")[["cik", "assets_cur", "revenue_cur", "public_float"]])
    desc = pd.read_parquet(P / "descriptions.parquet").sort_values("year").drop_duplicates("cik", keep="last")
    u = latest.merge(comp, on="cik", how="left").merge(fund, on="cik", how="left").merge(
        desc[["cik", "desc"]], on="cik", how="left")
    acq_path = P / "acquirers.parquet"
    if acq_path.exists():
        n_acq = pd.read_parquet(acq_path).groupby("acquirer_cik").size().rename("past_acquisitions")
        u = u.join(n_acq, on="cik")
    u["past_acquisitions"] = u.get("past_acquisitions", pd.Series(0, index=u.index)).fillna(0).astype(int)
    return u.reset_index(drop=True), E[u["doc_row"].to_numpy()]


@lru_cache(1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


def embed_text(text: str) -> np.ndarray:
    from src.features.embed import chunks
    v = _model().encode(chunks(" ".join(text.split())) or [text], normalize_embeddings=True).mean(0)
    return v / np.linalg.norm(v)


def search(query_vec: np.ndarray, size_usd: float | None = None, n_peers: int = 15,
           n_buyers: int = 10, buyer_multiple: float = 3.0, exclude_cik: int | None = None,
           scores: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    u, X = universe()
    sim = X @ query_vec.astype(np.float32)
    r = u.assign(similarity=sim)
    if exclude_cik is not None:
        r = r[r["cik"] != exclude_cik]
    if scores is not None:  # latest takeover probability from the model, if available
        r = r.merge(scores, on="cik", how="left")
    cols = ["name", "tickers", "sic_desc", "similarity", "assets_cur", "revenue_cur", "public_float",
            "past_acquisitions"] + (["takeover_prob"] if scores is not None else [])
    peers = r.nlargest(n_peers, "similarity")[cols]
    pool = r.nlargest(300, "similarity")
    if size_usd:
        pool = pool[pool["assets_cur"] >= buyer_multiple * size_usd]
    pool = pool.assign(buyer_score=pool["similarity"] * np.log1p(pool["assets_cur"].fillna(0) / 1e6)
                       * (1 + 0.1 * np.minimum(pool["past_acquisitions"], 5)))
    buyers = pool.nlargest(n_buyers, "buyer_score")[cols + ["buyer_score"]]
    return {"peers": peers.reset_index(drop=True), "buyers": buyers.reset_index(drop=True)}


def latest_scores(model: str = "M5_gbm_emb") -> pd.DataFrame | None:
    P = config.PROCESSED
    try:
        pred = pd.read_parquet(P / "oos_predictions.parquet")
        panel = pd.read_parquet(P / "panel.parquet", columns=["obs_id", "cik", "fyear"])
    except FileNotFoundError:
        return None
    s = pred[pred.model == model].merge(panel[["obs_id", "cik"]], on="obs_id")
    return s.sort_values("fyear").drop_duplicates("cik", keep="last")[["cik", "p"]].rename(
        columns={"p": "takeover_prob"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="company description (PT or EN)")
    ap.add_argument("--cik", type=int)
    ap.add_argument("--size", type=float, help="target size in USD (total assets)")
    a = ap.parse_args()
    if a.cik:
        u, X = universe()
        row = u.index[u["cik"] == a.cik]
        if not len(row):
            raise SystemExit(f"CIK {a.cik} not in universe")
        q = X[row[0]]
        size = a.size or u.loc[row[0], "assets_cur"]
    else:
        q, size = embed_text(a.text), a.size
    res = search(q, size, exclude_cik=a.cik, scores=latest_scores())
    for k in res:  # show money in US$ millions
        for c in ("assets_cur", "revenue_cur", "public_float"):
            res[k][c] = res[k][c] / 1e6
        res[k] = res[k].rename(columns={"assets_cur": "assets_usdm", "revenue_cur": "revenue_usdm",
                                        "public_float": "float_usdm"})
    with pd.option_context("display.width", 220, "display.max_colwidth", 40,
                           "display.float_format", "{:,.2f}".format):
        print("\nCLOSEST PEERS\n", res["peers"].to_string())
        print("\nLIKELY STRATEGIC BUYERS\n", res["buyers"].to_string())


if __name__ == "__main__":
    main()
