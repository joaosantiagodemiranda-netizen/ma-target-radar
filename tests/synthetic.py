"""Synthetic data with the exact schemas of the real pipeline outputs.

Used only to test the panel/model/evaluation code end-to-end without SEC access.
Takeover hazard is driven by known features (low ROA, high cash, sector deal
waves, text 'strategic alternatives') so tests can check the models learn signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make(out: Path, n_firms: int = 900, dim: int = 32, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    years = range(2009, 2021)
    sectors = rng.integers(0, 12, n_firms)
    sic = 2000 + sectors * 100 + rng.integers(0, 3, n_firms) * 10
    centroids = rng.normal(size=(12, dim))
    wave = {(s, y): rng.random() < 0.25 for s in range(12) for y in years}  # sector merger waves

    filings, events, fund, docs, emb = [], [], [], [], []
    alive = np.ones(n_firms, bool)
    quality = rng.normal(size=n_firms)              # persistent profitability
    cashy = rng.normal(size=n_firms)
    for y in years:
        for i in np.where(alive)[0]:
            cik = 1000 + i
            d = pd.Timestamp(f"{y}-03-01") + pd.Timedelta(days=int(rng.integers(0, 30)))
            acc = f"{cik:010d}-{y % 100:02d}-{1:06d}"
            filings.append((cik, f"F{cik}", "10-K", d, acc))
            for q in (5, 8, 11):
                filings.append((cik, f"F{cik}", "10-Q", pd.Timestamp(f"{y}-{q:02d}-10"), f"q{cik}{y}{q}"))
            A = float(np.exp(rng.normal(19, 1.5)))
            roa = 0.05 * quality[i] + rng.normal(0, 0.03)
            cash = float(np.clip(0.15 + 0.08 * cashy[i] + rng.normal(0, 0.03), 0.01, 0.9))
            strat = int(rng.random() < 0.05)
            fund.append(dict(accession=acc, cik=cik, filed=d, fy_end=pd.Timestamp(f"{y-1}-12-31"),
                             assets_cur=A, assets_prev=A / (1 + rng.normal(0.05, 0.1)),
                             liabilities_cur=A * rng.uniform(0.2, 0.8), equity_cur=A * 0.4,
                             cash_cur=A * cash, cur_assets_cur=A * 0.4, cur_liab_cur=A * 0.25,
                             lt_debt_cur=A * 0.2, goodwill_cur=A * 0.05,
                             revenue_cur=A * 0.8, revenue_prev=A * 0.75,
                             net_income_cur=A * roa, op_income_cur=A * (roa + 0.02),
                             da_cur=A * 0.03, rnd_cur=A * 0.02, cfo_cur=A * (roa + 0.03),
                             capex_cur=A * 0.04, dividends_cur=A * 0.01 * (quality[i] > 0),
                             public_float=A * np.exp(rng.normal(0, 0.4)), shares_out=1e8))
            v = centroids[sectors[i]] + rng.normal(0, 0.8, dim)
            emb.append(v / np.linalg.norm(v))
            docs.append(dict(cik=cik, year=y, filename=f"{cik}_10K_{y}_{acc}.htm", accession=acc,
                             n_words=3000, pending_deal=False, sig_strategic_alt=strat,
                             sig_activist=0, sig_going_concern=0, sig_change_control=1,
                             sig_consolidation=0))
            # takeover hazard
            logit = -3.6 - 0.8 * quality[i] + 0.6 * cashy[i] + 1.2 * wave[(sectors[i], y)] + 1.5 * strat
            if y < 2020 and rng.random() < 1 / (1 + np.exp(-logit)):
                ev = d + pd.Timedelta(days=int(rng.integers(100, 400)))
                filings.append((cik, f"F{cik}", "DEFM14A", ev, f"m{cik}"))
                alive[i] = False
                # firm stops filing; its last 10-Q is right after the event
                filings.append((cik, f"F{cik}", "10-Q", ev + pd.Timedelta(days=30), f"z{cik}"))
            elif rng.random() < 0.01:                     # acquirer-type proxy, keeps filing
                filings.append((cik, f"F{cik}", "DEFM14A", d + pd.Timedelta(days=120), f"a{cik}{y}"))
    # index extends to 2023 for survivors (outcome observation window)
    for y in (2021, 2022, 2023):
        for i in np.where(alive)[0]:
            filings.append((1000 + i, "x", "10-K", pd.Timestamp(f"{y}-03-01"), f"k{i}{y}"))
    F = pd.DataFrame(filings, columns=["cik", "name", "form", "date", "accession"])
    F.to_parquet(out / "filings.parquet", index=False)
    pd.DataFrame(fund).to_parquet(out / "fundamentals.parquet", index=False)
    D = pd.DataFrame(docs)
    D.to_parquet(out / "doc_index.parquet", index=False)
    E = np.vstack(emb).astype(np.float16)
    np.save(out / "embeddings.npy", E)
    pd.DataFrame({"cik": 1000 + np.arange(n_firms), "name": [f"F{i}" for i in range(n_firms)],
                  "sic": sic, "sic_desc": "x", "state_inc": "DE", "entity_type": "operating",
                  "tickers": "", "exchanges": ""}).to_parquet(out / "companies.parquet", index=False)
    pd.DataFrame({"cik": D.cik, "year": D.year, "desc": "synthetic firm"}).to_parquet(
        out / "descriptions.parquet", index=False)
    # TF-IDF neighbours: noisy version of embedding neighbours
    rows = []
    for y, g in D.groupby("year"):
        idx = g.index.to_numpy()
        S = E[idx].astype(np.float32) @ E[idx].astype(np.float32).T + rng.normal(0, 0.3, (len(idx), len(idx)))
        np.fill_diagonal(S, -9)
        top = np.argsort(-S, axis=1)[:, :20]
        for r, cols in enumerate(top):
            for c in cols:
                rows.append((y, g.cik.iloc[r], g.cik.iloc[c], float(S[r, c])))
    pd.DataFrame(rows, columns=["year", "cik", "nbr_cik", "sim"]).to_parquet(
        out / "tfidf_neighbors.parquet", index=False)

    from src.data import labels
    ev = labels.build_events(F)
    ev.to_parquet(out / "events.parquet", index=False)
    labels.build_acquirers(F, ev).to_parquet(out / "acquirers.parquet", index=False)
