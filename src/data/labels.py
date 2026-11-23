"""Step 2 - Build takeover-target events from the filings index (no SDC needed).

Label rule (documented in the manuscript, Section 3.2):
  A firm is a *completed takeover target* at date t if
    (i)  it is associated with a deal filing at t (merger proxy, information
         statement, tender-offer filing or going-private schedule), and
    (ii) it stops filing periodic reports (10-K / 10-Q) within 18 months after t,
         i.e. it disappeared as a stand-alone registrant.
Condition (ii) separates targets from acquirers that also file merger proxies
(acquirers keep filing) and from failed deals (target keeps filing).

Outputs
  events.parquet      cik, event_date, forms, last_periodic, completed
  acquirers.parquet   target_cik, acquirer_cik, date   (tender offers / 13E-3 only)
"""
from __future__ import annotations

import pandas as pd

from src import config


def cluster_events(ma: pd.DataFrame, gap_days: int = config.EVENT_CLUSTER_DAYS) -> pd.DataFrame:
    """Collapse each firm's deal filings into events (first filing = event date)."""
    ma = ma.sort_values(["cik", "date"])
    rows = []
    for cik, g in ma.groupby("cik", sort=False):
        start, forms = None, set()
        for d, f in zip(g["date"], g["form"]):
            if start is None or (d - start).days > gap_days:
                if start is not None:
                    rows.append((cik, start, "|".join(sorted(forms))))
                start, forms = d, set()
            forms.add(f)
        rows.append((cik, start, "|".join(sorted(forms))))
    return pd.DataFrame(rows, columns=["cik", "event_date", "forms"])


def build_events(filings: pd.DataFrame) -> pd.DataFrame:
    ma = filings[filings["form"].isin(config.MA_FORMS)]
    ev = cluster_events(ma)
    last_per = (filings[filings["form"].isin(config.PERIODIC_FORMS)]
                .groupby("cik")["date"].max().rename("last_periodic"))
    ev = ev.join(last_per, on="cik")
    lag = (ev["last_periodic"] - ev["event_date"]).dt.days
    # stopped filing within the window, and was an active filer around the event
    ev["completed"] = lag.between(-180, config.DELIST_WINDOW_DAYS)
    # outcomes after this date cannot be verified with the index we downloaded
    index_end = filings["date"].max()
    ev["verifiable"] = ev["event_date"] <= index_end - pd.Timedelta(days=config.DELIST_WINDOW_DAYS)
    return ev


def build_acquirers(filings: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Counterparties sharing an accession with a completed target's deal filing."""
    deal = filings[filings["form"].isin({"SC TO-T", "SC 13E3", "SC 14D9"})]
    tgt = events.loc[events["completed"], ["cik", "event_date"]]
    tgt_acc = deal.merge(tgt, on="cik")
    tgt_acc = tgt_acc[(tgt_acc["date"] - tgt_acc["event_date"]).dt.days.between(0, 365)]
    pairs = (tgt_acc[["accession", "cik", "event_date"]]
             .merge(deal[["accession", "cik"]], on="accession", suffixes=("_t", "_a")))
    pairs = pairs[pairs["cik_t"] != pairs["cik_a"]]
    return (pairs.rename(columns={"cik_t": "target_cik", "cik_a": "acquirer_cik",
                                  "event_date": "date"})
            [["target_cik", "acquirer_cik", "date"]].drop_duplicates())


def build() -> pd.DataFrame:
    filings = pd.read_parquet(config.PROCESSED / "filings.parquet")
    ev = build_events(filings)
    ev.to_parquet(config.PROCESSED / "events.parquet", index=False)
    acq = build_acquirers(filings, ev)
    acq.to_parquet(config.PROCESSED / "acquirers.parquet", index=False)
    v = ev[ev["verifiable"]]
    print(f"events: {len(ev):,} | verifiable: {len(v):,} | completed targets: "
          f"{v['completed'].sum():,} | acquirer pairs: {len(acq):,}")
    print(v[v["completed"]].groupby(v["event_date"].dt.year).size().to_string())
    return ev


if __name__ == "__main__":
    build()
