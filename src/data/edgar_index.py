"""Step 1 - EDGAR quarterly master index -> filings.parquet.

The master index lists every filing (CIK | company | form | date | path). A filing
associated with several entities (e.g. a tender offer: bidder + subject company)
appears once per CIK, which is what lets us recover both sides of a deal.

Output columns: cik, name, form, date, accession
"""
from __future__ import annotations

import io

import pandas as pd

from src import config
from src.data import sec_http

URL = "https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/master.idx"


def parse_master_idx(text: str, keep_forms: set[str] | None = None) -> pd.DataFrame:
    """Parse the pipe-delimited body of a master.idx file."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("CIK|Company Name"))
    body = "\n".join(l for l in lines[start + 2:] if l.count("|") == 4)  # skip dashed line
    df = pd.read_csv(
        io.StringIO(body), sep="|", header=None, dtype=str,
        names=["cik", "name", "form", "date", "path"], quoting=3,
    )
    if keep_forms is not None:
        df = df[df["form"].isin(keep_forms)]
    df["cik"] = df["cik"].astype(int)
    df["date"] = pd.to_datetime(df["date"])
    # edgar/data/320193/0000320193-20-000096.txt -> 0000320193-20-000096
    df["accession"] = df["path"].str.extract(r"(\d{10}-\d{2}-\d{6})", expand=False)
    return df.drop(columns="path")


def build(years=config.INDEX_YEARS) -> pd.DataFrame:
    out = config.PROCESSED / "filings.parquet"
    frames = []
    for y in years:
        for q in (1, 2, 3, 4):
            cache = config.RAW / "index" / f"master_{y}Q{q}.parquet"
            cache.parent.mkdir(parents=True, exist_ok=True)
            if cache.exists():
                frames.append(pd.read_parquet(cache))
                continue
            text = sec_http.get(URL.format(y=y, q=q)).content.decode("latin-1")
            df = parse_master_idx(text, config.INDEX_FORMS)
            df.to_parquet(cache, index=False)
            frames.append(df)
            print(f"  {y}Q{q}: {len(df):,} relevant filings")
    filings = pd.concat(frames, ignore_index=True).drop_duplicates()
    filings.to_parquet(out, index=False)
    print(f"filings.parquet: {len(filings):,} rows, {filings.cik.nunique():,} CIKs")
    return filings


if __name__ == "__main__":
    build()
