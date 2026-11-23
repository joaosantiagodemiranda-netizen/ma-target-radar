"""Step 4 - Company metadata (SIC industry, name, state) from SEC submissions.zip.

Note: SIC codes are the *current* EDGAR assignment, not point-in-time. SIC codes
rarely change; this limitation is stated in the manuscript.
"""
from __future__ import annotations

import json
import zipfile

import pandas as pd

from src import config
from src.data import sec_http

URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"


def build(ciks: set[int] | None = None) -> pd.DataFrame:
    zpath = sec_http.download(URL, config.RAW / "submissions.zip")
    rows = []
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if "-submissions-" in n or not n.startswith("CIK"):
                continue
            cik = int(n[3:13])
            if ciks is not None and cik not in ciks:
                continue
            d = json.loads(z.read(n))
            rows.append({
                "cik": cik, "name": d.get("name"),
                "sic": pd.to_numeric(d.get("sic"), errors="coerce"),
                "sic_desc": d.get("sicDescription"),
                "state_inc": d.get("stateOfIncorporation"),
                "entity_type": d.get("entityType"),
                "tickers": "|".join(d.get("tickers") or []),
                "exchanges": "|".join(x for x in (d.get("exchanges") or []) if x),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(config.PROCESSED / "companies.parquet", index=False)
    print(f"companies.parquet: {len(df):,} firms")
    return df


if __name__ == "__main__":
    build()
