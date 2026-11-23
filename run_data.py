"""Run the full data pipeline (Steps 1-6) and package the outputs.

    export SEC_USER_AGENT="Your Name your@email.com"
    python run_data.py                 # everything (GPU recommended for step 6)
    python run_data.py --steps 1 2     # selected steps

Produces data/processed/*.parquet + embeddings.npy and a single
data/processed_bundle.zip (~200-400 MB) with everything the modelling stage needs.
The raw Item 1 text (data/processed/text/) is excluded from the bundle.
"""
from __future__ import annotations

import argparse
import time
import zipfile

import pandas as pd

from src import config


def step(n: int, title: str):
    print(f"\n=== Step {n}: {title} ===")
    return time.time()


def main(steps: list[int]) -> None:
    if 1 in steps:
        t = step(1, "EDGAR filings index")
        from src.data import edgar_index; edgar_index.build()
        print(f"({time.time() - t:.0f}s)")
    if 2 in steps:
        t = step(2, "takeover target labels")
        from src.data import labels; labels.build()
        print(f"({time.time() - t:.0f}s)")
    if 3 in steps:
        t = step(3, "XBRL fundamentals")
        from src.data import xbrl; xbrl.build()
        print(f"({time.time() - t:.0f}s)")
    if 4 in steps:
        t = step(4, "company metadata")
        from src.data import submissions
        f = pd.read_parquet(config.PROCESSED / "filings.parquet")
        submissions.build(set(f.loc[f["form"].isin(config.ANNUAL_FORMS), "cik"]))
        print(f"({time.time() - t:.0f}s)")
    if 5 in steps:
        t = step(5, "10-K Item 1 text (streaming EDGAR-CORPUS)")
        from src.data import text; text.build()
        print(f"({time.time() - t:.0f}s)")
    if 6 in steps:
        t = step(6, "TF-IDF neighbours + transformer embeddings")
        from src.features import embed; embed.build()
        print(f"({time.time() - t:.0f}s)")
    bundle()


def bundle() -> None:
    out = config.ROOT / "data" / "processed_bundle.zip"
    files = [p for p in config.PROCESSED.iterdir() if p.is_file()]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in files:
            z.write(p, p.name)
    print(f"\nbundle: {out} ({out.stat().st_size / 1e6:.0f} MB, {len(files)} files)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6])
    main(ap.parse_args().steps)
