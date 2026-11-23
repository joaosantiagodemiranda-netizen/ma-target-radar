"""Step 5 - 10-K Item 1 (Business description) from EDGAR-CORPUS (Loukas et al., 2021).

The corpus is ~40 GB, so we *stream* each year's JSONL file and keep only:
  - Item 1 (first TEXT_CHARS_KEEP characters, whitespace-normalised)
  - a 'pending deal' flag: Item 1 or Item 7 (MD&A) already discloses a signed
    merger agreement / tender offer. Such observations are dropped later for
    every firm, regardless of outcome, because the event is already public at
    the prediction date (look-ahead guard; no selection on the label).

Output: data/processed/text/item1_<year>.parquet  (cik, year, filename, accession,
        item1, n_words, pending_deal)
"""
from __future__ import annotations

import json
import re

import pandas as pd
import requests

from src import config

URL = "https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/{y}/{split}.jsonl"
SPLITS = ("train", "validate", "test")

PENDING_RE = re.compile(
    r"agreement and plan of merger"
    r"|entered into (?:a|an|the) (?:definitive )?(?:merger|acquisition|arrangement) agreement"
    r"|(?:pending|proposed) (?:merger|acquisition) of the company"
    r"|will (?:become|be) a wholly[- ]owned subsidiary of"
    r"|tender offer (?:to purchase|for) all (?:of the )?(?:outstanding|issued)",
    re.IGNORECASE,
)
# Public "in play" / distress signals a coverage banker watches (counted in Items 1, 1A, 7)
SIGNALS = {
    "sig_strategic_alt": re.compile(r"strategic alternatives|exploring (?:a )?(?:potential )?sale of the company", re.I),
    "sig_activist": re.compile(r"activist (?:share|stock)holders?|activist investors?|proxy contest", re.I),
    "sig_going_concern": re.compile(r"substantial doubt about (?:our|the company.s|its) ability to continue as a going concern", re.I),
    "sig_change_control": re.compile(r"change[- ]in[- ]control|change[- ]of[- ]control", re.I),
    "sig_consolidation": re.compile(r"industry (?:is )?consolidat|consolidation (?:in|of) (?:our|the) industry", re.I),
}
WS = re.compile(r"\s+")
ACC = re.compile(r"(\d{10}-\d{2}-\d{6})")


def clean(t: str | None) -> str:
    return WS.sub(" ", t or "").strip()


def process_record(d: dict) -> dict | None:
    item1 = clean(d.get("section_1"))
    if len(item1) < 500:                       # parsing failure / empty Item 1
        return None
    mdna = d.get("section_7") or ""
    scan = " ".join([d.get("section_1") or "", d.get("section_1A") or "", mdna])
    acc = ACC.search(d.get("filename", ""))
    return {
        "cik": int(d["cik"]),
        "year": int(d["year"]),
        "filename": d.get("filename"),
        "accession": acc.group(1) if acc else None,
        "item1": item1[: config.TEXT_CHARS_KEEP],
        "n_words": len(item1.split()),
        "pending_deal": bool(PENDING_RE.search(item1) or PENDING_RE.search(mdna)),
        **{k: len(rx.findall(scan)) for k, rx in SIGNALS.items()},
    }


def build_year(y: int) -> pd.DataFrame:
    out = config.PROCESSED / "text" / f"item1_{y}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"  cached: {out.name}")
        return pd.read_parquet(out)
    rows = []
    for split in SPLITS:
        with requests.get(URL.format(y=y, split=split), stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    rec = process_record(json.loads(line))
                    if rec:
                        rows.append(rec)
    df = pd.DataFrame(rows).drop_duplicates(["cik", "year", "filename"])
    df.to_parquet(out, index=False)
    print(f"  {y}: {len(df):,} 10-Ks, pending-deal flag {df.pending_deal.mean():.1%}")
    return df


def build(years=config.TEXT_YEARS) -> None:
    for y in years:
        build_year(y)


if __name__ == "__main__":
    build()
