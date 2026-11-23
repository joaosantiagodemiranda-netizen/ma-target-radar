"""Central configuration: sample window, paths, form lists and label rules.

Every script imports from here so that one change propagates everywhere and the
manuscript can cite a single source of truth for each design choice.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(os.environ.get("MTR_ROOT", Path(__file__).resolve().parents[1]))
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
for _p in (RAW, PROCESSED, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- SEC access
# SEC requires a descriptive User-Agent ("Name email"). Set it before running:
#   export SEC_USER_AGENT="Joao Miranda joao.miranda@student.ie.edu"
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
SEC_SLEEP = 0.12  # seconds between requests -> < 10 req/s (SEC fair-access limit)

# --------------------------------------------------------------------------- sample window
# 10-K text (EDGAR-CORPUS) ends with filings made in 2020; XBRL fundamentals are
# reliable from 2010 onwards. Observations = 10-Ks filed 2010-2020.
OBS_FIRST_YEAR = 2010
OBS_LAST_YEAR = 2020
TEXT_YEARS = list(range(2009, 2021))      # 2009 needed for year-on-year text change
INDEX_YEARS = list(range(2009, 2024))     # filings index; 2021-23 used to observe outcomes

# --------------------------------------------------------------------------- label rules
LABEL_HORIZON_DAYS = 365      # target if a deal filing occurs within 12 months of the 10-K
GUARD_DAYS = 90               # deals with first filing < 90d after the 10-K are likely
                              # already announced -> observation dropped (look-ahead guard)
DELIST_WINDOW_DAYS = 540      # deal counts as completed if periodic filings stop within 18m
EVENT_CLUSTER_DAYS = 365      # deal filings for the same firm within 1y = one event

# Filings that the *target* (subject company) of a takeover files or is named in.
MA_FORMS = {
    "PREM14A", "DEFM14A", "PREM14C", "DEFM14C",   # merger proxy / information statement
    "SC 14D9", "SC TO-T",                          # tender offer (target response / bidder)
    "SC 13E3",                                     # going-private transaction
}
PERIODIC_FORMS = {"10-K", "10-K405", "10-KSB", "10-KT", "10-Q", "10-QSB"}
ANNUAL_FORMS = {"10-K", "10-K405", "10-KSB", "10-KT"}
DEREG_FORMS = {"15-12B", "15-12G", "15-15D", "25-NSE", "25"}
INDEX_FORMS = MA_FORMS | PERIODIC_FORMS | DEREG_FORMS

# --------------------------------------------------------------------------- universe filters
# Financials (banks, insurers, REITs, SPACs = SIC 6770) are excluded: their balance
# sheets are not comparable and de-SPAC proxies would be mislabeled as targets.
EXCLUDED_SIC_RANGES = [(6000, 6999)]
MIN_ASSETS_USD = 10e6         # drop shells / micro filers

# --------------------------------------------------------------------------- text / embeddings
TEXT_CHARS_KEEP = 20_000      # first ~3,000 words of Item 1 (business description)
EMBED_MODEL = os.environ.get(
    "MTR_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)  # multilingual -> Portuguese teasers live in the same vector space (Brazil demo)
EMBED_CHUNK_WORDS = 120
EMBED_MAX_CHUNKS = 10
N_NEIGHBORS = 50

# --------------------------------------------------------------------------- modelling
TEST_YEARS = list(range(2014, 2021))   # expanding-window out-of-sample years
RANDOM_STATE = 42
