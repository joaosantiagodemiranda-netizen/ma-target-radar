"""Step 3 - Point-in-time fundamentals from SEC XBRL company facts.

Key design choice (look-ahead bias): every variable for a firm-year is taken from
the facts *reported inside that 10-K's own accession number*. Values therefore
reflect exactly what an analyst could read on the filing date - later
restatements are never used. Prior-year values come from the comparative
columns of the same filing, which also gives growth rates without look-ahead.

Output fundamentals.parquet: one row per 10-K accession with <var>_cur, <var>_prev.
"""
from __future__ import annotations

import json
import zipfile
from datetime import date

import pandas as pd

from src import config
from src.data import sec_http

URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"

# variable -> (taxonomy, [tags in priority order], kind)
TAGS: dict[str, tuple[str, list[str], str]] = {
    "assets":      ("us-gaap", ["Assets"], "instant"),
    "liabilities": ("us-gaap", ["Liabilities"], "instant"),
    "equity":      ("us-gaap", ["StockholdersEquity",
                                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "instant"),
    "cash":        ("us-gaap", ["CashAndCashEquivalentsAtCarryingValue",
                                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"], "instant"),
    "cur_assets":  ("us-gaap", ["AssetsCurrent"], "instant"),
    "cur_liab":    ("us-gaap", ["LiabilitiesCurrent"], "instant"),
    "lt_debt":     ("us-gaap", ["LongTermDebtNoncurrent", "LongTermDebt",
                                "LongTermDebtAndCapitalLeaseObligations"], "instant"),
    "goodwill":    ("us-gaap", ["Goodwill"], "instant"),
    "revenue":     ("us-gaap", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
                                "SalesRevenueGoodsNet"], "flow"),
    "net_income":  ("us-gaap", ["NetIncomeLoss", "ProfitLoss"], "flow"),
    "op_income":   ("us-gaap", ["OperatingIncomeLoss"], "flow"),
    "da":          ("us-gaap", ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                                "DepreciationAmortizationAndAccretionNet"], "flow"),
    "rnd":         ("us-gaap", ["ResearchAndDevelopmentExpense"], "flow"),
    "cfo":         ("us-gaap", ["NetCashProvidedByUsedInOperatingActivities"], "flow"),
    "capex":       ("us-gaap", ["PaymentsToAcquirePropertyPlantAndEquipment"], "flow"),
    "dividends":   ("us-gaap", ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"], "flow"),
    "public_float": ("dei", ["EntityPublicFloat"], "cover"),
    "shares_out":  ("dei", ["EntityCommonStockSharesOutstanding"], "cover"),
}


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(a) - date.fromisoformat(b)).days


def extract_company(facts_json: dict) -> list[dict]:
    """Return one dict per 10-K accession with current and prior-year values."""
    cik = int(facts_json.get("cik", 0))
    facts = facts_json.get("facts", {})
    per_accn: dict[str, dict] = {}

    for var, (tax, tags, kind) in TAGS.items():
        done_accn: set[str] = set()          # first tag in priority order wins
        for tag in tags:
            units = facts.get(tax, {}).get(tag, {}).get("units", {})
            obs = units.get("USD") or units.get("shares") or []
            by_accn: dict[str, list[dict]] = {}
            for o in obs:
                form = o.get("form", "")
                if not form.startswith("10-K") or form.endswith("/A"):
                    continue
                if kind == "flow":
                    if "start" not in o or not 330 <= _days(o["end"], o["start"]) <= 400:
                        continue
                by_accn.setdefault(o["accn"], []).append(o)
            for accn, lst in by_accn.items():
                if accn in done_accn:
                    continue
                lst.sort(key=lambda o: o["end"], reverse=True)
                cur = lst[0]
                rec = per_accn.setdefault(accn, {"cik": cik, "accession": accn,
                                                 "filed": cur.get("filed")})
                if kind == "cover":
                    rec[var] = cur["val"]
                else:
                    rec[f"{var}_cur"] = cur["val"]
                    # comparative (prior fiscal year) column of the same filing
                    prev = next((o for o in lst[1:] if 300 <= _days(cur["end"], o["end"]) <= 430), None)
                    rec[f"{var}_prev"] = prev["val"] if prev else None
                    if var == "assets":
                        rec["fy_end"] = cur["end"]
                done_accn.add(accn)
    return list(per_accn.values())


def build() -> pd.DataFrame:
    zpath = sec_http.download(URL, config.RAW / "companyfacts.zip")
    rows: list[dict] = []
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
        for i, n in enumerate(names):
            try:
                rows.extend(extract_company(json.loads(z.read(n))))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if i % 2000 == 0:
                print(f"  {i:,}/{len(names):,} companies, {len(rows):,} 10-K records")
    df = pd.DataFrame(rows)
    df["filed"] = pd.to_datetime(df["filed"])
    df["fy_end"] = pd.to_datetime(df["fy_end"])
    df.to_parquet(config.PROCESSED / "fundamentals.parquet", index=False)
    print(f"fundamentals.parquet: {len(df):,} 10-K accessions, {df.cik.nunique():,} firms")
    return df


if __name__ == "__main__":
    build()
