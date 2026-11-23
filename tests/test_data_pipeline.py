"""Unit tests for the data layer on synthetic fixtures (run: pytest -q)."""
import numpy as np
import pandas as pd

from src.data import edgar_index, labels, text, xbrl
from src.features import embed

MASTER = """Description:           Master Index of EDGAR Dissemination Feed
Last Data Received:    March 31, 2015

CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
100|TARGET CO|10-K|2015-03-01|edgar/data/100/0000000100-15-000001.txt
100|TARGET CO|SC 14D9|2015-06-10|edgar/data/100/0000000200-15-000009.txt
200|BIDDER INC|SC TO-T|2015-06-01|edgar/data/200/0000000200-15-000005.txt
100|TARGET CO|SC TO-T|2015-06-01|edgar/data/100/0000000200-15-000005.txt
300|ACQUIRER CORP|DEFM14A|2015-05-01|edgar/data/300/0000000300-15-000002.txt
999|IGNORED|8-K|2015-05-01|edgar/data/999/0000000999-15-000002.txt
"""


def _filings():
    f = edgar_index.parse_master_idx(MASTER, keep_forms={"10-K", "10-Q", "SC 14D9", "SC TO-T", "DEFM14A"})
    extra = pd.DataFrame({
        "cik": [100, 300, 300, 200], "name": ["T", "A", "A", "B"],
        "form": ["10-Q", "10-Q", "10-K", "10-K"],
        "date": pd.to_datetime(["2015-08-01", "2017-05-01", "2018-03-01", "2019-03-01"]),
        "accession": ["a1", "a2", "a3", "a4"]})
    return pd.concat([f, extra], ignore_index=True)


def test_parse_master_idx():
    f = edgar_index.parse_master_idx(MASTER, keep_forms={"10-K", "SC 14D9", "SC TO-T", "DEFM14A"})
    assert len(f) == 5 and 999 not in set(f.cik)
    assert f.loc[f.form == "10-K", "accession"].item() == "0000000100-15-000001"


def test_labels_separate_target_from_acquirer():
    ev = labels.build_events(_filings())
    t = ev.set_index("cik")
    assert bool(t.loc[100, "completed"]) is True          # stops filing after tender offer
    assert bool(t.loc[300, "completed"]) is False         # acquirer keeps filing
    assert t.loc[100, "event_date"] == pd.Timestamp("2015-06-01")
    assert t.loc[100, "forms"] == "SC 14D9|SC TO-T"       # clustered into one event


def test_acquirer_pairs():
    f = _filings()
    acq = labels.build_acquirers(f, labels.build_events(f))
    assert (acq.target_cik == 100).all() and set(acq.acquirer_cik) == {200}


def test_cluster_gap():
    ma = pd.DataFrame({"cik": [1, 1, 1], "form": ["PREM14A", "DEFM14A", "PREM14A"],
                       "date": pd.to_datetime(["2012-01-01", "2012-02-01", "2014-01-01"])})
    ev = labels.cluster_events(ma)
    assert len(ev) == 2


def test_xbrl_point_in_time():
    js = {"cik": 100, "facts": {
        "us-gaap": {
            "Assets": {"units": {"USD": [
                {"end": "2014-12-31", "val": 500, "accn": "A15", "form": "10-K", "filed": "2015-03-01"},
                {"end": "2013-12-31", "val": 400, "accn": "A15", "form": "10-K", "filed": "2015-03-01"},
                {"end": "2014-12-31", "val": 999, "accn": "A16", "form": "10-K", "filed": "2016-03-01"},  # restated later
                {"end": "2014-12-31", "val": 777, "accn": "Q", "form": "10-K/A", "filed": "2015-06-01"}]}},
            "Revenues": {"units": {"USD": [
                {"start": "2014-01-01", "end": "2014-12-31", "val": 100, "accn": "A15", "form": "10-K", "filed": "2015-03-01"},
                {"start": "2014-10-01", "end": "2014-12-31", "val": 30, "accn": "A15", "form": "10-K", "filed": "2015-03-01"},
                {"start": "2013-01-01", "end": "2013-12-31", "val": 80, "accn": "A15", "form": "10-K", "filed": "2015-03-01"}]}},
            "SalesRevenueNet": {"units": {"USD": [
                {"start": "2014-01-01", "end": "2014-12-31", "val": -1, "accn": "A15", "form": "10-K", "filed": "2015-03-01"}]}},
        },
        "dei": {"EntityPublicFloat": {"units": {"USD": [
            {"end": "2014-06-30", "val": 1000, "accn": "A15", "form": "10-K", "filed": "2015-03-01"}]}}}}}
    recs = {r["accession"]: r for r in xbrl.extract_company(js)}
    r = recs["A15"]
    assert r["assets_cur"] == 500 and r["assets_prev"] == 400       # own filing, not restated 999
    assert r["revenue_cur"] == 100 and r["revenue_prev"] == 80      # quarterly 30 ignored, priority tag wins
    assert r["public_float"] == 1000
    assert "Q" not in recs                                          # amendments ignored


def test_text_record_and_pending_flag():
    d = {"cik": "100", "year": "2015", "filename": "100_10K_2015_0000000100-15-000001.htm",
         "section_1": "We make widgets.  " * 100,
         "section_7": "On May 1 we entered into an Agreement and Plan of Merger with Big Co."}
    r = text.process_record(d)
    assert r["accession"] == "0000000100-15-000001" and r["pending_deal"]
    d["section_7"] = "Revenue grew 5%."
    assert not text.process_record(d)["pending_deal"]
    assert text.process_record({**d, "section_1": "short"}) is None


def test_chunks_and_tfidf():
    assert len(embed.chunks("w " * 5000)) == 10
    rng = np.random.default_rng(0)
    vocab = ["software", "cloud", "banking", "mining", "copper", "gold", "retail", "stores",
             "apparel", "pharma", "drug", "clinical", "oil", "gas", "drilling"]
    docs = pd.DataFrame({
        "year": 2015, "cik": range(60),
        "item1": [" ".join(rng.choice(vocab[(i % 5) * 3:(i % 5) * 3 + 3], 200)) + " " +
                  " ".join(rng.choice(vocab, 1)) for i in range(60)]})
    nb = embed.tfidf_neighbors(docs, k=5)
    same = (nb.cik % 5 == nb.nbr_cik % 5).mean()
    assert same > 0.9 and (nb.cik != nb.nbr_cik).all()


def test_signal_counts():
    d = {"cik": "1", "year": "2016", "filename": "x", "section_1": "We sell tools. " * 60,
         "section_1A": "An activist stockholder may launch a proxy contest.",
         "section_7": "The Board is reviewing strategic alternatives."}
    r = text.process_record(d)
    assert r["sig_activist"] == 2 and r["sig_strategic_alt"] == 1 and r["sig_going_concern"] == 0
