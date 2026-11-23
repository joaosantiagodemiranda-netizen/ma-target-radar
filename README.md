# M&A Target Radar

**Does text-based product-market similarity improve the prediction of takeover targets?**
Capstone project, *AI for Finance* (Prof. Josep Gisbert), IE University, 2026. Author: João Santiago de Miranda.

An M&A advisory team can only cover a limited number of companies actively. This
project ranks every US-listed non-financial company by its probability of becoming
a completed takeover target within 12 months of its 10-K. It then measures the
economic value of that ranking as *coverage economics*: the targets, deal value
and sell-side fee pool captured by a top-K coverage list, compared with
practitioner rules of thumb.

## Approach

| Rung | Model | Information set |
|---|---|---|
| M0 | Base rate | none (naive) |
| M1 | Logistic regression | Palepu (1986) fundamentals |
| M2 | LightGBM | 18 point-in-time fundamentals |
| M3 | LightGBM | + SIC-industry deal activity (merger waves) |
| M4 | LightGBM | + TF-IDF product-market peers (Hoberg & Phillips-style) + text signals |
| M5 | LightGBM | + multilingual transformer embeddings of the 10-K business description |
| M6 | Logistic regression | all M5 features (separates feature value from model non-linearity) |

All data is free and public: SEC EDGAR index, XBRL company facts, SEC
submissions, and EDGAR-CORPUS for 10-K text. There is no WRDS, CRSP or SDC
dependency.

### Guarding against leakage and look-ahead bias
- **Point-in-time fundamentals.** Values come from the 10-K's own XBRL accession, never from later restatements.
- **Purged expanding-window validation.** A training observation is used only if its full label window closed before the test year starts.
- **Tuning inside the training window.** Hyperparameters are tuned on the last fully resolved training year.
- **Deals already in play are removed.** Observations are dropped if a deal filing comes within 90 days of the 10-K, or if the 10-K text discloses a signed merger agreement. This rule is applied regardless of outcome.
- **Neighbourhood features look backward only.** Deal-activity features count filings strictly before the prediction date. The target-likeness feature uses only deals whose completion was observable before the test year.
- **In-fold preprocessing.** Winsorisation bounds and embedding PCA are fitted on training folds only.

### Label without SDC
A firm counts as a **completed target** if it is named in a merger proxy,
information statement, tender-offer filing or 13E-3 schedule, *and* it stops
filing 10-Ks/10-Qs within 18 months. This separates targets from acquirers
(which keep filing) and from failed deals. See `src/data/labels.py`.

## Reproduce

```bash
pip install -r requirements.txt
export SEC_USER_AGENT="Your Name your@email.com"   # required by SEC

python run_data.py      # steps 1-6, ~2 h, GPU recommended (or notebooks/01_data_pipeline.ipynb on Colab)
python run_models.py    # steps 7-10, regenerates every table and figure
pytest -q               # unit tests + end-to-end test on synthetic data
```

Peer and buyer engine (Portuguese or English input):
```bash
python -m src.app.peer_engine "Companhia de concessões rodoviárias em São Paulo..." --size 1.5e9
```

## Repository map

```
src/config.py            every design parameter (sample, label rules, filters)
src/data/                1 index · 2 labels · 3 XBRL · 4 metadata · 5 10-K text
src/features/embed.py    6 TF-IDF neighbours + transformer embeddings
src/features/panel.py    7 firm-year panel, point-in-time features, label
src/models/cv.py         purged expanding-window splits
src/models/train.py      8 model ladder + tuning + SHAP
src/models/evaluate.py   9 statistical + economic evaluation, robustness
src/models/figures.py    10 manuscript figures
src/app/peer_engine.py   peer / strategic-buyer search
tests/                   unit tests, synthetic data generator, end-to-end test
```

## Data sources
- SEC EDGAR full-index: https://www.sec.gov/Archives/edgar/full-index/
- SEC XBRL company facts and submissions (bulk): https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- EDGAR-CORPUS, Loukas et al. (2021): https://huggingface.co/datasets/eloukas/edgar-corpus
- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

## Use of generative AI
Claude (Anthropic) was used to help write and test the code and to draft
documentation. All design choices, results and interpretation were reviewed and
are the author's responsibility.
