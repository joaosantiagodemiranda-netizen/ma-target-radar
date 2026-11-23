"""Step 8 - Model ladder, tuned and evaluated out-of-sample year by year.

Each rung adds one information set, so the incremental value of each is isolated:

  M0  base rate                         (naive: every firm gets the same score)
  M1  Logit, Palepu (1986) variables    (classical benchmark)
  M2  LightGBM, fundamentals            (non-linearity)
  M3  + SIC industry deal activity      (classical industry-wave benchmark)
  M4  + TF-IDF peers + text signals     (classical NLP)
  M5  + transformer embeddings          (deep-learning text representation)
  M6  Logit with all M5 features        (is the gain from GBM or from features?)

Outputs: data/processed/oos_predictions.parquet, reports/tables/tuning.csv,
         data/processed/shap_<model>.parquet (last fold)
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import panel as F
from src.models import cv

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")

PALEPU = ["log_assets", "roa", "sales_growth", "leverage", "cash_assets", "mtb", "grd", "div_payer"]
N_PCA = 16
EMB_PCA = [f"emb_pc{i}" for i in range(N_PCA)]

LADDER = {
    "M1_logit_palepu": ("logit", PALEPU),
    "M2_gbm_fund":     ("gbm", F.FUND_FEATURES),
    "M3_gbm_sic":      ("gbm", F.FUND_FEATURES + F.SIC_FEATURES),
    "M4_gbm_tfidf":    ("gbm", F.FUND_FEATURES + F.SIC_FEATURES + F.TFIDF_FEATURES + F.SIGNAL_FEATURES),
    "M5_gbm_emb":      ("gbm", F.FUND_FEATURES + F.SIC_FEATURES + F.TFIDF_FEATURES + F.SIGNAL_FEATURES
                        + F.EMB_FEATURES + EMB_PCA),
    "M6_logit_all":    ("logit", F.FUND_FEATURES + F.SIC_FEATURES + F.TFIDF_FEATURES + F.SIGNAL_FEATURES
                        + F.EMB_FEATURES + EMB_PCA),
}

GBM_GRID = [dict(num_leaves=nl, min_child_samples=mc) for nl in (7, 15, 31) for mc in (50, 200)]
LOGIT_GRID = [0.01, 0.1, 1.0]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column at train-set 1st/99th percentiles (bounds learned in-fold)."""
    def __init__(self, q=(0.01, 0.99)):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_, self.hi_ = np.nanquantile(X, self.q[0], axis=0), np.nanquantile(X, self.q[1], axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)


def add_pca(p: pd.DataFrame, E: np.ndarray, tr: np.ndarray, te: np.ndarray) -> pd.DataFrame:
    """Embedding PCA fit on training rows only (no future text in the representation)."""
    X = E[p["doc_row"].astype(int).to_numpy()].astype(np.float32)
    pca = PCA(N_PCA, random_state=config.RANDOM_STATE).fit(X[tr])
    Z = pca.transform(X)
    return p.assign(**{c: Z[:, i] for i, c in enumerate(EMB_PCA)})


def fit_logit(Xtr, ytr, Xva, yva):
    best = None
    for C in LOGIT_GRID:
        m = make_pipeline(Winsorizer(), SimpleImputer(strategy="median", add_indicator=True),
                          StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        m.fit(Xtr, ytr)
        ap = average_precision_score(yva, m.predict_proba(Xva)[:, 1]) if yva.sum() else 0
        if best is None or ap > best[0]:
            best = (ap, C)
    return {"C": best[1]}, best[0]


def make_logit(params):
    return make_pipeline(Winsorizer(), SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(), LogisticRegression(C=params["C"], max_iter=2000))


def gbm(params, n_estimators=2000):
    return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.03, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=config.RANDOM_STATE, verbose=-1, **params)


def fit_gbm(Xtr, ytr, Xva, yva):
    best = None
    for g in GBM_GRID:
        m = gbm(g).fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        it = max(m.best_iteration_ or 100, 50)
        ap = average_precision_score(yva, m.predict_proba(Xva, num_iteration=it)[:, 1]) if yva.sum() else 0
        if best is None or ap > best[0]:
            best = (ap, {**g, "n_estimators": it})
    return best[1], best[0]


def run(p: pd.DataFrame | None = None, E: np.ndarray | None = None,
        models: dict = LADDER, shap_models=("M5_gbm_emb",)) -> pd.DataFrame:
    P = config.PROCESSED
    p = pd.read_parquet(P / "panel.parquet") if p is None else p
    E = np.load(P / "embeddings.npy") if E is None else E
    preds, tuning = [], []
    for year, tr, te in cv.expanding_splits(p):
        q = add_pca(p, E, tr, te)
        ptr = q.iloc[tr]
        fi, va = cv.inner_split(ptr)
        ytr_all, yte = ptr["y"].to_numpy(), q["y"].iloc[te].to_numpy()
        base = ytr_all.mean()
        preds.append(pd.DataFrame({"obs_id": q["obs_id"].iloc[te], "fyear": year, "y": yte,
                                   "model": "M0_base_rate", "p": base}))
        for name, (kind, cols) in models.items():
            X = q[cols].to_numpy(float)
            Xtr, Xte = X[tr], X[te]
            if kind == "logit":
                params, ap_val = fit_logit(Xtr[fi], ytr_all[fi], Xtr[va], ytr_all[va])
                m = make_logit(params).fit(Xtr, ytr_all)
            else:
                params, ap_val = fit_gbm(Xtr[fi], ytr_all[fi], Xtr[va], ytr_all[va])
                m = gbm({k: v for k, v in params.items() if k != "n_estimators"},
                        n_estimators=params["n_estimators"]).fit(Xtr, ytr_all)
            pte = m.predict_proba(Xte)[:, 1]
            preds.append(pd.DataFrame({"obs_id": q["obs_id"].iloc[te], "fyear": year, "y": yte,
                                       "model": name, "p": pte}))
            tuning.append({"test_year": year, "model": name, "n_train": len(tr),
                           "train_targets": int(ytr_all.sum()), "val_ap": ap_val, **params})
            if name in shap_models and year == max(config.TEST_YEARS):
                import shap
                sv = shap.TreeExplainer(m).shap_values(Xte)
                sv = sv[1] if isinstance(sv, list) else sv
                pd.DataFrame(sv, columns=cols).assign(obs_id=q["obs_id"].iloc[te].to_numpy()).to_parquet(
                    P / f"shap_{name}.parquet", index=False)
                pd.DataFrame(Xte, columns=cols).to_parquet(P / f"shap_X_{name}.parquet", index=False)
        print(f"  test {year}: train {len(tr):,} ({int(ytr_all.sum())} targets) | "
              f"test {len(te):,} ({int(yte.sum())} targets)")
    out = pd.concat(preds, ignore_index=True)
    out.to_parquet(P / "oos_predictions.parquet", index=False)
    pd.DataFrame(tuning).to_csv(config.TABLES / "tuning.csv", index=False)
    return out


if __name__ == "__main__":
    run()
