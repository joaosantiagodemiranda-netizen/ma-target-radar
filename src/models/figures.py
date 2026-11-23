"""Step 10 - Manuscript figures (static PNG, 200 dpi).

Palette: validated categorical slots (blue, orange, aqua, yellow) in fixed order;
the naive baseline is always neutral grey (dashed), so identity never relies on
colour alone.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GREY = "#8f8e8a"
INK, INK2 = "#0b0b0b", "#52514e"
SHOW = {  # model -> (label, colour)
    "M0_base_rate": ("Naive (base rate)", GREY),
    "M1_logit_palepu": ("Palepu logit", SLOTS[0]),
    "M3_gbm_sic": ("GBM + SIC industry", SLOTS[1]),
    "M4_gbm_tfidf": ("GBM + TF-IDF text", SLOTS[2]),
    "M5_gbm_emb": ("GBM + embeddings", SLOTS[3]),
}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIGURES / name, dpi=200)
    plt.close(fig)


def coverage_curve(pred: pd.DataFrame, kmax: int = 300):
    """Average number of realised targets captured by a top-K list, per year."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ks = np.arange(1, kmax + 1)
    for m, (lab, col) in SHOW.items():
        g = pred[pred.model == m]
        if g.empty:
            continue
        curves = []
        for _, h in g.groupby("fyear"):
            if h["p"].nunique() == 1:
                curves.append(ks * h["y"].mean())
            else:
                curves.append(np.cumsum(h.sort_values("p", ascending=False)["y"].to_numpy()[:kmax]))
        c = np.mean([np.pad(x, (0, kmax - len(x)), mode="edge") for x in curves], axis=0)
        ax.plot(ks, c, color=col, label=lab, ls="--" if m == "M0_base_rate" else "-")
    ax.set_xlabel("Coverage list size K (companies per year, ranked by score)")
    ax.set_ylabel("Targets in list (avg. per year)")
    ax.set_xlim(0, kmax)
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, "fig1_coverage_curve.png")


def ap_by_year(by_year: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for m, (lab, col) in SHOW.items():
        g = by_year[by_year.model == m].sort_values("fyear")
        g = g[g.targets > 0]
        if g.empty:
            continue
        y = g["base_rate"] if m == "M0_base_rate" else g["ap"]
        ax.plot(g["fyear"], y, marker="o", ms=4, color=col, label=lab,
                ls="--" if m == "M0_base_rate" else "-")
    ax.set_ylabel("Average precision (out-of-sample)")
    ax.set_xlabel("Test year (10-K filing year)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "fig2_ap_by_year.png")


def shap_importance(name: str = "M5_gbm_emb", top: int = 15):
    P = config.PROCESSED
    try:
        sv = pd.read_parquet(P / f"shap_{name}.parquet").drop(columns="obs_id")
    except FileNotFoundError:
        return
    imp = sv.abs().mean().sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.barh(imp.index, imp.values, color=SLOTS[0], height=0.6)
    ax.set_xlabel("Mean |SHAP value| (log-odds), last test year")
    ax.grid(axis="y", visible=False)
    _save(fig, "fig3_shap_importance.png")
    # direction: SHAP vs feature value for the top 6
    X = pd.read_parquet(P / f"shap_X_{name}.parquet")
    feats = list(imp.index[::-1][:6])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    for ax, f in zip(axes.ravel(), feats):
        x = X[f].clip(*X[f].quantile([0.01, 0.99]))
        ax.scatter(x, sv[f], s=6, alpha=0.35, color=SLOTS[0], linewidths=0)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_title(f, fontsize=8, color=INK)
    fig.supylabel("SHAP value (log-odds)", fontsize=8, color=INK2)
    _save(fig, "fig4_shap_dependence.png")


def calibration(pred: pd.DataFrame, models=("M1_logit_palepu", "M5_gbm_emb")):
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    lim = 0
    for m in models:
        lab, col = SHOW[m]
        g = pred[pred.model == m].copy()
        g["bin"] = pd.qcut(g["p"].rank(method="first"), 10, labels=False)
        c = g.groupby("bin")[["p", "y"]].mean()
        ax.plot(c["p"], c["y"], marker="o", ms=4, color=col, label=lab)
        lim = max(lim, c.values.max())
    ax.plot([0, lim], [0, lim], color=GREY, lw=1, ls="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability (decile)")
    ax.set_ylabel("Realised takeover rate")
    ax.legend(fontsize=8)
    _save(fig, "fig5_calibration.png")


def sample_over_time(panel: pd.DataFrame):
    g = panel.groupby("fyear")["y"].agg(["size", "sum"])
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.bar(g.index, g["sum"], color=SLOTS[0], width=0.6)
    for x, (n, s) in g.iterrows():
        ax.annotate(f"{s/n:.1%}", (x, s), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, color=INK2)
    ax.set_ylabel("Completed targets (next 12m)")
    ax.set_xlabel("10-K filing year  (labels: share of firms acquired)")
    ax.grid(axis="x", visible=False)
    _save(fig, "fig0_targets_by_year.png")


def run():
    P, T = config.PROCESSED, config.TABLES
    pred = pd.read_parquet(P / "oos_predictions.parquet")
    sample_over_time(pd.read_parquet(P / "panel.parquet"))
    coverage_curve(pred)
    ap_by_year(pd.read_csv(T / "by_year.csv"))
    shap_importance()
    calibration(pred)
    print(f"figures -> {config.FIGURES}")


if __name__ == "__main__":
    run()
