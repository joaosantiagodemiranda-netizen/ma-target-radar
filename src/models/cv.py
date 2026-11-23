"""Purged expanding-window validation for an event label with a 12-month horizon.

For test year Y (10-Ks filed during Y), the training set contains only
observations whose *entire* label window closed before 1 Jan Y:
    filing date + GUARD_DAYS + LABEL_HORIZON_DAYS < 1 Jan Y.
Without this purge, a 10-K filed in mid Y-1 would carry a label that resolves
inside year Y - a subtle look-ahead leak that inflates out-of-sample scores.

Inner validation (hyperparameter tuning / early stopping) uses the most recent
fully-resolved training year, so tuning never sees the test year either.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from src import config

LABEL_SPAN = pd.Timedelta(days=config.GUARD_DAYS + config.LABEL_HORIZON_DAYS)


def resolved_before(dates: pd.Series, cutoff: pd.Timestamp) -> np.ndarray:
    return (dates + LABEL_SPAN < cutoff).to_numpy()


def expanding_splits(p: pd.DataFrame, test_years=config.TEST_YEARS) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    for y in test_years:
        cutoff = pd.Timestamp(f"{y}-01-01")
        train = resolved_before(p["date"], cutoff)
        test = (p["fyear"] == y).to_numpy()
        if train.sum() and test.sum():
            yield y, np.where(train)[0], np.where(test)[0]


def inner_split(p_train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Last training year with fully resolved labels = validation; earlier ones = fit."""
    last = p_train["fyear"].max()
    cutoff = pd.Timestamp(f"{last}-01-01")
    fit = resolved_before(p_train["date"], cutoff)
    val = (p_train["fyear"] == last).to_numpy()
    return np.where(fit)[0], np.where(val)[0]
