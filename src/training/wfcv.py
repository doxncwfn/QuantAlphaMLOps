"""Purged Non-overlapping Walk-Forward Cross-Validation (NF-WFCV) per v1.3 §9.5."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FoldSplit:
    """Container for a single walk-forward cross-validation fold partition."""

    fold_idx: int
    train_dates: np.ndarray
    val_dates: np.ndarray
    test_dates: np.ndarray

    @property
    def train_start(self) -> np.datetime64:
        return np.datetime64(self.train_dates[0])

    @property
    def train_end(self) -> np.datetime64:
        return np.datetime64(self.train_dates[-1])

    @property
    def val_start(self) -> np.datetime64:
        return np.datetime64(self.val_dates[0])

    @property
    def val_end(self) -> np.datetime64:
        return np.datetime64(self.val_dates[-1])

    @property
    def test_start(self) -> np.datetime64:
        return np.datetime64(self.test_dates[0])

    @property
    def test_end(self) -> np.datetime64:
        return np.datetime64(self.test_dates[-1])


def purged_walk_forward_cv(
    dates: np.ndarray | Sequence,
    n_splits: int = 5,
    horizon: int = 5,
    embargo: int = 5,
    val_ratio: float = 0.15,
) -> Iterator[FoldSplit]:
    """Purged Walk-Forward Cross-Validation protocol with embargo per v1.3 §9.5.

    Derivation: A training sample at t uses forward target information up to t+horizon.
    To prevent overlap and feature serial correlation with the test set starting at
    test_start, the last valid training date is: test_start - (horizon + embargo).

    Parameters
    ----------
    dates : np.ndarray | Sequence
        Collection of unique evaluation trading dates.
    n_splits : int
        Number of equal-sized blocks (default 5, yielding 4 expanding folds).
    horizon : int
        Forward return horizon in trading sessions (5 sessions).
    embargo : int
        Embargo buffer in trading sessions to prevent serial leakage (5 sessions).
    val_ratio : float
        Fraction of pre-test training window reserved for validation (0.15).

    Yields
    ------
    FoldSplit
        Data splits for each fold k in [1, n_splits-1].
    """
    unique_dates = np.sort(np.unique(dates))
    n_dates = len(unique_dates)
    fold_size = n_dates // n_splits
    purge_gap = horizon + embargo  # 10 sessions

    for k in range(1, n_splits):
        test_start_idx = k * fold_size
        test_end_idx = min((k + 1) * fold_size, n_dates)

        # Enforce purge and embargo before test set
        last_train_idx = test_start_idx - purge_gap
        if last_train_idx <= 0:
            continue

        # Available training dates prior to test purge
        available_train_dates = unique_dates[: last_train_idx + 1]
        val_size = int(len(available_train_dates) * val_ratio)
        val_start_idx = len(available_train_dates) - val_size

        # Enforce purge and embargo before validation set
        last_subtrain_idx = val_start_idx - purge_gap
        if last_subtrain_idx <= 0:
            continue

        train_dates = unique_dates[: last_subtrain_idx + 1]
        val_dates = available_train_dates[val_start_idx:]
        test_dates = unique_dates[test_start_idx:test_end_idx]

        # Invariant checks: strictly non-overlapping and gapped
        assert len(train_dates) > 0, f"Fold {k}: empty train set"
        assert len(val_dates) > 0, f"Fold {k}: empty val set"
        assert len(test_dates) > 0, f"Fold {k}: empty test set"

        yield FoldSplit(
            fold_idx=k,
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=test_dates,
        )
