"""Time splits for walk-forward validation.

Contract: a class whose split(points: np.ndarray) -> list[Fold]. points are the decision dates
in time order (the train stage passes their calendar positions; any sorted array works) and
each Fold holds subsets of them. Train, validation and test must be disjoint and ordered in
time, with gaps of at least the label horizon between them so no label overlaps the next segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qam.registry import register


@dataclass(frozen=True)
class Fold:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@register("splitter", "purged_walk_forward")
class PurgedWalkForward:
    """Expanding-window walk-forward (legacy PurgedWalkForwardSplitter) with a validation split.

    The span is cut into n_folds + 1 blocks; fold i tests on block i + 1. Before each test block:
        [ train ][ purge ][ val ][ purge + embargo ][ test ]
    val is the last val_frac of the pre-test history, so checkpoints are chosen without
    touching the test block (the legacy code selected checkpoints on the test fold).
    No burn-in is needed: the lookback window reads features from before the test block,
    which is legitimate input, never a label.
    """

    def __init__(self, n_folds: int = 5, purge: int = 5, embargo: int = 20, val_frac: float = 0.15):
        if not 0 < val_frac < 1:
            raise ValueError("val_frac must be in (0, 1)")
        self.n_folds, self.purge, self.embargo, self.val_frac = n_folds, purge, embargo, val_frac

    def split(self, dates: np.ndarray) -> list[Fold]:
        n = len(dates)
        block = n // (self.n_folds + 1)
        if block == 0:
            raise ValueError(f"{n} dates are too few for {self.n_folds} folds")
        folds = []
        for i in range(self.n_folds):
            test_start = n - (self.n_folds - i) * block
            test_end = n if i == self.n_folds - 1 else test_start + block
            history_end = test_start - self.purge - self.embargo
            val_start = history_end - int(history_end * self.val_frac)
            train_end = val_start - self.purge
            if train_end <= 0 or val_start >= history_end:
                raise ValueError(f"fold {i + 1}: not enough history before the test block")
            folds.append(
                Fold(
                    name=f"fold_{i + 1}",
                    train=dates[:train_end],
                    val=dates[val_start:history_end],
                    test=dates[test_start:test_end],
                )
            )
        return folds
