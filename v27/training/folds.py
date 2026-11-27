"""Walk-forward cross-validation fold builder with dynamic purge and embargo."""

from __future__ import annotations

import numpy as np


def build_walk_forward_folds(
    decision_idx: np.ndarray,
    calendar: np.ndarray,
    test_start: str = "2010-01-01",
    test_end: str = "2024-12-31",
    n_folds: int = 14,
    purge: int = 2,
    embargo: int = 5,
    val_days: int = 400,
) -> list[dict[str, object]]:
    """Build chronological expanding train/val/test folds without label leakage."""
    d_dates = calendar[decision_idx]
    t0 = int(np.searchsorted(d_dates, np.datetime64(test_start, "D")))
    t1 = int(np.searchsorted(d_dates, np.datetime64(test_end, "D"), side="right"))
    folds: list[dict[str, object]] = []
    for i, test_pos in enumerate(np.array_split(np.arange(t0, t1), n_folds), 1):
        if len(test_pos) == 0:
            continue
        hist_end = int(test_pos[0]) - purge - embargo
        val_start = max(0, hist_end - val_days)
        train_end = val_start - purge
        folds.append(
            {
                "name": f"fold_{i:02d}",
                "train": decision_idx[:train_end],
                "val": decision_idx[val_start:hist_end],
                "test": decision_idx[test_pos],
            }
        )
    return folds
