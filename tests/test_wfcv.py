"""Tests for Purged Walk-Forward Cross-Validation (NF-WFCV) protocol and leakage guards."""

from __future__ import annotations

import pandas as pd

from src.training.wfcv import purged_walk_forward_cv


def test_purged_wfcv_expanding_folds_and_leakage_embargo() -> None:
    """Verify that:

    1. NF-WFCV produces 4 expanding folds for n_splits=5.
    2. Zero set overlap between train, val, and test partitions in every fold.
    3. Purge + Embargo gap >= 10 sessions between train and val.
    4. Purge + Embargo gap >= 10 sessions between val and test.
    5. Training size strictly expands across successive folds.
    """
    n_days = 2500
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    splits = list(
        purged_walk_forward_cv(dates, n_splits=5, horizon=5, embargo=5, val_ratio=0.15)
    )

    assert len(splits) == 4, f"Expected 4 folds, got {len(splits)}"

    prev_train_len = 0
    date_to_pos = {d: i for i, d in enumerate(dates)}

    for s in splits:
        assert s.fold_idx in (1, 2, 3, 4)

        # 1. Zero set overlap
        train_s = set(s.train_dates)
        val_s = set(s.val_dates)
        test_s = set(s.test_dates)

        assert len(train_s.intersection(val_s)) == 0, (
            f"Fold {s.fold_idx}: train and val overlap"
        )
        assert len(train_s.intersection(test_s)) == 0, (
            f"Fold {s.fold_idx}: train and test overlap"
        )
        assert len(val_s.intersection(test_s)) == 0, (
            f"Fold {s.fold_idx}: val and test overlap"
        )

        # 2. Strict temporal ordering and purge + embargo gap (>= 10 sessions)
        train_end_pos = date_to_pos[s.train_end]
        val_start_pos = date_to_pos[s.val_start]
        val_end_pos = date_to_pos[s.val_end]
        test_start_pos = date_to_pos[s.test_start]

        train_val_gap = val_start_pos - train_end_pos
        val_test_gap = test_start_pos - val_end_pos

        assert train_val_gap >= 10, (
            f"Fold {s.fold_idx}: train-to-val gap is {train_val_gap} < 10"
        )
        assert val_test_gap >= 10, (
            f"Fold {s.fold_idx}: val-to-test gap is {val_test_gap} < 10"
        )

        # 3. Expanding training set
        assert len(s.train_dates) > prev_train_len, (
            f"Fold {s.fold_idx}: train set did not expand"
        )
        prev_train_len = len(s.train_dates)

    # Test boundary matches the final date
    assert splits[-1].test_end == dates[-1]
