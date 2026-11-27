"""Walk-forward cross-validation splits, metrics, and trainer modules for v27."""

from __future__ import annotations

from v27.training.folds import build_walk_forward_folds
from v27.training.metrics import daily_rank_ic, daily_tail_ic, daily_top_bot_hit
from v27.training.trainer import (
    cross_sectional_zscore,
    evaluate_fold_metrics,
    make_day_weights,
    predict_with_alpha,
    run_experiment,
    seed_all,
    train_one_fold,
)

__all__ = [
    "build_walk_forward_folds",
    "cross_sectional_zscore",
    "daily_rank_ic",
    "daily_tail_ic",
    "daily_top_bot_hit",
    "evaluate_fold_metrics",
    "make_day_weights",
    "predict_with_alpha",
    "run_experiment",
    "seed_all",
    "train_one_fold",
]
