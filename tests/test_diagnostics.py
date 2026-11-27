"""Unit tests for diagnostic routines."""

import numpy as np

from src.training.diagnostics import (
    compute_ic_time_series,
    evaluate_turnover_sensitivity,
)


def test_compute_ic_time_series():
    T = 25
    N = 50
    np.random.seed(42)
    scores = np.random.randn(T, N)
    target = scores + 0.1 * np.random.randn(T, N)
    mask = np.ones((T, N), dtype=bool)
    dates = [f"2020-01-{i + 1:02d}" for i in range(T)]

    res = compute_ic_time_series(scores, target, mask, dates)
    assert len(res["per_date_ic"]) == T
    assert res["mean"] > 0.8
    assert len(res["rolling_20d"]) == T
    assert isinstance(res["autocorr_lag1"], float)


def test_evaluate_turnover_sensitivity():
    T = 20
    N = 40
    np.random.seed(42)
    scores = np.random.randn(T, N)
    target = np.random.randn(T, N)
    mask = np.ones((T, N), dtype=bool)
    permno = np.tile(np.arange(100, 100 + N), (T, 1))

    res = evaluate_turnover_sensitivity(scores, target, mask, permno)
    assert "smooth_alpha_1.0" in res
    assert "smooth_alpha_0.5" in res
    assert "rebalance_10d" in res
    assert "position_limit_cap_half" in res
    # Smoothing should reduce turnover
    assert res["smooth_alpha_0.5"]["turnover"] <= res["smooth_alpha_1.0"]["turnover"]
