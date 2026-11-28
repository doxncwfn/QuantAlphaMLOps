"""Unit tests for A2 target_5d recomputation (ret_total + log1p delisting)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.fix_a2_target_5d import compute_target_5d_v2, compute_y_star


def test_target_5d_v2_uses_ret_total_and_leaves_tail_null() -> None:
    """5-day forward log total return includes dividends and leaves <5 remaining dates NaN."""
    dates = pd.bdate_range("2020-01-02", periods=7)
    df = pd.DataFrame(
        {
            "permno": [10001] * 7,
            "date": dates,
            "ret_price": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "ret_total": [0.01, 0.02, 0.01, 0.03, 0.01, 0.02, 0.01],
            "dlstcd": [np.nan] * 7,
            "dlret": [np.nan] * 7,
            "dlretx": [np.nan] * 7,
            "era": ["CRSP"] * 7,
        }
    )
    out = compute_target_5d_v2(df)
    assert np.isclose(out[0], 0.02 + 0.01 + 0.03 + 0.01 + 0.02, atol=1e-12)
    assert np.isclose(out[1], 0.01 + 0.03 + 0.01 + 0.02 + 0.01, atol=1e-12)
    assert np.isnan(out[2:]).all()


def test_target_5d_v2_compounds_delisting_via_log1p() -> None:
    """Delisting uses log1p(dlret) and stops forward accumulation at CRSP exit."""
    dates = pd.bdate_range("2008-09-10", periods=6)
    df = pd.DataFrame(
        {
            "permno": [80599] * 6,
            "date": dates,
            "ret_total": [-0.10, -0.50, -0.20, -2.50, 0.30, -0.80],
            "dlstcd": [np.nan, np.nan, np.nan, np.nan, np.nan, 574.0],
            "dlret": [np.nan, np.nan, np.nan, np.nan, np.nan, -0.60],
            "dlretx": [np.nan, np.nan, np.nan, np.nan, np.nan, -0.60],
            "era": ["CRSP"] * 6,
        }
    )
    out = compute_target_5d_v2(df)
    expected_log_dl = float(np.log1p(-0.60))
    assert np.isclose(out[0], -0.50 - 0.20 - 2.50 + 0.30 - 0.80, atol=1e-12)
    assert np.isclose(out[1], -0.20 - 2.50 + 0.30 - 0.80 + expected_log_dl, atol=1e-12)
    assert np.isclose(out[5], expected_log_dl, atol=1e-12)


def test_target_5d_v2_handles_crsp_to_yahoo_delisting_boundary() -> None:
    """CRSP delisting followed later by Yahoo ticker reuse stops at CRSP delisting."""
    dates = list(pd.bdate_range("2024-08-01", periods=3)) + [pd.Timestamp("2026-02-06")]
    df = pd.DataFrame(
        {
            "permno": [14554] * 4,
            "date": dates,
            "ret_total": [0.02, -0.10, -0.05, np.nan],
            "dlstcd": [np.nan, np.nan, 574.0, np.nan],
            "dlret": [np.nan, np.nan, -0.80, np.nan],
            "dlretx": [np.nan, np.nan, -0.80, np.nan],
            "era": ["CRSP", "CRSP", "CRSP", "Yahoo"],
        }
    )
    out = compute_target_5d_v2(df)
    log_dl = float(np.log1p(-0.80))
    assert np.isclose(out[0], -0.10 - 0.05 + log_dl, atol=1e-12)
    assert np.isclose(out[1], -0.05 + log_dl, atol=1e-12)
    assert np.isclose(out[2], log_dl, atol=1e-12)
    assert np.isnan(out[3])


def test_compute_y_star_ranking() -> None:
    """y_star ranks only valid in-universe rows per date into [-0.5, +0.5]."""
    dates = pd.Series([pd.Timestamp("2020-01-02")] * 4)
    in_univ = np.array([True, True, True, False])
    t5 = np.array([0.01, -0.02, 0.05, 0.99])
    ys = compute_y_star(dates, in_univ, t5)
    assert np.isclose(ys[0], 0.0)
    assert np.isclose(ys[1], -0.5)
    assert np.isclose(ys[2], 0.5)
    assert np.isnan(ys[3])
