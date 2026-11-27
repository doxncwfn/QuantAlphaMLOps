"""Condition-number guard verification suite per §3.5.

Verifies that when a 252-session factor window exhibits severe multicollinearity
(`smb = 2 * hml + 1e-13 * noise`, making `cond([1, X]) > 1e10`), both `compute_residuals`
and `compute_layer4_features` set `resid_ret` to NaN and propagate NaNs into
`resid_mom_20` and `resid_vol_20`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.compute_features import (
    FACTOR_COLS,
    compute_layer4_features,
    compute_residuals,
)


def test_condition_number_guard_triggers_nan() -> None:
    """Verify collinear 252-session factor window triggers NaN in resid_ret and cascades to mom/vol."""
    rng = np.random.RandomState(42)
    n_days = 280
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    mktrf = rng.normal(0.0005, 0.01, size=n_days)
    hml = rng.normal(0.0001, 0.008, size=n_days)
    # Construct near-exact collinearity: smb = 2 * hml + 1e-13 * noise -> cond([1, X]) > 1e10
    smb = 2.0 * hml + 1e-13 * rng.normal(0.0, 1.0, size=n_days)
    rmw = rng.normal(0.0002, 0.006, size=n_days)
    cma = rng.normal(0.0001, 0.005, size=n_days)
    umd = rng.normal(0.0003, 0.009, size=n_days)
    rf = np.full(n_days, 0.0001, dtype=np.float64)

    factors_df = pd.DataFrame(
        {
            "date": dates,
            "mktrf": mktrf,
            "smb": smb,
            "hml": hml,
            "rmw": rmw,
            "cma": cma,
            "umd": umd,
            "rf": rf,
        }
    )

    # Confirm condition number of [1, X] on [0:252] exceeds 1e10
    X_win = np.column_stack(
        [np.ones(252), factors_df.loc[:251, FACTOR_COLS].to_numpy()]
    )
    cond_val = float(np.linalg.cond(X_win))
    assert cond_val > 1e10, (
        f"Expected synthetic condition number > 1e10, got {cond_val:.3e}"
    )

    # Construct valid stock return series
    ret_total = rng.normal(0.0006, 0.015, size=n_days)
    stock_excess = pd.Series(ret_total - rf, index=dates)

    # 1. Test single-stock reference implementation compute_residuals()
    res_ref = compute_residuals(stock_excess, factors_df.set_index("date")[FACTOR_COLS])
    assert res_ref["epsilon"].iloc[252:].isna().all(), (
        "compute_residuals did not set epsilon to NaN under cond([1, X]) > 1e10"
    )
    assert res_ref["resid_mom_20"].iloc[252:].isna().all()
    assert res_ref["resid_vol_20"].iloc[252:].isna().all()

    # 2. Test panel solver compute_layer4_features()
    stock_df = pd.DataFrame(
        {
            "permno": np.full(n_days, 10001, dtype=np.int64),
            "date": dates,
            "ret_total": ret_total,
        }
    )
    panel_out = compute_layer4_features(stock_df, factors_df)
    assert panel_out["resid_ret"].iloc[252:].isna().all(), (
        "compute_layer4_features did not set resid_ret to NaN under cond([1, X]) > 1e10"
    )
    assert panel_out["resid_mom_20"].iloc[252:].isna().all(), (
        "resid_mom_20 did not propagate NaN when resid_ret is NaN"
    )
    assert panel_out["resid_vol_20"].iloc[252:].isna().all(), (
        "resid_vol_20 did not propagate NaN when resid_ret is NaN"
    )
