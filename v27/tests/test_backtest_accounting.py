"""Tests for backtest simulator accounting, HAC Sharpe, Ledoit-Wolf z-test, and DD episodes."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from v27.backtest.metrics import compounded_dd_episodes, lo_hac_sharpe
from v27.backtest.simulator import backtest_weights_daily
from v27.backtest.statistics import circular_block_bootstrap, ledoit_wolf_sharpe_test


def test_backtest_daily_mtm_and_allin_accounting():
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    W = np.array([[0.5, 0.5, -0.5, -0.5]] * 6)
    r = np.zeros((6, 4))
    adv = np.full((6, 4), 20_000_000.0)
    sig = np.full((6, 4), 0.02)
    rf = np.full(6, 0.0001)
    bt = backtest_weights_daily(
        W,
        r,
        dates,
        cost_bps=2.0,
        adv20_np=adv,
        sigma_np=sig,
        rf_np=rf,
        inception_zero_trade=True,
    )
    np.testing.assert_allclose(bt["gross"], 0.0, atol=1e-12)
    np.testing.assert_allclose(bt["traded"], 0.0, atol=1e-12)
    # Borrow on 1.0 short leg = 0.005 / 252
    assert math.isclose(float(bt["borrow"].iloc[0]), 0.005 / 252.0, rel_tol=1e-9)
    assert float(bt["net_allin"].iloc[0]) < float(bt["net"].iloc[0])


def test_drawdown_episodes_and_lw_test():
    r = pd.Series([0.02, -0.10, -0.05, 0.08, 0.10, 0.01])
    dd = compounded_dd_episodes(r)
    assert dd["Comp_MaxDD(%)"] > 14.0
    assert dd["MaxDD_Dur(d)"] >= 3.0

    rng = np.random.default_rng(42)
    a = rng.normal(0.0015, 0.01, size=500)
    b = rng.normal(0.0001, 0.01, size=500)
    dsh, z, p = ledoit_wolf_sharpe_test(a, b)
    assert dsh > 1.0 and z > 2.0 and p < 0.05

    boot = circular_block_bootstrap(a, n_boot=100, block_len=21, seed=7)
    assert boot["Sh_CI_Lo"] < boot["Sh_CI_Hi"]
    assert lo_hac_sharpe(pd.Series(a)) > 0.0
