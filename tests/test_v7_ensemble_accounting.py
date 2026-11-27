"""Unit tests for Section 6.5: consolidated position-level accounting & 1d vs 5d audit."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.qam.v7_ensemble.engine import (
    apply_overlapping_hold,
    apply_portfolio_constraints,
    backtest_weights_daily,
    consolidate_positions,
)


def _toy_panel():
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    rng = np.random.default_rng(7)
    r = rng.normal(0.0005, 0.015, size=(10, 6))
    w1 = np.array([[0.5, 0.5, 0.0, -0.5, -0.5, 0.0]] * 10)
    w2 = np.array([[-0.5, 0.0, 0.5, 0.5, 0.0, -0.5]] * 10)
    return dates, r, w1, w2


def test_single_strategy_ensemble_reproduces_original():
    """1. A one-strategy ensemble reproduces that strategy's original positions and PnL."""
    dates, r, w1, _ = _toy_panel()
    sched = pd.DataFrame({"S1": np.ones(10)}, index=dates)
    W_cons = consolidate_positions(sched, {"S1": w1})
    np.testing.assert_allclose(W_cons, w1, atol=1e-12)
    bt1 = backtest_weights_daily(w1, r, dates, cost_bps=2.0)
    bt_c = backtest_weights_daily(W_cons, r, dates, cost_bps=2.0)
    pd.testing.assert_series_equal(bt1["gross"], bt_c["gross"])
    pd.testing.assert_series_equal(bt1["net"], bt_c["net"])
    pd.testing.assert_series_equal(bt1["traded"], bt_c["traded"])


def test_combining_identical_strategies_preserves_positions():
    """2. Combining identical strategies with weights summing to 1 produces identical positions."""
    dates, _r, w1, _ = _toy_panel()
    sched = pd.DataFrame(
        {"S1": np.full(10, 0.35), "S2": np.full(10, 0.65)}, index=dates
    )
    W_cons = consolidate_positions(sched, {"S1": w1, "S2": w1})
    np.testing.assert_allclose(W_cons, w1, atol=1e-12)


def test_opposing_positions_properly_netted():
    """3. Opposing stock positions across sleeves cancel out in consolidated holdings."""
    dates, r, w1, w2 = _toy_panel()
    sched = pd.DataFrame({"S1": np.full(10, 0.5), "S2": np.full(10, 0.5)}, index=dates)
    W_cons = consolidate_positions(sched, {"S1": w1, "S2": w2})
    # Stock 0 (+0.5 vs -0.5) and Stock 3 (-0.5 vs +0.5) must net exactly to 0.0!
    np.testing.assert_allclose(W_cons[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(W_cons[:, 3], 0.0, atol=1e-12)
    # Gross exposure compresses from 2.0 to 1.0
    bt_c = backtest_weights_daily(W_cons, r, dates, cost_bps=2.0)
    np.testing.assert_allclose(bt_c["gross_exp"].to_numpy(), 1.0, atol=1e-12)


def test_transaction_costs_on_consolidated_orders():
    """4. Transaction costs are charged on consolidated net orders, strictly lower than un-netted sum."""
    dates, r, _, _ = _toy_panel()
    # Two sleeves flipping opposite directions on day 5
    w_a = np.zeros((10, 4))
    w_b = np.zeros((10, 4))
    w_a[:5] = [0.5, 0.5, -0.5, -0.5]
    w_a[5:] = [
        -0.5,
        0.5,
        0.5,
        -0.5,
    ]  # stock 0 flips +0.5 -> -0.5, stock 2 flips -0.5 -> +0.5
    w_b[:5] = [-0.5, 0.5, 0.5, -0.5]
    w_b[5:] = [
        0.5,
        0.5,
        -0.5,
        -0.5,
    ]  # exact opposite flip -> consolidated portfolio never changes!
    sched = pd.DataFrame({"A": np.full(10, 0.5), "B": np.full(10, 0.5)}, index=dates)
    W_cons = consolidate_positions(sched, {"A": w_a, "B": w_b})
    r_sub = r[:, :4]
    bt_cons = backtest_weights_daily(
        W_cons, r_sub, dates, cost_bps=2.0, inception_zero_trade=True
    )
    bt_a = backtest_weights_daily(
        w_a, r_sub, dates, cost_bps=2.0, inception_zero_trade=True
    )
    bt_b = backtest_weights_daily(
        w_b, r_sub, dates, cost_bps=2.0, inception_zero_trade=True
    )
    unnetted_tc = 0.5 * bt_a["tc"] + 0.5 * bt_b["tc"]
    assert float(bt_cons["tc"].sum()) < float(unnetted_tc.sum()) * 0.25


def test_strategy_allocation_shifts_generate_trades():
    """5. Reallocating weights across static sleeves generates the exact expected portfolio trades."""
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    r = np.zeros((4, 4))
    w1 = np.array([[0.5, 0.5, -0.5, -0.5]] * 4)
    w2 = np.array([[-0.5, -0.5, 0.5, 0.5]] * 4)
    sched = pd.DataFrame(
        {"S1": [1.0, 1.0, 0.0, 0.0], "S2": [0.0, 0.0, 1.0, 1.0]}, index=dates
    )
    W_cons = consolidate_positions(sched, {"S1": w1, "S2": w2})
    bt = backtest_weights_daily(
        W_cons, r, dates, cost_bps=10.0, inception_zero_trade=True
    )
    # On day 2, every stock moves by 1.0 -> 4 * 1.0 = 4.0 two-way traded
    assert math.isclose(float(bt["traded"].iloc[2]), 4.0, rel_tol=1e-9)
    assert math.isclose(float(bt["traded"].iloc[1]), 0.0, abs_tol=1e-12)
    assert math.isclose(float(bt["tc"].iloc[2]), 4.0 * 10e-4, rel_tol=1e-9)


def test_constraints_enforced():
    """6. Portfolio weights respect single-name, gross, sector, and beta exposure constraints."""
    rng = np.random.default_rng(123)
    W = rng.normal(0.0, 0.04, size=(20, 40))
    W = W - W.mean(axis=1, keepdims=True)
    beta = rng.uniform(0.5, 1.8, size=(20, 40))
    sectors = np.array(["Tech" if i < 20 else "Health" for i in range(40)])
    W_c = apply_portfolio_constraints(
        W,
        beta60_np=beta,
        sectors=sectors,
        max_single_w=0.025,
        max_gross=1.50,
        max_abs_beta=0.05,
        max_sector_net=0.10,
    )
    assert float(np.abs(W_c).max()) <= 0.025 + 1e-6
    assert float(np.abs(W_c).sum(axis=1).max()) <= 1.50 + 1e-6


def test_1d_vs_5d_overlapping_hold_uses_daily_returns():
    """7. Overlapping 5-day hold marks each tranche using 1d daily returns, never 5d forward labels."""
    dates = pd.date_range("2020-01-01", periods=8, freq="B")
    cols = [1, 2]
    r_1d = pd.DataFrame([[0.01, -0.01]] * 8, index=dates, columns=cols)
    w_tar = pd.DataFrame([[1.0, -1.0]] * 8, index=dates, columns=cols)
    w_h5 = apply_overlapping_hold(w_tar, r_1d, hold_days=5)
    bt = backtest_weights_daily(w_h5.to_numpy(), r_1d.to_numpy(), dates, cost_bps=0.0)
    # Daily return on (+1.0, -1.0) with (+0.01, -0.01) is 0.02/day (NOT 5x compounded ~0.104/day!)
    assert float(bt["gross"].max()) < 0.025
    assert math.isclose(float(bt["gross"].iloc[0]), 0.02, rel_tol=1e-6)
