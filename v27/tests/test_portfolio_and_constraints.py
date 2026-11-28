"""Tests for neutralization, 8 portfolio constructors, consolidation netting, and constraints."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from v27.backtest.simulator import backtest_weights_daily
from v27.portfolio.consolidation import consolidate_positions
from v27.portfolio.constructors import (
    apply_overlapping_hold,
    build_8_strategies_for_score,
)
from v27.portfolio.neutralization import neutralize_beta, renorm_legs
from v27.risk.constraints import apply_portfolio_constraints


def test_beta_neutralization_and_8_constructions():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=25, freq="B")
    cols = np.arange(50)
    beta = pd.DataFrame(rng.uniform(0.5, 1.5, size=(25, 50)), index=dates, columns=cols)
    score = pd.DataFrame(
        2.0 * beta.to_numpy() + rng.normal(0.0, 0.3, size=(25, 50)),
        index=dates,
        columns=cols,
    )
    ret1d = pd.DataFrame(
        rng.normal(0.0, 0.015, size=(25, 50)), index=dates, columns=cols
    )
    sigma = pd.DataFrame(
        rng.uniform(0.01, 0.03, size=(25, 50)), index=dates, columns=cols
    )

    bn = neutralize_beta(score, beta)
    # Cross-sectional correlation with beta_60 must vanish
    corrs = bn.corrwith(beta, axis=1).abs()
    assert float(corrs.max()) < 1e-10

    strats = build_8_strategies_for_score(score, ret1d, sigma, beta)
    assert len(strats) == 8
    for W in strats.values():
        np.testing.assert_allclose(W.sum(axis=1), 0.0, atol=1e-9)
        np.testing.assert_allclose(np.abs(W).sum(axis=1), 2.0, atol=1e-9)


def test_consolidation_netting_and_constraints():
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    r = np.zeros((10, 4))
    w_a = np.zeros((10, 4))
    w_b = np.zeros((10, 4))
    w_a[:5] = [0.5, 0.5, -0.5, -0.5]
    w_a[5:] = [-0.5, 0.5, 0.5, -0.5]
    w_b[:5] = [-0.5, 0.5, 0.5, -0.5]
    w_b[5:] = [0.5, 0.5, -0.5, -0.5]
    sched = pd.DataFrame({"A": np.full(10, 0.5), "B": np.full(10, 0.5)}, index=dates)
    W_cons = consolidate_positions(sched, {"A": w_a, "B": w_b})
    bt_cons = backtest_weights_daily(
        W_cons, r, dates, cost_bps=2.0, inception_zero_trade=True
    )
    bt_a = backtest_weights_daily(
        w_a, r, dates, cost_bps=2.0, inception_zero_trade=True
    )
    bt_b = backtest_weights_daily(
        w_b, r, dates, cost_bps=2.0, inception_zero_trade=True
    )
    assert float(bt_cons["tc"].sum()) < 0.25 * float(
        (0.5 * bt_a["tc"] + 0.5 * bt_b["tc"]).sum()
    )

    rng = np.random.default_rng(123)
    W = rng.normal(0.0, 0.04, size=(20, 40))
    W -= W.mean(axis=1, keepdims=True)
    beta = rng.uniform(0.5, 1.8, size=(20, 40))
    sectors = np.array(
        ["Tech_BusinessEquip" if i < 20 else "Healthcare" for i in range(40)]
    )
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


def test_overlapping_hold_uses_1d_returns():
    dates = pd.date_range("2020-01-01", periods=8, freq="B")
    r_1d = pd.DataFrame([[0.01, -0.01]] * 8, index=dates, columns=[1, 2])
    w_tar = renorm_legs(pd.DataFrame([[1.0, -1.0]] * 8, index=dates, columns=[1, 2]))
    w_h5 = apply_overlapping_hold(w_tar, r_1d, hold_days=5)
    bt = backtest_weights_daily(w_h5.to_numpy(), r_1d.to_numpy(), dates, cost_bps=0.0)
    assert math.isclose(float(bt["gross"].iloc[0]), 0.02, rel_tol=1e-6)
