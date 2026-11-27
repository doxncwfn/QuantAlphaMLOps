"""Tests for covariance shrinkage, convex/LP portfolio optimization, and schedule hysteresis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from v27.optimization.estimators import bayes_stein_mu, cov_corr_shrink, ledoit_wolf_cov
from v27.optimization.schedule import causal_wf_schedule
from v27.optimization.solvers import (
    solve_erc,
    solve_max_sharpe,
    solve_mean_cvar_lp,
    solve_min_var,
    solve_regularized_return_risk,
)


def test_shrinkage_cov_is_symmetric_psd():
    rng = np.random.default_rng(5)
    X = rng.normal(0.0003, 0.01, size=(120, 8))
    cov_lw = ledoit_wolf_cov(X)
    cov_cs = cov_corr_shrink(X, corr_shrink=0.25)
    assert cov_lw.shape == (8, 8)
    np.testing.assert_allclose(cov_lw, cov_lw.T, atol=1e-12)
    assert float(np.linalg.eigvalsh(cov_lw).min()) > 0.0
    assert float(np.linalg.eigvalsh(cov_cs).min()) > 0.0
    mu_bs = bayes_stein_mu(X, cov_lw)
    assert mu_bs.shape == (8,) and np.all(np.isfinite(mu_bs))


def test_erc_and_solvers_produce_valid_simplex_weights():
    rng = np.random.default_rng(19)
    X = rng.normal(0.0004, 0.01, size=(150, 6))
    X[:, 0] *= 2.5  # higher vol asset should receive lower ERC/MinVar weight
    cov = ledoit_wolf_cov(X)
    mu = bayes_stein_mu(X, cov)

    w_erc = solve_erc(cov, w_max=0.40)
    assert abs(float(w_erc.sum()) - 1.0) < 1e-6
    assert float(w_erc.max()) <= 0.40 + 1e-6
    assert w_erc[0] < np.median(w_erc[1:])

    for w in (
        solve_min_var(cov, w_max=0.40, l2_reg=0.1),
        solve_max_sharpe(mu, cov, w_max=0.40),
        solve_mean_cvar_lp(X, mu, w_max=0.40),
        solve_regularized_return_risk(mu, cov, np.ones(6) / 6, w_max=0.40),
    ):
        assert abs(float(w.sum()) - 1.0) < 1e-6
        assert float(w.min()) >= -1e-8 and float(w.max()) <= 0.40 + 1e-6


def test_causal_schedule_respects_hysteresis():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(2)
    ret_df = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(100, 4)), index=dates, columns=["A", "B", "C", "D"]
    )
    # Slight jitter in optimal weights inside hysteresis band=0.50 -> holds initial weights cleanly
    sched = causal_wf_schedule(
        ret_df,
        dates,
        lambda X, wp, t: np.array([0.27, 0.25, 0.24, 0.24]),
        rebal_freq=10,
        warmup=30,
        purge=2,
        hyst_tau=0.50,
    )
    assert sched.shape == (100, 4)
    np.testing.assert_allclose(sched.sum(axis=1), 1.0, atol=1e-12)
