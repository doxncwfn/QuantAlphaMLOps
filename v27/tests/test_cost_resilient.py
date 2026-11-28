"""Native v27 causality, neutrality, and turnover-accounting tests for v27.cost_resilient."""

from __future__ import annotations

import numpy as np
import pandas as pd

from v27.cost_resilient.optimizers import (
    apply_l1_no_trade_band,
    apply_partial_rebalance,
    apply_turnover_throttle,
    solve_admm_tail_l1,
    solve_admm_turnover_projection,
)
from v27.cost_resilient.signals import causal_ema_panel, causal_kalman_panel, xs_zscore


def _synthetic_panel(T: int = 40, N: int = 60, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=T)
    cols = np.arange(N, dtype=np.int64)
    latent = np.zeros((T, N), dtype=np.float64)
    latent[0] = rng.normal(0, 1, N)
    for t in range(1, T):
        latent[t] = 0.85 * latent[t - 1] + 0.4 * rng.normal(0, 1, N)
    scores = pd.DataFrame(
        latent + 0.5 * rng.normal(0, 1, (T, N)), index=dates, columns=cols
    )
    sigma = np.clip(rng.uniform(0.01, 0.04, (T, N)), 0.005, 0.10)
    beta = rng.normal(1.0, 0.35, (T, N))
    r1 = rng.normal(0.0, 0.015, (T, N))
    adv = rng.uniform(2e7, 2e8, (T, N))
    sectors = np.array([f"Sec_{i % 6}" for i in range(N)], dtype=object)
    return dates, scores, sigma, beta, sectors, r1, adv


def test_v27_signal_smoothing_strict_causality():
    _, scores, *_ = _synthetic_panel(T=35, N=50)
    z = xs_zscore(scores)
    z_pert = z.copy()
    t_cut = 20
    z_pert.iloc[t_cut + 1 :] = -5.0 * z_pert.iloc[t_cut + 1 :] + 3.0

    ema_a = causal_ema_panel(z, span=4.0)
    ema_b = causal_ema_panel(z_pert, span=4.0)
    np.testing.assert_array_equal(
        ema_a.iloc[: t_cut + 1].to_numpy(), ema_b.iloc[: t_cut + 1].to_numpy()
    )

    kal_a = causal_kalman_panel(z, q_ratio=0.30, phi=0.88, adaptive_phi=True)
    kal_b = causal_kalman_panel(z_pert, q_ratio=0.30, phi=0.88, adaptive_phi=True)
    np.testing.assert_array_equal(
        kal_a.iloc[: t_cut + 1].to_numpy(), kal_b.iloc[: t_cut + 1].to_numpy()
    )


def test_v27_admm_tail_l1_causality_neutrality_and_monotonic_turnover():
    _, scores, sigma, beta, sectors, r1, adv = _synthetic_panel(T=30, N=60)
    z = xs_zscore(scores)
    t_cut = 15

    z_pert = z.copy()
    z_pert.iloc[t_cut + 1 :] *= -2.0
    r1_pert = r1.copy()
    r1_pert[t_cut + 1 :] = 0.05

    W_a = solve_admm_tail_l1(
        z,
        sigma,
        beta,
        sectors,
        r1,
        adv_np=adv,
        l1_cost_bps=4.0,
        max_single_w=0.05,
        max_gross=1.80,
        n_iter=40,
    )
    W_b = solve_admm_tail_l1(
        z_pert,
        sigma,
        beta,
        sectors,
        r1_pert,
        adv_np=adv,
        l1_cost_bps=4.0,
        max_single_w=0.05,
        max_gross=1.80,
        n_iter=40,
    )
    np.testing.assert_allclose(W_a[: t_cut + 1], W_b[: t_cut + 1], rtol=0.0, atol=1e-12)

    assert float(np.max(np.abs(W_a))) <= 0.05 + 1e-9
    assert float(np.max(np.abs(W_a).sum(axis=1))) <= 1.80 + 1e-9
    assert float(np.max(np.abs(W_a[2:].sum(axis=1)))) < 0.02
    betas_p = np.einsum("ij,ij->i", W_a[2:], beta[2:])
    assert float(np.max(np.abs(betas_p))) < 0.03
    for s in np.unique(sectors):
        sec_net = W_a[2:, sectors == s].sum(axis=1)
        assert float(np.max(np.abs(sec_net))) < 0.03

    W_l0 = solve_admm_tail_l1(
        z,
        sigma,
        beta,
        sectors,
        r1,
        adv_np=adv,
        l1_cost_bps=0.0,
        max_single_w=0.05,
        max_gross=1.80,
    )
    W_l8 = solve_admm_tail_l1(
        z,
        sigma,
        beta,
        sectors,
        r1,
        adv_np=adv,
        l1_cost_bps=8.0,
        max_single_w=0.05,
        max_gross=1.80,
    )
    to_0 = float(np.abs(np.diff(W_l0, axis=0)).sum(axis=1).mean())
    to_4 = float(np.abs(np.diff(W_a, axis=0)).sum(axis=1).mean())
    to_8 = float(np.abs(np.diff(W_l8, axis=0)).sum(axis=1).mean())
    assert to_0 > to_4 > to_8


def test_v27_admm_turnover_projection_causality():
    _, scores, _, beta, sectors, r1, adv = _synthetic_panel(T=25, N=50)
    W_tar = xs_zscore(scores).to_numpy(dtype=np.float64) * 0.01
    t_cut = 12
    W_tar_p = W_tar.copy()
    W_tar_p[t_cut + 1 :] *= -1.0
    Wa = solve_admm_turnover_projection(
        W_tar, beta, sectors, r1, adv_np=adv, l1_weight=0.001
    )
    Wb = solve_admm_turnover_projection(
        W_tar_p, beta, sectors, r1, adv_np=adv, l1_weight=0.001
    )
    np.testing.assert_allclose(Wa[: t_cut + 1], Wb[: t_cut + 1], rtol=0.0, atol=1e-12)


def test_v27_pillar2_ntb_gp_and_throttle_exact_mechanics():
    rng = np.random.default_rng(11)
    T, N = 20, 40
    W_tar = rng.normal(0.0, 0.02, (T, N))
    W_tar -= W_tar.mean(axis=1, keepdims=True)
    r1 = np.zeros((T, N), dtype=np.float64)

    W_small = np.repeat(W_tar[:1], T, axis=0)
    W_small[1:] += 0.0005
    W_ntb = apply_l1_no_trade_band(W_small, r1, band=0.0010, max_gross=None)
    np.testing.assert_allclose(
        W_ntb[1:], np.repeat(W_ntb[:1], T - 1, axis=0), rtol=0.0, atol=1e-12
    )

    W_gp = apply_partial_rebalance(W_tar, r1, gamma=0.40, max_gross=None)
    np.testing.assert_allclose(
        W_gp[1], 0.60 * W_tar[0] + 0.40 * W_tar[1], rtol=1e-12, atol=1e-12
    )

    cap = 0.15
    W_th = apply_turnover_throttle(
        W_tar, r1, max_daily_1way=cap, prioritize_largest=True
    )
    daily_1way = 0.5 * np.abs(np.diff(W_th, axis=0)).sum(axis=1)
    assert float(daily_1way.max()) <= cap + 1e-9
