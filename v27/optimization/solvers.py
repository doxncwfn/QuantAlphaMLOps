"""Constrained portfolio weight solvers (InvVol, ERC, MinVar, MaxSharpe, Mean-CVaR LP, RegRetRisk)."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import csc_matrix, hstack, vstack
from scipy.sparse import eye as sp_eye

from v27.optimization.estimators import ledoit_wolf_cov


def project_simplex_bounded(
    w: np.ndarray, w_max: float = 1.0, w_min: float = 0.0
) -> np.ndarray:
    """Project weights onto simplex sum(w)=1 with elementwise [w_min, w_max] bounds."""
    w = np.clip(np.nan_to_num(w, nan=0.0), w_min, w_max)
    s = float(w.sum())
    if s <= 1e-12:
        return np.ones_like(w) / len(w)
    w /= s
    for _ in range(10):
        w = np.clip(w, w_min, w_max)
        diff = 1.0 - float(w.sum())
        if abs(diff) < 1e-9:
            break
        if diff > 0:
            room = w < w_max - 1e-9
            if not room.any():
                break
            w[room] += diff * (
                (w_max - w[room]) / max(float((w_max - w[room]).sum()), 1e-12)
            )
        else:
            room = w > w_min + 1e-9
            if not room.any():
                break
            w[room] += diff * (
                (w[room] - w_min) / max(float((w[room] - w_min).sum()), 1e-12)
            )
    return w / max(float(w.sum()), 1e-12)


def solve_inv_vol(X: np.ndarray, w_max: float = 0.40) -> np.ndarray:
    sd = np.nanstd(X, axis=0, ddof=1)
    iv = 1.0 / np.maximum(sd, 1e-5)
    return project_simplex_bounded(iv / iv.sum(), w_max=w_max)


def solve_erc(cov: np.ndarray, w_max: float = 0.40) -> np.ndarray:
    """Equal Risk Contribution portfolio via SLSQP."""
    N = cov.shape[0]
    if N == 1:
        return np.array([1.0])
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    w0 = project_simplex_bounded((1.0 / sd) / np.sum(1.0 / sd), w_max=w_max)
    cov_s = cov * 1e4

    def obj(w: np.ndarray) -> float:
        Sw = cov_s @ w
        pv = max(float(w @ Sw), 1e-12)
        rc = (w * Sw) / pv
        return float(np.sum((rc - 1.0 / N) ** 2)) * 1e4

    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=[(0.0, w_max)] * N,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"ftol": 1e-11, "maxiter": 180},
    )
    return project_simplex_bounded(res.x if res.success else w0, w_max=w_max)


def solve_min_var(
    cov: np.ndarray, w_max: float = 0.40, l2_reg: float = 0.0
) -> np.ndarray:
    N = cov.shape[0]
    w0 = np.ones(N) / N
    cov_s = cov * 1e4
    tr_scale = float(np.trace(cov_s)) / N

    def obj(w: np.ndarray) -> float:
        return float(w @ cov_s @ w) + l2_reg * tr_scale * float(np.sum(w**2))

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * (cov_s @ w) + 2.0 * l2_reg * tr_scale * w

    res = minimize(
        obj,
        w0,
        method="SLSQP",
        jac=jac,
        bounds=[(0.0, w_max)] * N,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"ftol": 1e-11, "maxiter": 180},
    )
    return project_simplex_bounded(res.x if res.success else w0, w_max=w_max)


def solve_max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    w_max: float = 0.35,
    rf_daily: float = 0.0,
    ann_days: float = 252.0,
) -> np.ndarray:
    N = len(mu)
    ex_mu = (mu - rf_daily) * ann_days
    cov_ann = cov * ann_days
    w0 = solve_erc(cov, w_max=w_max)

    def neg_sh(w: np.ndarray) -> float:
        ret = float(w @ ex_mu)
        vol = math.sqrt(max(float(w @ cov_ann @ w), 1e-10))
        return -ret / vol

    res = minimize(
        neg_sh,
        w0,
        method="SLSQP",
        bounds=[(0.0, w_max)] * N,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"ftol": 1e-10, "maxiter": 180},
    )
    return project_simplex_bounded(res.x if res.success else w0, w_max=w_max)


def solve_mean_cvar_lp(
    X: np.ndarray,
    mu: np.ndarray,
    alpha: float = 0.95,
    w_max: float = 0.35,
    min_ret_frac: float = 0.50,
) -> np.ndarray:
    """Rockafellar-Uryasev (2000) Mean-CVaR LP via SciPy HiGHS."""
    T, N = X.shape
    R = np.nan_to_num(X.astype(np.float64), nan=0.0)
    mu_eq, mu_hi, mu_lo = float(np.mean(mu)), float(np.max(mu)), float(np.min(mu))
    mu_target = min(
        mu_lo + min_ret_frac * max(mu_hi - mu_lo, 0.0), 0.85 * mu_hi + 0.15 * mu_eq
    )

    n_vars = N + 1 + T
    c = np.zeros(n_vars, dtype=np.float64)
    c[N] = 1.0
    c[N + 1 :] = 1.0 / ((1.0 - alpha) * T)

    A_tail = hstack(
        [csc_matrix(-R), csc_matrix(-np.ones((T, 1))), -sp_eye(T, format="csc")]
    )
    b_tail = np.zeros(T, dtype=np.float64)
    A_ret = csc_matrix(np.hstack([-mu.reshape(1, N), np.zeros((1, 1 + T))]))
    b_ret = np.array([-mu_target], dtype=np.float64)
    A_ub = vstack([A_tail, A_ret]).tocsc()
    b_ub = np.concatenate([b_tail, b_ret])
    A_eq = csc_matrix(np.hstack([np.ones((1, N)), np.zeros((1, 1 + T))]))
    b_eq = np.array([1.0], dtype=np.float64)

    bounds = [(0.0, w_max)] * N + [(-1.0, 1.0)] + [(0.0, None)] * T
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if res.success:
        return project_simplex_bounded(res.x[:N], w_max=w_max)
    return solve_min_var(ledoit_wolf_cov(R), w_max=w_max, l2_reg=0.1)


def solve_regularized_return_risk(
    mu: np.ndarray,
    cov: np.ndarray,
    w_prev: np.ndarray,
    w_max: float = 0.35,
    risk_aversion: float = 2.5,
    l2_div: float = 1.0,
    l1_turn: float = 0.02,
    est_err_pen: float = 0.25,
    ann_days: float = 252.0,
) -> np.ndarray:
    """Regularized mean-variance with L2 diversification, L1 turnover penalty, and std-error penalty."""
    N = len(mu)
    mu_ann = mu * ann_days
    cov_ann = cov * ann_days
    sd_ann = np.sqrt(np.maximum(np.diag(cov_ann), 1e-8))
    w0 = project_simplex_bounded(w_prev.copy(), w_max=w_max)
    tr_var = float(np.trace(cov_ann)) / N

    def obj(w: np.ndarray) -> float:
        ret = float(w @ mu_ann)
        var = float(w @ cov_ann @ w)
        pen_l2 = l2_div * tr_var * float(np.sum(w**2))
        pen_l1 = l1_turn * float(np.sum(np.sqrt((w - w_prev) ** 2 + 1e-8)))
        pen_err = est_err_pen * float(w @ sd_ann)
        return -ret + 0.5 * risk_aversion * var + pen_l2 + pen_l1 + pen_err

    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=[(0.0, w_max)] * N,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"ftol": 1e-10, "maxiter": 180},
    )
    return project_simplex_bounded(res.x if res.success else w0, w_max=w_max)
