"""Covariance and expected-return shrinkage estimators."""

from __future__ import annotations

import numpy as np


def ledoit_wolf_cov(X: np.ndarray) -> np.ndarray:
    """Closed-form Ledoit-Wolf (2004) analytical shrinkage covariance toward scaled identity."""
    X = np.nan_to_num(X.astype(np.float64), nan=0.0)
    T, N = X.shape
    if N == 1:
        return np.array([[max(float(np.var(X, ddof=1)), 1e-10)]], dtype=np.float64)
    if T < 5:
        return np.eye(N, dtype=np.float64) * 1e-4
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / T
    mu = float(np.trace(S)) / N
    F = mu * np.eye(N)
    delta2 = float(np.sum((S - F) ** 2))
    if delta2 < 1e-16:
        return S + 1e-8 * np.eye(N)
    Y2 = Xc**2
    phi = float(np.sum((Y2.T @ Y2) / T - S**2)) / T
    shrink = float(np.clip(phi / delta2, 0.0, 1.0))
    cov = 0.5 * ((1.0 - shrink) * S + shrink * F)
    cov = cov + cov.T
    eig_v, eig_u = np.linalg.eigh(cov)
    eig_v = np.maximum(eig_v, 1e-10)
    return (eig_u * eig_v) @ eig_u.T


def cov_corr_shrink(X: np.ndarray, corr_shrink: float = 0.25) -> np.ndarray:
    """Ledoit-Wolf covariance with additional correlation shrinkage toward diagonal."""
    S = ledoit_wolf_cov(X)
    sd = np.sqrt(np.maximum(np.diag(S), 1e-12))
    C = np.clip(S / np.outer(sd, sd), -0.99, 0.99)
    C_s = (1.0 - corr_shrink) * C + corr_shrink * np.eye(len(sd))
    return C_s * np.outer(sd, sd)


def bayes_stein_mu(X: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Jorion (1986) Bayes-Stein shrinkage of sample mean returns toward minimum-variance mean."""
    T, N = X.shape
    mu_hat = np.nanmean(X, axis=0)
    if N <= 2 or T < 20:
        return mu_hat
    inv_cov = np.linalg.pinv(cov)
    ones = np.ones(N)
    w_mv = inv_cov @ ones
    denom_mv = float(ones @ w_mv)
    w_mv = w_mv / denom_mv if abs(denom_mv) > 1e-12 else ones / N
    mu_0 = float(w_mv @ mu_hat)
    diff = mu_hat - mu_0
    quad = float(diff @ inv_cov @ diff)
    lam = (N + 2.0) / max((N + 2.0) + T * max(quad, 1e-12), 1e-12)
    lam = float(np.clip(lam, 0.15, 0.92))
    return (1.0 - lam) * mu_hat + lam * mu_0
