"""Post-consolidation single-name, gross, beta, sector, and ADV liquidity constraints."""

from __future__ import annotations

import math

import numpy as np


def apply_portfolio_constraints(
    W_in: np.ndarray,
    beta60_np: np.ndarray | None = None,
    sectors: np.ndarray | None = None,
    adv20_np: np.ndarray | None = None,
    r1d_np: np.ndarray | None = None,
    max_single_w: float | None = 0.030,
    max_gross: float | None = None,
    max_abs_beta: float | None = None,
    max_sector_net: float | None = None,
    max_adv_frac: float | None = None,
    book_size: float = 20_000_000.0,
) -> np.ndarray:
    """Enforce single-name cap, sector net neutrality, beta bound, gross cap, and 5% ADV turnover limit."""
    W = np.nan_to_num(W_in.astype(np.float64), nan=0.0)
    T, _ = W.shape

    if max_single_w is not None:
        W = np.clip(W, -max_single_w, max_single_w)

    if max_sector_net is not None and sectors is not None:
        for sec in np.unique(sectors):
            sm = sectors == sec
            sec_net = W[:, sm].sum(axis=1)
            over_p = sec_net > max_sector_net
            if np.any(over_p):
                ex = sec_net[over_p] - max_sector_net
                sub = W[over_p][:, sm]
                pos = np.maximum(sub, 0.0)
                W[np.ix_(over_p, sm)] = (
                    sub
                    - (pos / np.maximum(pos.sum(axis=1, keepdims=True), 1e-12))
                    * ex[:, None]
                )
            over_n = sec_net < -max_sector_net
            if np.any(over_n):
                ex = (-sec_net[over_n]) - max_sector_net
                sub = W[over_n][:, sm]
                neg = np.maximum(-sub, 0.0)
                W[np.ix_(over_n, sm)] = (
                    sub
                    + (neg / np.maximum(neg.sum(axis=1, keepdims=True), 1e-12))
                    * ex[:, None]
                )

    if max_abs_beta is not None and beta60_np is not None:
        B = np.nan_to_num(beta60_np, nan=1.0)
        for t in range(T):
            w_t = W[t]
            b_t = B[t]
            act = np.abs(w_t) > 1e-10
            if act.sum() < 10:
                continue
            port_b = float(np.dot(w_t, b_t))
            if abs(port_b) > max_abs_beta:
                target_b = math.copysign(max_abs_beta, port_b)
                b_a = b_t[act]
                bc = b_a - b_a.mean()
                denom = float(np.dot(bc, bc))
                if denom > 1e-10:
                    adj = ((port_b - target_b) / denom) * bc
                    w_t_new = w_t.copy()
                    w_t_new[act] = w_t[act] - adj
                    g_old = float(np.abs(w_t).sum())
                    g_new = float(np.abs(w_t_new).sum())
                    if g_new > 1e-12:
                        w_t_new *= g_old / g_new
                    W[t] = w_t_new

    if max_single_w is not None:
        W = np.clip(W, -max_single_w, max_single_w)

    if max_gross is not None:
        g = np.abs(W).sum(axis=1)
        s = np.where(g > max_gross, max_gross / np.maximum(g, 1e-12), 1.0)
        W = W * s[:, None]

    if max_adv_frac is not None and adv20_np is not None and r1d_np is not None:
        ADV = np.maximum(np.nan_to_num(adv20_np, nan=50e6), 1e6)
        R = np.nan_to_num(r1d_np, nan=0.0)
        max_dw = (max_adv_frac * ADV) / book_size
        for t in range(1, T):
            gp = float(np.dot(W[t - 1], R[t - 1]))
            dr = W[t - 1] * (1.0 + R[t - 1]) / max(1.0 + gp, 0.25)
            dw = np.clip(W[t] - dr, -max_dw[t], max_dw[t])
            W[t] = dr + dw

    return W
