"""Pillars 2 & 3: Rebalancing mechanics, exact Woodbury-ADMM L1-turnover QP, and cost-aware overlays."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from v27.config import PortfolioBacktestConfig
from v27.data.bundle import V7DataBundle
from v27.portfolio.constructors import apply_overlapping_hold, strat_top10_invvol
from v27.portfolio.neutralization import neutralize_beta
from v27.risk.constraints import apply_portfolio_constraints

_BT_CFG = PortfolioBacktestConfig()
ANN_DAYS = float(_BT_CFG.ann_days)
BOOK_SIZE = float(_BT_CFG.book_size)


# ---------------------------------------------------------------------------
# Pillar 2: Rebalancing Mechanics (NTB, Garleanu-Pedersen, Throttle, BetaNeu+HOLD)
# ---------------------------------------------------------------------------
def build_betaneu_invvol_hold(
    score_df: pd.DataFrame,
    bundle: V7DataBundle,
    hold_days: int = 5,
    q: float = 0.10,
) -> np.ndarray:
    """Beta-residualized top/bottom-decile InvVol portfolio with overlapping multi-day tranches."""
    sc_bn = neutralize_beta(score_df, bundle.beta60_df)
    inv_vol = (1.0 / bundle.sigma_prev_df).where(score_df.notna()).fillna(0.0)
    w_t10 = strat_top10_invvol(sc_bn, inv_vol, q=q)
    return apply_overlapping_hold(w_t10, bundle.ret1d_df, hold_days=hold_days).to_numpy(
        dtype=np.float64
    )


def apply_l1_no_trade_band(
    W_tar: np.ndarray,
    r1_np: np.ndarray,
    band: float = 0.0010,
    max_gross: float | None = 2.0,
) -> np.ndarray:
    """Elementwise L1 no-trade band (proximal soft-thresholding) around drifted prior weights."""
    T, N = W_tar.shape
    W = np.zeros((T, N), dtype=np.float64)
    W[0] = np.nan_to_num(W_tar[0], nan=0.0)
    for t in range(1, T):
        gp = float(np.dot(W[t - 1], r1_np[t - 1]))
        dr = W[t - 1] * (1.0 + r1_np[t - 1]) / max(1.0 + gp, 0.25)
        diff = np.nan_to_num(W_tar[t], nan=0.0) - dr
        dw = np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
        w = dr + dw
        if max_gross is not None:
            g = float(np.abs(w).sum())
            if g > max_gross:
                w *= max_gross / g
        W[t] = w
    return W


def apply_partial_rebalance(
    W_tar: np.ndarray,
    r1_np: np.ndarray,
    gamma: float = 0.35,
    max_gross: float | None = 2.0,
) -> np.ndarray:
    """Garleanu-Pedersen partial trading toward aim portfolio: w_t = (1-gamma)*w_drift + gamma*w_tar."""
    T, N = W_tar.shape
    W = np.zeros((T, N), dtype=np.float64)
    W[0] = np.nan_to_num(W_tar[0], nan=0.0)
    for t in range(1, T):
        gp = float(np.dot(W[t - 1], r1_np[t - 1]))
        dr = W[t - 1] * (1.0 + r1_np[t - 1]) / max(1.0 + gp, 0.25)
        w = (1.0 - gamma) * dr + gamma * np.nan_to_num(W_tar[t], nan=0.0)
        if max_gross is not None:
            g = float(np.abs(w).sum())
            if g > max_gross:
                w *= max_gross / g
        W[t] = w
    return W


def apply_turnover_throttle(
    W_tar: np.ndarray,
    r1_np: np.ndarray,
    max_daily_1way: float = 0.20,
    prioritize_largest: bool = True,
) -> np.ndarray:
    """Hard daily 1-way turnover budget cap (`sum|dw| <= 2 * max_daily_1way`).

    When `prioritize_largest=True`, solves the exact L1 soft-thresholding level tau_t so
    sum_i max(|W_tar_i - dr_i| - tau_t, 0) == 2 * max_daily_1way (executing highest-conviction
    weight adjustments first rather than proportionally scaling every trade).
    """
    T, N = W_tar.shape
    W = np.zeros((T, N), dtype=np.float64)
    W[0] = np.nan_to_num(W_tar[0], nan=0.0)
    max_2way = 2.0 * max_daily_1way
    for t in range(1, T):
        gp = float(np.dot(W[t - 1], r1_np[t - 1]))
        dr = W[t - 1] * (1.0 + r1_np[t - 1]) / max(1.0 + gp, 0.25)
        diff = np.nan_to_num(W_tar[t], nan=0.0) - dr
        abs_d = np.abs(diff)
        tot = float(abs_d.sum())
        if tot <= max_2way or tot <= 1e-12:
            W[t] = dr + diff
        elif not prioritize_largest:
            W[t] = dr + diff * (max_2way / tot)
        else:
            # Exact projection onto L1 ball of radius max_2way via Duchi et al. (2008) water-filling
            u = np.sort(abs_d)[::-1]
            cssv = np.cumsum(u) - max_2way
            ind = np.arange(1, N + 1, dtype=np.float64)
            cond = u - cssv / ind > 0.0
            rho = int(np.nonzero(cond)[0][-1])
            tau = cssv[rho] / float(rho + 1)
            dw = np.sign(diff) * np.maximum(abs_d - tau, 0.0)
            W[t] = dr + dw
    return W


# ---------------------------------------------------------------------------
# Pillar 3: Exact Woodbury-ADMM Tail-Alpha + L1 Turnover-Penalized Optimizer
# ---------------------------------------------------------------------------
def solve_admm_tail_l1(
    score_df: pd.DataFrame,
    sigma_np: np.ndarray,
    beta_np: np.ndarray,
    sectors: np.ndarray,
    r1_np: np.ndarray,
    mom_np: np.ndarray | None = None,
    adv_np: np.ndarray | None = None,
    tail_deadzone: float = 0.80,
    base_gross: float = 1.80,
    l1_cost_bps: float = 4.0,
    l2_cost: float = 0.0,
    ic_daily: float = 0.025,
    neutralize_sectors: bool = True,
    neutralize_vol: bool = True,
    neutralize_mom: bool = False,
    vol_target_ann: float | None = None,
    vol_deadband: float = 0.10,
    dd_brake: bool = False,
    max_single_w: float = 0.025,
    max_gross: float = 2.00,
    max_adv_frac: float = 0.05,
    n_iter: int = 35,
) -> np.ndarray:
    """Exact Woodbury-ADMM Grinold-Kahn Tail-Alpha + L1 Transaction Cost + Factor-Neutral Optimizer.

    Each trading day t solves the strictly convex problem:
        min_{w, z}  -alpha_t^T w + 0.5 * gamma_t * sum_i (sigma_{i,t}^2 * w_i^2)
                    + (l1_cost_bps / 1e4) * ||z - w_drift_t||_1
                    + 0.5 * l2_cost * ||z - w_drift_t||_2^2
        subject to  C_t w = 0         (exact dollar-, beta-, optional sector-/vol-/mom-neutrality)
                    w = z
                    |z_i| <= max_single_w
                    |z_i - w_drift_{i,t}| <= max_adv_frac * ADV_{i,t} / BOOK_SIZE
                    ||z||_1 <= eff_max_gross

    Analytical risk-aversion calibration sets gamma_t = ic_daily * sum(|ts_i| / sigma_i) / target_gross
    so unconstrained weights scale naturally as InvVol with gross == target_gross, ensuring interior
    solutions where `l1_cost_bps` is measured directly in daily expected return basis points.
    """
    S_raw = score_df.to_numpy(dtype=np.float64)
    T, N = S_raw.shape
    uniq_secs = np.unique(sectors)
    K_sec = len(uniq_secs)
    sec_idx = np.searchsorted(uniq_secs, sectors)
    S_onehot = np.zeros((K_sec, N), dtype=np.float64)
    S_onehot[sec_idx, np.arange(N)] = 1.0

    l1_cost = l1_cost_bps / 1e4
    max_dw_mat = None
    if adv_np is not None:
        ADV = np.maximum(np.nan_to_num(adv_np, nan=50e6), 1e6)
        max_dw_mat = (max_adv_frac * ADV) / BOOK_SIZE

    W = np.zeros((T, N), dtype=np.float64)
    port_rets = np.zeros(T, dtype=np.float64)
    nav_run = 1.0
    hwm_run = 1.0
    gross_scale_state = 1.0

    for t in range(T):
        if t > 0:
            gp = float(np.dot(W[t - 1], r1_np[t - 1]))
            port_rets[t - 1] = gp
            nav_run *= max(1.0 + gp, 0.05)
            hwm_run = max(hwm_run, nav_run)
            w_dr_full = W[t - 1] * (1.0 + r1_np[t - 1]) / max(1.0 + gp, 0.25)
        else:
            w_dr_full = np.zeros(N, dtype=np.float64)

        if vol_target_ann is not None and t >= 25:
            v20 = float(np.std(port_rets[max(0, t - 20) : t], ddof=1)) * math.sqrt(
                ANN_DAYS
            )
            v63 = (
                float(np.std(port_rets[max(0, t - 63) : t], ddof=1))
                * math.sqrt(ANN_DAYS)
                if t >= 63
                else v20
            )
            vhat = max(v20, 0.5 * v20 + 0.5 * v63, 0.02)
            raw_scale = float(np.clip(vol_target_ann / vhat, 0.45, 1.35))
            if dd_brake:
                cur_dd = -(nav_run / hwm_run - 1.0)
                dd_s = float(np.clip(1.0 - max(cur_dd - 0.035, 0.0) / 0.085, 0.50, 1.0))
                raw_scale *= dd_s
            if (
                abs(raw_scale - gross_scale_state) / max(gross_scale_state, 1e-6)
                > vol_deadband
            ):
                gross_scale_state = 0.35 * raw_scale + 0.65 * gross_scale_state

        tgt_gross_t = base_gross * gross_scale_state
        sc_t = S_raw[t]
        act = np.isfinite(sc_t)
        n_act = int(act.sum())
        if n_act < 20:
            continue

        z_sc = sc_t[act]
        sd_z = float(np.std(z_sc))
        if sd_z < 1e-12:
            W[t] = w_dr_full
            continue
        z_sc = (z_sc - float(np.mean(z_sc))) / sd_z

        if tail_deadzone > 0.0:
            ts = np.sign(z_sc) * np.maximum(np.abs(z_sc) - tail_deadzone, 0.0)
            ts_sd = float(np.std(ts))
            if ts_sd > 1e-12:
                ts = (ts - float(np.mean(ts))) / ts_sd
        else:
            ts = z_sc

        sig = np.clip(np.nan_to_num(sigma_np[t, act], nan=0.02), 0.005, 0.20)
        alpha = ic_daily * sig * ts

        raw_invvol_sum = float(np.sum(np.abs(ts) / sig))
        gam_t = max(ic_daily * raw_invvol_sum / max(tgt_gross_t, 0.15), 1.0)

        beta = np.nan_to_num(beta_np[t, act], nan=1.0)
        w0 = w_dr_full[act]

        c_rows = []
        if neutralize_sectors:
            Sa = S_onehot[:, act]
            non_empty = Sa.sum(axis=1) > 0
            c_rows.append(Sa[non_empty])
        else:
            c_rows.append(np.ones((1, n_act), dtype=np.float64))
        c_rows.append((beta - beta.mean()).reshape(1, -1))
        if neutralize_vol:
            vz = (sig - sig.mean()) / max(float(sig.std()), 1e-6)
            c_rows.append(vz.reshape(1, -1))
        if neutralize_mom and mom_np is not None:
            m_a = np.nan_to_num(mom_np[t, act], nan=0.0)
            mz = (m_a - m_a.mean()) / max(float(m_a.std()), 1e-6)
            c_rows.append(mz.reshape(1, -1))
        C = np.vstack(c_rows)
        K = C.shape[0]

        diag_q = gam_t * (sig**2)
        rho = max(float(np.median(diag_q)) * 1.5, 1e-3)
        inv_D = 1.0 / (diag_q + rho)
        CDinv = C * inv_D[None, :]
        M = CDinv @ C.T
        M[np.arange(K), np.arange(K)] += 1e-10
        L_chol = np.linalg.cholesky(M)

        lo = np.full(n_act, -max_single_w, dtype=np.float64)
        hi = np.full(n_act, max_single_w, dtype=np.float64)
        if max_dw_mat is not None and t > 0:
            mdw = max_dw_mat[t, act]
            lo = np.maximum(lo, w0 - mdw)
            hi = np.minimum(hi, w0 + mdw)

        w = w0.copy()
        z = w0.copy()
        u = np.zeros(n_act, dtype=np.float64)
        denom_z = l2_cost + rho
        tau = l1_cost / denom_z
        rho_frac = rho / denom_z
        eff_max_gross = (
            min(max_gross, tgt_gross_t * 1.15)
            if vol_target_ann is not None
            else max_gross
        )

        for _ in range(n_iter):
            rhs = alpha + rho * (z - u)
            d_rhs = inv_D * rhs
            y_sol = np.linalg.solve(L_chol, C @ d_rhs)
            nu = np.linalg.solve(L_chol.T, y_sol)
            w = d_rhs - inv_D * (C.T @ nu)

            v = rho_frac * (w + u - w0)
            z_next = w0 + np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)
            z_next = np.clip(z_next, lo, hi)
            gr = float(np.abs(z_next).sum())
            if gr > eff_max_gross:
                z_next *= eff_max_gross / gr
            u = u + (w - z_next)
            z = z_next

        w_full = np.zeros(N, dtype=np.float64)
        w_full[act] = z
        W[t] = w_full
    return W


def solve_admm_turnover_projection(
    W_tar: np.ndarray,
    beta_np: np.ndarray,
    sectors: np.ndarray,
    r1_np: np.ndarray,
    adv_np: np.ndarray | None = None,
    l1_weight: float = 0.0012,
    max_single_w: float = 0.025,
    max_gross: float = 1.80,
    max_adv_frac: float = 0.05,
    neutralize_sectors: bool = True,
    n_iter: int = 25,
) -> np.ndarray:
    """Turnover-penalized projection: min 0.5||w - W_tar||_2^2 + l1*||w - w_drift||_1 s.t. Cw=0, box, ADV."""
    T, N = W_tar.shape
    uniq_secs = np.unique(sectors)
    K_sec = len(uniq_secs)
    sec_idx = np.searchsorted(uniq_secs, sectors)
    S_onehot = np.zeros((K_sec, N), dtype=np.float64)
    S_onehot[sec_idx, np.arange(N)] = 1.0

    max_dw_mat = None
    if adv_np is not None:
        ADV = np.maximum(np.nan_to_num(adv_np, nan=50e6), 1e6)
        max_dw_mat = (max_adv_frac * ADV) / BOOK_SIZE

    W = np.zeros((T, N), dtype=np.float64)
    rho = 1.5
    inv_d = 1.0 / (1.0 + rho)
    tau = l1_weight / rho

    for t in range(T):
        if t == 0:
            w_dr_full = np.zeros(N, dtype=np.float64)
        else:
            gp = float(np.dot(W[t - 1], r1_np[t - 1]))
            w_dr_full = W[t - 1] * (1.0 + r1_np[t - 1]) / max(1.0 + gp, 0.25)

        wt_full = np.nan_to_num(W_tar[t], nan=0.0)
        act = (np.abs(wt_full) > 1e-10) | (np.abs(w_dr_full) > 1e-10)
        n_act = int(act.sum())
        if n_act < 15:
            continue

        wt = wt_full[act]
        w0 = w_dr_full[act]
        beta = np.nan_to_num(beta_np[t, act], nan=1.0)
        if neutralize_sectors:
            Sa = S_onehot[:, act]
            non_empty = Sa.sum(axis=1) > 0
            C = np.vstack([Sa[non_empty], (beta - beta.mean()).reshape(1, -1)])
        else:
            C = np.vstack([np.ones((1, n_act)), (beta - beta.mean()).reshape(1, -1)])
        K = C.shape[0]
        M = (C @ C.T) * inv_d
        M[np.arange(K), np.arange(K)] += 1e-10
        L_chol = np.linalg.cholesky(M)

        lo = np.full(n_act, -max_single_w, dtype=np.float64)
        hi = np.full(n_act, max_single_w, dtype=np.float64)
        if max_dw_mat is not None and t > 0:
            mdw = max_dw_mat[t, act]
            lo = np.maximum(lo, w0 - mdw)
            hi = np.minimum(hi, w0 + mdw)

        w = w0.copy()
        z = w0.copy()
        u = np.zeros(n_act, dtype=np.float64)
        for _ in range(n_iter):
            rhs = wt + rho * (z - u)
            d_rhs = inv_d * rhs
            y_sol = np.linalg.solve(L_chol, C @ d_rhs)
            nu = np.linalg.solve(L_chol.T, y_sol)
            w = d_rhs - inv_d * (C.T @ nu)

            v = w + u - w0
            z_next = w0 + np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)
            z_next = np.clip(z_next, lo, hi)
            gr = float(np.abs(z_next).sum())
            if gr > max_gross:
                z_next *= max_gross / gr
            u = u + (w - z_next)
            z = z_next

        w_full = np.zeros(N, dtype=np.float64)
        w_full[act] = z
        W[t] = w_full
    return W


def apply_smooth_method_g_overlay(
    W0: np.ndarray,
    bundle: V7DataBundle,
    r1_np: np.ndarray,
    adv_np: np.ndarray,
    beta_np: np.ndarray,
    vol_target: float = 0.09,
    scale_deadband: float = 0.10,
    scale_ema: float = 0.30,
    use_dd: bool = True,
    use_cvar: bool = True,
    max_single_w: float = 0.025,
    max_gross: float = 1.80,
    max_abs_beta: float | None = 0.08,
    max_sector_net: float | None = 0.15,
    max_adv_frac: float | None = 0.05,
    post_ntb: float = 0.0,
) -> np.ndarray:
    """Turnover-resilient Method G overlay: hysteresis-smoothed causal Vol/DD/CVaR scaling + constraints + L1 NTB."""
    T, _N = W0.shape
    port_r = np.einsum("ij,ij->i", W0, r1_np)
    r_ser = pd.Series(port_r)
    v20 = r_ser.rolling(20, min_periods=10).std().shift(1).to_numpy(
        dtype=np.float64
    ) * math.sqrt(ANN_DAYS)
    v63 = r_ser.rolling(63, min_periods=20).std().shift(1).to_numpy(
        dtype=np.float64
    ) * math.sqrt(ANN_DAYS)
    vol_hat = np.where(
        np.isfinite(v20) & np.isfinite(v63),
        np.maximum(v20, 0.5 * v20 + 0.5 * v63),
        0.10,
    )
    sc_raw = np.clip(vol_target / np.maximum(vol_hat, 0.025), 0.35, 1.35)
    sc_raw[:25] = 1.0

    if use_dd:
        nav = np.cumprod(1.0 + np.clip(port_r, -0.5, 1.0))
        hwm = np.maximum.accumulate(np.maximum(nav, 1.0))
        dd_lag = np.zeros(T, dtype=np.float64)
        dd_lag[1:] = -(nav[:-1] / hwm[:-1] - 1.0)
        sc_dd = np.clip(1.0 - np.maximum(dd_lag - 0.035, 0.0) / 0.085, 0.45, 1.0)
        sc_raw *= sc_dd

    if use_cvar:
        cvar_lag = np.full(T, 0.01, dtype=np.float64)
        for t in range(65, T):
            win = port_r[t - 63 : t]
            q05 = np.percentile(win, 5.0)
            cvar_lag[t] = max(-float(win[win <= q05].mean()), 1e-4)
        sc_cvar = np.clip(0.0125 / np.maximum(cvar_lag, 0.005), 0.50, 1.0)
        sc_raw *= sc_cvar

    sc_sm = np.ones(T, dtype=np.float64)
    if scale_deadband > 0.0 or scale_ema < 1.0:
        cur = 1.0
        for t in range(25, T):
            if abs(sc_raw[t] - cur) / max(cur, 1e-6) > scale_deadband:
                cur = scale_ema * sc_raw[t] + (1.0 - scale_ema) * cur
            sc_sm[t] = cur
    else:
        sc_sm = sc_raw

    W = apply_portfolio_constraints(
        W0 * sc_sm[:, None],
        beta60_np=beta_np if max_abs_beta is not None else None,
        sectors=bundle.sectors if max_sector_net is not None else None,
        adv20_np=adv_np if max_adv_frac is not None else None,
        r1d_np=r1_np if max_adv_frac is not None else None,
        max_single_w=max_single_w,
        max_gross=max_gross,
        max_abs_beta=max_abs_beta,
        max_sector_net=max_sector_net,
        max_adv_frac=max_adv_frac,
    )

    if post_ntb > 0.0:
        W_out = np.zeros_like(W)
        W_out[0] = W[0]
        for t in range(1, T):
            gp = float(np.dot(W_out[t - 1], r1_np[t - 1]))
            dr = W_out[t - 1] * (1.0 + r1_np[t - 1]) / max(1.0 + gp, 0.25)
            diff = W[t] - dr
            dw = np.sign(diff) * np.maximum(np.abs(diff) - post_ntb, 0.0)
            w = dr + dw
            if max_abs_beta is not None:
                b_t = np.nan_to_num(beta_np[t], nan=1.0)
                pb = float(np.dot(w, b_t))
                if abs(pb) > max_abs_beta * 1.25:
                    act = np.abs(w) > 1e-10
                    if act.sum() >= 10:
                        ba = b_t[act]
                        bc = ba - ba.mean()
                        den = float(np.dot(bc, bc))
                        if den > 1e-10:
                            w[act] -= (
                                (pb - math.copysign(max_abs_beta, pb)) / den
                            ) * bc
            if max_gross is not None:
                g = float(np.abs(w).sum())
                if g > max_gross:
                    w *= max_gross / g
            W_out[t] = w
        W = W_out
    return W
