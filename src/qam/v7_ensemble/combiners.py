"""Sections 5, 6, 7 & 10: Strategy Combination Methods (A-F), Risk Overlays (G), and Ablations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.sparse import csc_matrix, hstack, vstack
from scipy.sparse import eye as sp_eye

from src.qam.v7_ensemble.diagnostics import _non_overlapping_rebal_weights
from src.qam.v7_ensemble.engine import (
    ANN_DAYS,
    BASELINE_COST_BPS,
    CANDIDATES_1D_16,
    CANDIDATES_24,
    EXPERIMENTS_V7,
    STRATEGIES_V7,
    V7DataBundle,
    apply_portfolio_constraints,
    backtest_weights_daily,
    build_8_strategies_for_score,
    consolidate_positions,
    neutralize_beta,
    strat_top10_buf,
)

WARMUP_DAYS = 252  # 2010 warmup year -> strict OOS evaluation 2011-01-04 to 2024-12-31
PURGE_DAYS = 6  # Max prediction purge horizon (matches risk_t5 purge=6)


# ---------------------------------------------------------------------------
# 1. Statistical & Optimization Primitives
# ---------------------------------------------------------------------------
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


def _project_simplex_bounded(
    w: np.ndarray, w_max: float = 1.0, w_min: float = 0.0
) -> np.ndarray:
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
            if room.any():
                w[room] += diff * (
                    (w_max - w[room]) / max(float((w_max - w[room]).sum()), 1e-12)
                )
            else:
                break
        else:
            room = w > w_min + 1e-9
            if room.any():
                w[room] += diff * (
                    (w[room] - w_min) / max(float((w[room] - w_min).sum()), 1e-12)
                )
            else:
                break
    return w / max(float(w.sum()), 1e-12)


def solve_inv_vol(X: np.ndarray, w_max: float = 0.40) -> np.ndarray:
    sd = np.nanstd(X, axis=0, ddof=1)
    iv = 1.0 / np.maximum(sd, 1e-5)
    return _project_simplex_bounded(iv / iv.sum(), w_max=w_max)


def solve_erc(cov: np.ndarray, w_max: float = 0.40) -> np.ndarray:
    """Exact Equal Risk Contribution via SLSQP."""
    N = cov.shape[0]
    if N == 1:
        return np.array([1.0])
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    w0 = _project_simplex_bounded((1.0 / sd) / np.sum(1.0 / sd), w_max=w_max)
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
    return _project_simplex_bounded(res.x if res.success else w0, w_max=w_max)


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
    return _project_simplex_bounded(res.x if res.success else w0, w_max=w_max)


def solve_max_sharpe(
    mu: np.ndarray, cov: np.ndarray, w_max: float = 0.35, rf_daily: float = 0.0
) -> np.ndarray:
    N = len(mu)
    ex_mu = (mu - rf_daily) * ANN_DAYS
    cov_ann = cov * ANN_DAYS
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
    return _project_simplex_bounded(res.x if res.success else w0, w_max=w_max)


def solve_mean_cvar_lp(
    X: np.ndarray,
    mu: np.ndarray,
    alpha: float = 0.95,
    w_max: float = 0.35,
    min_ret_frac: float = 0.50,
) -> np.ndarray:
    """Exact Rockafellar-Uryasev (2000) Mean-CVaR LP via SciPy HiGHS."""
    T, N = X.shape
    R = np.nan_to_num(X.astype(np.float64), nan=0.0)
    mu_eq = float(np.mean(mu))
    mu_hi = float(np.max(mu))
    mu_lo = float(np.min(mu))
    mu_target = min(
        mu_lo + min_ret_frac * max(mu_hi - mu_lo, 0.0), 0.85 * mu_hi + 0.15 * mu_eq
    )

    n_vars = N + 1 + T
    c = np.zeros(n_vars, dtype=np.float64)
    c[N] = 1.0
    c[N + 1 :] = 1.0 / ((1.0 - alpha) * T)

    A_tail = hstack(
        [
            csc_matrix(-R),
            csc_matrix(-np.ones((T, 1))),
            -sp_eye(T, format="csc"),
        ]
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
        return _project_simplex_bounded(res.x[:N], w_max=w_max)
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
) -> np.ndarray:
    """Regularized mean-variance with L2 diversification, L1 allocation turnover, & std-error penalty."""
    N = len(mu)
    mu_ann = mu * ANN_DAYS
    cov_ann = cov * ANN_DAYS
    sd_ann = np.sqrt(np.maximum(np.diag(cov_ann), 1e-8))
    w0 = _project_simplex_bounded(w_prev.copy(), w_max=w_max)
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
    return _project_simplex_bounded(res.x if res.success else w0, w_max=w_max)


# ---------------------------------------------------------------------------
# 2. Execution Helpers (Consolidated Positions -> Constraints -> Backtest)
# ---------------------------------------------------------------------------
def run_consolidated_strategy(
    name: str,
    group: str,
    alloc_schedule: pd.DataFrame,
    cand_weights: dict[str, np.ndarray],
    bundle: V7DataBundle,
    max_single_w: float | None = 0.05,
    cost_bps: float = BASELINE_COST_BPS,
    apply_full_constraints: bool = False,
) -> dict[str, Any]:
    """Consolidate sleeve weights daily, net internal trades, apply optional constraints, and backtest."""
    W = consolidate_positions(alloc_schedule, cand_weights)
    if apply_full_constraints:
        W = apply_portfolio_constraints(
            W,
            beta60_np=bundle.beta60_df.to_numpy(dtype=np.float64),
            sectors=bundle.sectors,
            adv20_np=bundle.adv20_df.to_numpy(dtype=np.float64),
            r1d_np=bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64),
            max_single_w=0.025,
            max_gross=1.80,
            max_abs_beta=0.08,
            max_sector_net=0.15,
            max_adv_frac=0.05,
        )
    elif max_single_w is not None:
        W = apply_portfolio_constraints(W, max_single_w=max_single_w)

    bt = backtest_weights_daily(
        W,
        bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64),
        bundle.dates,
        cost_bps=cost_bps,
        adv20_np=bundle.adv20_df.to_numpy(dtype=np.float64),
        sigma_np=bundle.sigma_prev_df.to_numpy(dtype=np.float64),
        rf_np=bundle.ff6_df["rf"].to_numpy(dtype=np.float64),
        inception_zero_trade=False,
    )
    return {
        "name": name,
        "group": group,
        "schedule": alloc_schedule,
        "weights": W,
        "bt": bt,
    }


def _static_schedule(
    dates: pd.DatetimeIndex, weights_map: dict[str, float]
) -> pd.DataFrame:
    df = pd.DataFrame(0.0, index=dates, columns=list(weights_map.keys()))
    for k, v in weights_map.items():
        df[k] = float(v)
    return df


def _causal_wf_schedule(
    ret_mat: pd.DataFrame,
    dates: pd.DatetimeIndex,
    solve_fn,
    rebal_freq: int = 21,
    warmup: int = WARMUP_DAYS,
    purge: int = PURGE_DAYS,
    roll_window: int | None = None,
    ema_alpha: float = 1.0,
    hyst_tau: float = 0.0,
) -> pd.DataFrame:
    """Strictly causal expanding or rolling allocation schedule over candidates."""
    cols = list(ret_mat.columns)
    R = ret_mat.to_numpy(dtype=np.float64)
    T, N = R.shape
    sched = np.zeros((T, N), dtype=np.float64)
    w_curr = np.ones(N, dtype=np.float64) / N
    sched[:warmup] = w_curr

    for t in range(warmup, T):
        if t == warmup or ((t - warmup) % rebal_freq == 0):
            end_idx = max(t - purge, 20)
            st_idx = 0 if roll_window is None else max(0, end_idx - roll_window)
            W_hist = R[st_idx:end_idx]
            if len(W_hist) >= 20:
                w_target = np.asarray(solve_fn(W_hist, w_curr, t), dtype=np.float64)
                w_target = np.clip(np.nan_to_num(w_target, nan=0.0), 0.0, 1.0)
                w_target /= max(float(w_target.sum()), 1e-12)
                if t > warmup and ema_alpha < 1.0:
                    w_blend = ema_alpha * w_target + (1.0 - ema_alpha) * w_curr
                    w_blend /= max(float(w_blend.sum()), 1e-12)
                else:
                    w_blend = w_target
                if (
                    t > warmup
                    and hyst_tau > 0.0
                    and float(np.abs(w_blend - w_curr).sum()) < hyst_tau
                ):
                    pass  # Keep w_curr inside hysteresis no-trade band
                else:
                    w_curr = w_blend
        sched[t] = w_curr
    return pd.DataFrame(sched, index=dates, columns=cols)


def _select_orthogonal_subset(
    ret_df: pd.DataFrame, warmup: int = WARMUP_DAYS, corr_thresh: float = 0.88
) -> list[str]:
    """Greedy diversity selection strictly using [0:warmup] net returns."""
    sub = ret_df.iloc[:warmup]
    sharpe_w = (sub.mean() / (sub.std(ddof=1) + 1e-12)) * math.sqrt(ANN_DAYS)
    order = sharpe_w.sort_values(ascending=False).index.tolist()
    corr = sub.corr()
    chosen: list[str] = []
    for cid in order:
        if not chosen:
            chosen.append(cid)
            continue
        max_c = max(float(corr.loc[cid, c]) for c in chosen)
        if max_c < corr_thresh:
            chosen.append(cid)
    # Ensure at least one representative from each prediction experiment
    for exp in EXPERIMENTS_V7:
        if not any(c.startswith(f"{exp}__") for c in chosen):
            exp_cands = [c for c in order if c.startswith(f"{exp}__")]
            if exp_cands:
                chosen.append(exp_cands[0])
    return sorted(chosen)


# ---------------------------------------------------------------------------
# 3. Section 5: Methods A - F
# ---------------------------------------------------------------------------
def build_method_a_static(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    cand_bts: dict[str, dict[str, pd.Series]],
    ret_net_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Method A: Static Equal-Weight, Within/Across-Family, Priority Pairs & Corrected 5d Rebal."""
    dates = bundle.dates
    out: list[dict[str, Any]] = []

    def add_static(name: str, group: str, wmap: dict[str, float]) -> None:
        sched = _static_schedule(dates, wmap)
        out.append(run_consolidated_strategy(name, group, sched, cand_weights, bundle))

    add_static(
        "A1_EW_All24",
        "Method_A_Static",
        {c: 1.0 / len(CANDIDATES_24) for c in CANDIDATES_24},
    )
    add_static(
        "A2_EW_1d16",
        "Method_A_Static",
        {c: 1.0 / len(CANDIDATES_1D_16) for c in CANDIDATES_1D_16},
    )

    for exp in EXPERIMENTS_V7:
        cols = [f"{exp}__{s}" for s in STRATEGIES_V7]
        add_static(
            f"A3_EW_Exp_{exp}", "Method_A_Static", {c: 1.0 / len(cols) for c in cols}
        )

    for fam in (
        "rank_RAW",
        "top10_InvVol",
        "top10_InvVol_Buf",
        "top10_BetaNeu_InvVol_Buf",
        "top10_HOLD5",
        "top10_InvVol_HOLD5",
    ):
        cols = [f"{e}__{fam}" for e in EXPERIMENTS_V7]
        add_static(
            f"A4_EW_Fam_{fam}", "Method_A_Static", {c: 1.0 / len(cols) for c in cols}
        )

    # Three Section-5A Priority Pairs
    add_static(
        "A5_Pair_rawInvVol_riskBetaNeu",
        "Method_A_Static",
        {"raw_t1__top10_InvVol": 0.50, "risk_t1__top10_BetaNeu_InvVol_Buf": 0.50},
    )
    add_static(
        "A6_Pair_riskBetaNeu_risk5HOLD5",
        "Method_A_Static",
        {
            "risk_t1__top10_BetaNeu_InvVol_Buf": 0.50,
            "risk_t5__top10_InvVol_HOLD5": 0.50,
        },
    )
    add_static(
        "A7_Pair_rawBuf_riskBetaNeu",
        "Method_A_Static",
        {"raw_t1__top10_Buf10_20": 0.50, "risk_t1__top10_BetaNeu_InvVol_Buf": 0.50},
    )

    # Discovered lowest-correlation complementary strong pair strictly from [0:WARMUP_DAYS]
    w_ret = ret_net_df.iloc[:WARMUP_DAYS]
    w_sh = (w_ret.mean() / (w_ret.std(ddof=1) + 1e-12)) * math.sqrt(ANN_DAYS)
    top_pool = w_sh[w_sh > 0.6].index.tolist() or list(CANDIDATES_24)
    cmat = w_ret[top_pool].corr()
    best_pair = ("raw_t1__top10_BetaNeu_InvVol_Buf", "risk_t5__top10_InvVol_HOLD5")
    best_score = -1e9
    for i, c1 in enumerate(top_pool):
        for c2 in top_pool[i + 1 :]:
            if c1.split("__")[0] == c2.split("__")[0]:
                continue
            sc = 0.5 * (float(w_sh[c1]) + float(w_sh[c2])) - 1.5 * float(
                cmat.loc[c1, c2]
            )
            if sc > best_score:
                best_score = sc
                best_pair = (c1, c2)
    add_static(
        "A8_Discovered_LowCorr_Pair",
        "Method_A_Static",
        {best_pair[0]: 0.50, best_pair[1]: 0.50},
    )

    # Separate corrected 5-day rebalance evaluation pairing risk_t5 & 1d core on matched 5-day cadence
    W_5d_t5 = _non_overlapping_rebal_weights(
        pd.DataFrame(
            cand_weights["risk_t5__top10_InvVol_HOLD5"],
            index=dates,
            columns=bundle.security_ids,
        ),
        bundle.ret1d_df,
        stride=5,
    ).to_numpy(dtype=np.float64)
    W_5d_t1 = _non_overlapping_rebal_weights(
        pd.DataFrame(
            cand_weights["risk_t1__top10_BetaNeu_InvVol_Buf"],
            index=dates,
            columns=bundle.security_ids,
        ),
        bundle.ret1d_df,
        stride=5,
    ).to_numpy(dtype=np.float64)
    W_5d_blend = apply_portfolio_constraints(
        0.50 * W_5d_t5 + 0.50 * W_5d_t1, max_single_w=0.05
    )
    bt_5d = backtest_weights_daily(
        W_5d_blend,
        bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64),
        dates,
        cost_bps=BASELINE_COST_BPS,
        adv20_np=bundle.adv20_df.to_numpy(dtype=np.float64),
        sigma_np=bundle.sigma_prev_df.to_numpy(dtype=np.float64),
        rf_np=bundle.ff6_df["rf"].to_numpy(dtype=np.float64),
    )
    out.append(
        {
            "name": "A9_Corr5dRebal_Pair_t5_t1",
            "group": "Method_A_5dRebal",
            "schedule": None,
            "weights": W_5d_blend,
            "bt": bt_5d,
        }
    )
    return out


def build_method_b_score_ensembles(
    bundle: V7DataBundle, tab_dir: Path
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Method B: Prediction-level score combinations fed through all 8 portfolio constructions."""
    dates = bundle.dates
    r1_df = bundle.ret1d_df
    r1_np = r1_df.fillna(0.0).to_numpy(dtype=np.float64)
    sig_df = bundle.sigma_prev_df
    beta_df = bundle.beta60_df
    adv_np = bundle.adv20_df.to_numpy(dtype=np.float64)
    sig_np = sig_df.to_numpy(dtype=np.float64)
    rf_np = bundle.ff6_df["rf"].to_numpy(dtype=np.float64)

    def _xs_z(df: pd.DataFrame) -> pd.DataFrame:
        mu = df.mean(axis=1)
        sd = df.std(axis=1).clip(lower=1e-8)
        return df.sub(mu, axis=0).div(sd, axis=0)

    def _xs_rank(df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, pct=True) - 0.5

    z_raw = _xs_z(bundle.scores["raw_t1"])
    z_risk1 = _xs_z(bundle.scores["risk_t1"])
    z_risk5 = _xs_z(bundle.scores["risk_t5"])

    rk_raw = _xs_rank(bundle.scores["raw_t1"])
    rk_risk1 = _xs_rank(bundle.scores["risk_t1"])
    rk_risk5 = _xs_rank(bundle.scores["risk_t5"])

    # Causal WF optimization of lambda * z_raw + (1-lambda) * z_risk1 every 63 days
    lam_grid = [0.0, 0.25, 0.40, 0.50, 0.60, 0.75, 1.0]
    grid_rets: dict[float, np.ndarray] = {}
    for lam in lam_grid:
        sc_g = lam * z_raw + (1.0 - lam) * z_risk1
        inv_vol_g = (1.0 / sig_df).where(sc_g.notna()).fillna(0.0)
        w_g = strat_top10_buf(neutralize_beta(sc_g, beta_df), inv_vol_g).to_numpy(
            dtype=np.float64
        )
        bt_g = backtest_weights_daily(w_g, r1_np, dates, cost_bps=BASELINE_COST_BPS)
        grid_rets[lam] = bt_g["net"].to_numpy(dtype=np.float64)

    T = len(dates)
    lam_series = np.full(T, 0.50, dtype=np.float64)
    for t in range(WARMUP_DAYS, T):
        if t == WARMUP_DAYS or ((t - WARMUP_DAYS) % 63 == 0):
            end_i = max(t - PURGE_DAYS, 20)
            best_lam, best_sh = 0.50, -1e9
            for lam in lam_grid:
                seg = grid_rets[lam][:end_i]
                sh = float(np.mean(seg) / (np.std(seg, ddof=1) + 1e-12))
                if sh > best_sh:
                    best_sh = sh
                    best_lam = lam
            lam_series[t:] = best_lam

    sc_wf_opt = z_raw.mul(lam_series, axis=0) + z_risk1.mul(1.0 - lam_series, axis=0)

    blended_scores = {
        "B1_ScoreEW_Rank_1d": 0.5 * rk_raw + 0.5 * rk_risk1,
        "B2_ScoreEW_Z_1d": 0.5 * z_raw + 0.5 * z_risk1,
        "B3_ScoreWF_Opt_1d": sc_wf_opt,
        "B4_ScoreEW_Z_All3": (z_raw + z_risk1 + z_risk5) / 3.0,
        "B5_ScoreEW_Rank_All3": (rk_raw + rk_risk1 + rk_risk5) / 3.0,
    }

    decomp_rows = []
    # Include standalone baselines in decomposition table
    all_score_sets: dict[str, pd.DataFrame] = {
        "Standalone_raw_t1": bundle.scores["raw_t1"],
        "Standalone_risk_t1": bundle.scores["risk_t1"],
        "Standalone_risk_t5": bundle.scores["risk_t5"],
        **blended_scores,
    }
    promoted: list[dict[str, Any]] = []
    key_constructions = (
        "top10_InvVol",
        "top10_InvVol_Buf",
        "top10_BetaNeu_InvVol_Buf",
        "top10_InvVol_HOLD5",
    )

    for sc_name, sc_df in all_score_sets.items():
        w_dict = build_8_strategies_for_score(sc_df, r1_df, sig_df, beta_df)
        for sname in STRATEGIES_V7:
            W_np = w_dict[sname].to_numpy(dtype=np.float64)
            bt = backtest_weights_daily(
                W_np,
                r1_np,
                dates,
                cost_bps=BASELINE_COST_BPS,
                adv20_np=adv_np,
                sigma_np=sig_np,
                rf_np=rf_np,
            )
            r_oos = bt["net"].iloc[WARMUP_DAYS:]
            nav = (1.0 + r_oos).cumprod()
            mdd = float(
                -(nav / np.maximum.accumulate(np.maximum(nav, 1.0)) - 1.0).min() * 100.0
            )
            cagr = float((nav.iloc[-1] ** (ANN_DAYS / len(r_oos)) - 1.0) * 100.0)
            sh = float(r_oos.mean() / (r_oos.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS))
            tvr = float(bt["traded"].iloc[WARMUP_DAYS:].mean() * 100.0)
            decomp_rows.append(
                {
                    "Score_Source": sc_name,
                    "Construction": sname,
                    "Is_Blended": not sc_name.startswith("Standalone_"),
                    "OOS_Net2_Sharpe": round(sh, 4),
                    "OOS_CAGR(%)": round(cagr, 2),
                    "OOS_MaxDD(%)": round(mdd, 2),
                    "OOS_Turnover(%)": round(tvr, 2),
                }
            )
            if sc_name in blended_scores and sname in key_constructions:
                promoted.append(
                    {
                        "name": f"{sc_name}__{sname}",
                        "group": "Method_B_ScoreBlend",
                        "schedule": None,
                        "weights": W_np,
                        "bt": bt,
                    }
                )

    decomp_df = pd.DataFrame(decomp_rows)
    decomp_df.to_csv(
        tab_dir / "tab05_methodB_score_vs_construction_decomposition.csv", index=False
    )
    return promoted, decomp_df


def build_method_d_risk_and_opt(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    ret_net_df: pd.DataFrame,
    ortho_cols: list[str],
) -> list[dict[str, Any]]:
    """Method D: Causal Expanding Risk-Based & Return-Risk Strategy Optimization."""
    dates = bundle.dates
    out: list[dict[str, Any]] = []
    pools = {
        "1d16": list(CANDIDATES_1D_16),
        "All24": list(CANDIDATES_24),
        "Ortho": ortho_cols,
    }
    w_caps = {"1d16": 0.25, "All24": 0.22, "Ortho": 0.35}

    for pname, cols in pools.items():
        R_df = ret_net_df[cols]
        w_max = w_caps[pname]

        sched_iv = _causal_wf_schedule(
            R_df, dates, lambda X, wp, t, m=w_max: solve_inv_vol(X, w_max=m)
        )
        out.append(
            run_consolidated_strategy(
                f"D1_ExpInvVol_{pname}",
                "Method_D_RiskOpt",
                sched_iv,
                cand_weights,
                bundle,
            )
        )

        _causal_wf_schedule(
            R_df, dates, lambda X, wp, t, m=w_max: solve_inv_vol(X, w_max=m)
        )  # Diagonal Risk Parity = InvVol
        sched_erc = _causal_wf_schedule(
            R_df,
            dates,
            lambda X, wp, t, m=w_max: solve_erc(ledoit_wolf_cov(X), w_max=m),
        )
        out.append(
            run_consolidated_strategy(
                f"D2_ExpERC_LW_{pname}",
                "Method_D_RiskOpt",
                sched_erc,
                cand_weights,
                bundle,
            )
        )

        if pname in ("1d16", "Ortho"):
            sched_mv = _causal_wf_schedule(
                R_df,
                dates,
                lambda X, wp, t, m=w_max: solve_min_var(
                    ledoit_wolf_cov(X), w_max=m, l2_reg=0.10
                ),
            )
            out.append(
                run_consolidated_strategy(
                    f"D3_ExpMinVar_LW_{pname}",
                    "Method_D_RiskOpt",
                    sched_mv,
                    cand_weights,
                    bundle,
                )
            )

            sched_cs = _causal_wf_schedule(
                R_df,
                dates,
                lambda X, wp, t, m=w_max: solve_min_var(
                    cov_corr_shrink(X, 0.30), w_max=m, l2_reg=0.10
                ),
            )
            out.append(
                run_consolidated_strategy(
                    f"D4_ExpCovCorrShrink_{pname}",
                    "Method_D_RiskOpt",
                    sched_cs,
                    cand_weights,
                    bundle,
                )
            )

            sched_sh = _causal_wf_schedule(
                R_df,
                dates,
                lambda X, wp, t, m=w_max: solve_max_sharpe(
                    bayes_stein_mu(X, ledoit_wolf_cov(X)), ledoit_wolf_cov(X), w_max=m
                ),
            )
            out.append(
                run_consolidated_strategy(
                    f"D5_ExpMaxSharpe_BS_{pname}",
                    "Method_D_ReturnRisk",
                    sched_sh,
                    cand_weights,
                    bundle,
                )
            )

            sched_cvar = _causal_wf_schedule(
                R_df,
                dates,
                lambda X, wp, t, m=w_max: solve_mean_cvar_lp(
                    X[-504:], bayes_stein_mu(X, ledoit_wolf_cov(X)), w_max=m
                ),
            )
            out.append(
                run_consolidated_strategy(
                    f"D6_ExpMeanCVaR95_LP_{pname}",
                    "Method_D_ReturnRisk",
                    sched_cvar,
                    cand_weights,
                    bundle,
                )
            )

            sched_reg = _causal_wf_schedule(
                R_df,
                dates,
                lambda X, wp, t, m=w_max: solve_regularized_return_risk(
                    bayes_stein_mu(X, ledoit_wolf_cov(X)),
                    ledoit_wolf_cov(X),
                    wp,
                    w_max=m,
                ),
            )
            out.append(
                run_consolidated_strategy(
                    f"D7_ExpRegRetRisk_{pname}",
                    "Method_D_ReturnRisk",
                    sched_reg,
                    cand_weights,
                    bundle,
                )
            )
    return out


def compute_dynamic_persistence_precheck(
    ret_net_df: pd.DataFrame, tab_dir: Path
) -> pd.DataFrame:
    """Section 5.E pre-check: test whether past strategy Sharpe predicts future Sharpe (momentum vs MR)."""
    rows = []
    R = ret_net_df.to_numpy(dtype=np.float64)
    T, _N = R.shape
    for lb in (21, 63, 126, 252):
        for fwd in (21, 63):
            cs_corrs = []
            top_f_sh, bot_f_sh = [], []
            for t in range(lb + PURGE_DAYS, T - fwd, fwd):
                past = R[t - lb - PURGE_DAYS : t - PURGE_DAYS]
                fut = R[t : t + fwd]
                p_sh = (
                    np.mean(past, axis=0)
                    / (np.std(past, axis=0, ddof=1) + 1e-12)
                    * math.sqrt(ANN_DAYS)
                )
                f_sh = (
                    np.mean(fut, axis=0)
                    / (np.std(fut, axis=0, ddof=1) + 1e-12)
                    * math.sqrt(ANN_DAYS)
                )
                if np.std(p_sh) > 1e-9 and np.std(f_sh) > 1e-9:
                    cs_corrs.append(float(np.corrcoef(p_sh, f_sh)[0, 1]))
                rk = np.argsort(p_sh)
                top_f_sh.append(float(np.mean(f_sh[rk[-6:]])))
                bot_f_sh.append(float(np.mean(f_sh[rk[:6]])))
            arr = np.asarray(cs_corrs, dtype=np.float64)
            rows.append(
                {
                    "Lookback(d)": lb,
                    "Forward(d)": fwd,
                    "N_Periods": len(arr),
                    "Mean_RankIC": round(float(arr.mean()), 4),
                    "IC_tstat": round(
                        float(
                            arr.mean()
                            / (arr.std(ddof=1) / math.sqrt(max(len(arr), 1)) + 1e-12)
                        ),
                        2,
                    ),
                    "Pos_IC_Frac(%)": round(float((arr > 0).mean() * 100.0), 1),
                    "TopQuartile_FwdSh": round(float(np.mean(top_f_sh)), 3),
                    "BotQuartile_FwdSh": round(float(np.mean(bot_f_sh)), 3),
                    "Spread_TopMinusBot": round(
                        float(np.mean(top_f_sh) - np.mean(bot_f_sh)), 3
                    ),
                    "Verdict": "Persistence"
                    if np.mean(arr) > 0.03
                    else ("MeanReversion" if np.mean(arr) < -0.03 else "Neutral"),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "tab05_dynamic_persistence_precheck.csv", index=False)
    return df


def build_method_e_dynamic(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    ret_net_df: pd.DataFrame,
    ortho_cols: list[str],
) -> list[dict[str, Any]]:
    """Method E: Causal Dynamic Allocations with EMA smoothing, bounds & hysteresis."""
    dates = bundle.dates
    R_df = ret_net_df[ortho_cols]
    N = len(ortho_cols)
    out: list[dict[str, Any]] = []

    # Causal market state signals aligned strictly prior to day t
    ew_mkt = bundle.ew_mkt_1d.reindex(dates).fillna(0.0).to_numpy(dtype=np.float64)
    vol20_mkt = pd.Series(ew_mkt).rolling(20).std().to_numpy(
        dtype=np.float64
    ) * math.sqrt(ANN_DAYS)
    xs_disp = (
        bundle.ret1d_df.std(axis=1, skipna=True)
        .rolling(20)
        .mean()
        .to_numpy(dtype=np.float64)
    )

    sched_riv = _causal_wf_schedule(
        R_df,
        dates,
        lambda X, wp, t: solve_inv_vol(X, w_max=0.35),
        roll_window=63,
        ema_alpha=0.35,
        hyst_tau=0.015,
    )
    out.append(
        run_consolidated_strategy(
            "E1_RollInvVol_63d_Hyst",
            "Method_E_Dynamic",
            sched_riv,
            cand_weights,
            bundle,
        )
    )

    sched_rerc = _causal_wf_schedule(
        R_df,
        dates,
        lambda X, wp, t: solve_erc(ledoit_wolf_cov(X), w_max=0.35),
        roll_window=126,
        ema_alpha=0.30,
        hyst_tau=0.015,
    )
    out.append(
        run_consolidated_strategy(
            "E2_RollERC_126d_Hyst", "Method_E_Dynamic", sched_rerc, cand_weights, bundle
        )
    )

    sched_rmv = _causal_wf_schedule(
        R_df,
        dates,
        lambda X, wp, t: solve_min_var(ledoit_wolf_cov(X), w_max=0.35, l2_reg=0.15),
        roll_window=126,
        ema_alpha=0.30,
        hyst_tau=0.015,
    )
    out.append(
        run_consolidated_strategy(
            "E3_RollMinVar_126d_Hyst",
            "Method_E_Dynamic",
            sched_rmv,
            cand_weights,
            bundle,
        )
    )

    def _roll_perf_solve(X: np.ndarray, wp: np.ndarray, t: int) -> np.ndarray:
        sh = (
            np.mean(X, axis=0)
            / (np.std(X, axis=0, ddof=1) + 1e-12)
            * math.sqrt(ANN_DAYS)
        )
        iv = solve_inv_vol(X, w_max=0.35)
        tilt = np.exp(0.35 * np.clip(sh - np.median(sh), -2.0, 2.0))
        return _project_simplex_bounded(iv * tilt, w_max=0.35, w_min=0.02)

    sched_perf = _causal_wf_schedule(
        R_df,
        dates,
        _roll_perf_solve,
        roll_window=63,
        ema_alpha=0.30,
        hyst_tau=0.02,
    )
    out.append(
        run_consolidated_strategy(
            "E4_RollPerfTilt_63d", "Method_E_Dynamic", sched_perf, cand_weights, bundle
        )
    )

    def _dd_aware_solve(X: np.ndarray, wp: np.ndarray, t: int) -> np.ndarray:
        base = solve_erc(ledoit_wolf_cov(X), w_max=0.35)
        nav = np.cumprod(1.0 + np.clip(X, -0.5, 1.0), axis=0)
        hwm = np.maximum.accumulate(np.maximum(nav, 1.0), axis=0)
        cur_dd = -(nav[-1] / hwm[-1] - 1.0)
        pen = np.exp(-8.0 * np.clip(cur_dd, 0.0, 0.25))
        return _project_simplex_bounded(base * pen, w_max=0.35, w_min=0.02)

    sched_dd = _causal_wf_schedule(
        R_df,
        dates,
        _dd_aware_solve,
        roll_window=126,
        ema_alpha=0.30,
        hyst_tau=0.015,
    )
    out.append(
        run_consolidated_strategy(
            "E5_DrawdownAware_ERC", "Method_E_Dynamic", sched_dd, cand_weights, bundle
        )
    )

    low_turn_mask = np.array(
        ["HOLD5" in c or "Buf" in c or "rank_RAW" in c for c in ortho_cols], dtype=float
    )

    def _regime_solve(X: np.ndarray, wp: np.ndarray, t: int) -> np.ndarray:
        base = solve_erc(ledoit_wolf_cov(X), w_max=0.35)
        v_hist = vol20_mkt[: max(t - 1, 20)]
        d_hist = xs_disp[: max(t - 1, 20)]
        v_now = vol20_mkt[t - 1]
        d_now = xs_disp[t - 1]
        hi_vol = np.isfinite(v_now) and v_now > np.nanpercentile(v_hist, 75)
        hi_disp = np.isfinite(d_now) and d_now > np.nanpercentile(d_hist, 65)
        tilt = np.ones(N, dtype=np.float64)
        if hi_vol:
            tilt = np.where(low_turn_mask > 0.5, 1.25, 0.80)
        elif hi_disp:
            tilt = np.where(low_turn_mask < 0.5, 1.20, 0.85)
        return _project_simplex_bounded(base * tilt, w_max=0.35, w_min=0.02)

    sched_reg = _causal_wf_schedule(
        R_df,
        dates,
        _regime_solve,
        roll_window=126,
        ema_alpha=0.25,
        hyst_tau=0.015,
    )
    out.append(
        run_consolidated_strategy(
            "E6_RegimeAware_DispVol",
            "Method_E_Dynamic",
            sched_reg,
            cand_weights,
            bundle,
        )
    )

    def _corr_aware_solve(X: np.ndarray, wp: np.ndarray, t: int) -> np.ndarray:
        cov = ledoit_wolf_cov(X)
        base = solve_erc(cov, w_max=0.35)
        sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        C = np.clip(cov / np.outer(sd, sd), -0.99, 0.99)
        avg_c = (C.sum(axis=1) - 1.0) / max(N - 1, 1)
        pen = np.exp(-0.75 * np.maximum(avg_c, 0.0))
        return _project_simplex_bounded(base * pen, w_max=0.35, w_min=0.02)

    sched_corr = _causal_wf_schedule(
        R_df,
        dates,
        _corr_aware_solve,
        roll_window=126,
        ema_alpha=0.30,
        hyst_tau=0.015,
    )
    out.append(
        run_consolidated_strategy(
            "E7_CorrAware_ERC", "Method_E_Dynamic", sched_corr, cand_weights, bundle
        )
    )
    return out


def build_method_f_hierarchical(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    ret_net_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Method F: Two-level hierarchical allocation (Prediction <-> Construction) vs Direct."""
    dates = bundle.dates
    len(dates)
    out: list[dict[str, Any]] = []

    def _build_two_level(
        groups: dict[str, list[str]], l1_erc: bool, name: str
    ) -> dict[str, Any]:
        # Step 1: causal Level-2 ERC weights inside each group
        l2_scheds: dict[str, pd.DataFrame] = {}
        grp_rets = {}
        for gname, cols in groups.items():
            sch = _causal_wf_schedule(
                ret_net_df[cols],
                dates,
                lambda X, wp, t, m=0.45: solve_erc(ledoit_wolf_cov(X), w_max=m),
                ema_alpha=0.50,
            )
            l2_scheds[gname] = sch
            # Group daily net return stream
            grp_rets[gname] = (
                ret_net_df[cols].to_numpy(dtype=np.float64)
                * sch.to_numpy(dtype=np.float64)
            ).sum(axis=1)

        grp_ret_df = pd.DataFrame(grp_rets, index=dates)
        if l1_erc:
            l1_sched = _causal_wf_schedule(
                grp_ret_df,
                dates,
                lambda X, wp, t: solve_erc(ledoit_wolf_cov(X), w_max=0.60),
                ema_alpha=0.50,
            )
        else:
            l1_sched = pd.DataFrame(
                1.0 / len(groups), index=dates, columns=list(groups.keys())
            )

        # Combine Level 1 x Level 2 into exact 24-candidate weights
        full_sched = pd.DataFrame(0.0, index=dates, columns=list(CANDIDATES_24))
        for gname, cols in groups.items():
            g_w = l1_sched[gname].to_numpy(dtype=np.float64)[:, None]
            full_sched[cols] += g_w * l2_scheds[gname][cols].to_numpy(dtype=np.float64)
        return run_consolidated_strategy(
            name, "Method_F_Hierarchical", full_sched, cand_weights, bundle
        )

    pred_groups = {e: [f"{e}__{s}" for s in STRATEGIES_V7] for e in EXPERIMENTS_V7}
    out.append(
        _build_two_level(pred_groups, l1_erc=False, name="F1_Hier_PredEW_StratERC")
    )
    out.append(
        _build_two_level(pred_groups, l1_erc=True, name="F2_Hier_PredERC_StratERC")
    )

    const_groups = {s: [f"{e}__{s}" for e in EXPERIMENTS_V7] for s in STRATEGIES_V7}
    out.append(
        _build_two_level(const_groups, l1_erc=True, name="F3_Hier_ConstERC_PredERC")
    )

    horizon_groups = {
        "Horiz_1d_FastCore": list(CANDIDATES_1D_16),
        "Horiz_5d_SlowDiv": [f"risk_t5__{s}" for s in STRATEGIES_V7],
    }
    out.append(
        _build_two_level(horizon_groups, l1_erc=True, name="F4_Hier_Dual1d5d_ERC")
    )
    return out


# ---------------------------------------------------------------------------
# 4. Section 7: Method G Incremental Portfolio-Level Risk Overlays
# ---------------------------------------------------------------------------
def build_method_g_risk_overlays(
    base_res: dict[str, Any],
    prefix: str,
    bundle: V7DataBundle,
) -> list[dict[str, Any]]:
    """Apply causal incremental portfolio-level risk overlays on top of consolidated weights."""
    W0 = base_res["weights"].copy()
    dates = bundle.dates
    T, _N = W0.shape
    r1_np = bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64)
    adv_np = bundle.adv20_df.to_numpy(dtype=np.float64)
    sig_np = bundle.sigma_prev_df.to_numpy(dtype=np.float64)
    rf_np = bundle.ff6_df["rf"].to_numpy(dtype=np.float64)
    beta_np = bundle.beta60_df.to_numpy(dtype=np.float64)

    # Pre-trade gross return stream of raw consolidated portfolio
    port_r_raw = np.einsum("ij,ij->i", W0, r1_np)

    # 1. Causal Vol Targeting (20d/63d blend, target ann vol = 9.0%, strict lag t-1)
    r_ser = pd.Series(port_r_raw)
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
    scale_vt = np.clip(0.09 / np.maximum(vol_hat, 0.025), 0.35, 1.35)
    scale_vt[:25] = 1.0

    # 2. Causal Drawdown Scaling (continuous brake above 3.5% lagged DD)
    nav_raw = np.cumprod(1.0 + np.clip(port_r_raw, -0.5, 1.0))
    hwm_raw = np.maximum.accumulate(np.maximum(nav_raw, 1.0))
    dd_lag = np.zeros(T, dtype=np.float64)
    dd_lag[1:] = -(nav_raw[:-1] / hwm_raw[:-1] - 1.0)
    scale_dd = np.clip(1.0 - np.maximum(dd_lag - 0.035, 0.0) / 0.085, 0.45, 1.0)

    # 3. Causal Rolling 95% CVaR cap (daily target max expected shortfall <= 1.25%)
    cvar_lag = np.full(T, 0.01, dtype=np.float64)
    for t in range(65, T):
        win = port_r_raw[t - 63 : t]
        q05 = np.percentile(win, 5.0)
        cvar_lag[t] = max(-float(win[win <= q05].mean()), 1e-4)
    scale_cvar = np.clip(0.0125 / np.maximum(cvar_lag, 0.005), 0.50, 1.0)

    def _eval_overlay(tag: str, W_mod: np.ndarray) -> dict[str, Any]:
        bt = backtest_weights_daily(
            W_mod,
            r1_np,
            dates,
            cost_bps=BASELINE_COST_BPS,
            adv20_np=adv_np,
            sigma_np=sig_np,
            rf_np=rf_np,
        )
        return {
            "name": f"{prefix}_{tag}",
            "group": "Method_G_RiskOverlay",
            "schedule": base_res["schedule"],
            "weights": W_mod,
            "bt": bt,
        }

    W_vt = apply_portfolio_constraints(W0 * scale_vt[:, None], max_single_w=0.035)
    W_vt_gc = apply_portfolio_constraints(
        W0 * scale_vt[:, None], max_single_w=0.035, max_gross=1.80
    )
    W_vt_dd = apply_portfolio_constraints(
        W0 * (scale_vt * scale_dd)[:, None], max_single_w=0.035, max_gross=1.80
    )
    W_vt_cvar = apply_portfolio_constraints(
        W0 * (scale_vt * scale_dd * scale_cvar)[:, None],
        max_single_w=0.035,
        max_gross=1.80,
    )
    W_full = apply_portfolio_constraints(
        W0 * (scale_vt * scale_dd * scale_cvar)[:, None],
        beta60_np=beta_np,
        sectors=bundle.sectors,
        adv20_np=adv_np,
        r1d_np=r1_np,
        max_single_w=0.025,
        max_gross=1.80,
        max_abs_beta=0.08,
        max_sector_net=0.15,
        max_adv_frac=0.05,
    )
    return [
        _eval_overlay("G0_Base", W0),
        _eval_overlay("G1_VolTgt09", W_vt),
        _eval_overlay("G2_VolTgt09_Gross18", W_vt_gc),
        _eval_overlay("G3_VolTgt09_DDScale", W_vt_dd),
        _eval_overlay("G4_VolTgt09_DD_CVaRCap", W_vt_cvar),
        _eval_overlay("G5_FullConstrainedOverlay", W_full),
    ]


# ---------------------------------------------------------------------------
# 5. Section 10: Mandatory Ablation Study
# ---------------------------------------------------------------------------
def run_section10_ablations(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    ret_net_df: pd.DataFrame,
    flagship_res: dict[str, Any],
    tab_dir: Path,
) -> pd.DataFrame:
    """Section 10 ablations measuring incremental contribution of key pipeline & ensemble blocks."""
    dates = bundle.dates

    def _oos_row(
        label: str,
        res: dict[str, Any],
        base_sh: float,
        base_cagr: float,
        base_mdd: float,
    ) -> dict[str, Any]:
        r = res["bt"]["net"].iloc[WARMUP_DAYS:]
        nav = (1.0 + r).cumprod()
        cagr = float((nav.iloc[-1] ** (ANN_DAYS / len(r)) - 1.0) * 100.0)
        mdd = float(
            -(nav / np.maximum.accumulate(np.maximum(nav, 1.0)) - 1.0).min() * 100.0
        )
        sh = float(r.mean() / (r.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS))
        tvr = float(res["bt"]["traded"].iloc[WARMUP_DAYS:].mean() * 100.0)
        return {
            "Ablation": label,
            "OOS_Net2_Sharpe": round(sh, 4),
            "Delta_Sharpe": round(sh - base_sh, 4),
            "OOS_CAGR(%)": round(cagr, 2),
            "Delta_CAGR(%)": round(cagr - base_cagr, 2),
            "OOS_MaxDD(%)": round(mdd, 2),
            "Delta_MaxDD(%)": round(mdd - base_mdd, 2),
            "OOS_Turnover(%)": round(tvr, 2),
        }

    # Reference baseline: Direct ERC across all 24 candidates
    ref_sched = _causal_wf_schedule(
        ret_net_df[list(CANDIDATES_24)],
        dates,
        lambda X, wp, t: solve_erc(ledoit_wolf_cov(X), w_max=0.22),
    )
    ref_res = run_consolidated_strategy(
        "Abl_Ref_ERC_All24", "Ablation", ref_sched, cand_weights, bundle
    )
    r_ref = ref_res["bt"]["net"].iloc[WARMUP_DAYS:]
    nav_ref = (1.0 + r_ref).cumprod()
    b_sh = float(r_ref.mean() / (r_ref.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS))
    b_cagr = float((nav_ref.iloc[-1] ** (ANN_DAYS / len(r_ref)) - 1.0) * 100.0)
    b_mdd = float(
        -(nav_ref / np.maximum.accumulate(np.maximum(nav_ref, 1.0)) - 1.0).min() * 100.0
    )

    abl_specs = {
        "Baseline_ERC_All24": list(CANDIDATES_24),
        "Drop_BetaNeu": [c for c in CANDIDATES_24 if "BetaNeu" not in c],
        "Drop_InvVol": [c for c in CANDIDATES_24 if "InvVol" not in c],
        "Drop_Buffer": [c for c in CANDIDATES_24 if "Buf" not in c],
        "Drop_HOLD5": [c for c in CANDIDATES_24 if "HOLD5" not in c],
        "Drop_risk_t1": [c for c in CANDIDATES_24 if not c.startswith("risk_t1__")],
        "Drop_raw_t1": [c for c in CANDIDATES_24 if not c.startswith("raw_t1__")],
        "Drop_risk_t5": [c for c in CANDIDATES_24 if not c.startswith("risk_t5__")],
    }
    rows = []
    for label, cols in abl_specs.items():
        if label == "Baseline_ERC_All24":
            rows.append(_oos_row(label, ref_res, b_sh, b_cagr, b_mdd))
            continue
        sch = _causal_wf_schedule(
            ret_net_df[cols],
            dates,
            lambda X, wp, t: solve_erc(ledoit_wolf_cov(X), w_max=0.25),
        )
        res = run_consolidated_strategy(
            f"Abl_{label}", "Ablation", sch, cand_weights, bundle
        )
        rows.append(_oos_row(label, res, b_sh, b_cagr, b_mdd))

    # Architecture ablations against ERC_All24
    ew_res = run_consolidated_strategy(
        "Abl_EW_All24",
        "Ablation",
        _static_schedule(dates, {c: 1.0 / len(CANDIDATES_24) for c in CANDIDATES_24}),
        cand_weights,
        bundle,
    )
    rows.append(_oos_row("Replace_CovOpt_With_EW", ew_res, b_sh, b_cagr, b_mdd))

    rows.append(_oos_row("Final_Flagship_Full", flagship_res, b_sh, b_cagr, b_mdd))
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "tab10_ablation_study.csv", index=False)
    return df
