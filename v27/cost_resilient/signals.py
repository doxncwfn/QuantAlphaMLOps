"""Pillar 1: Causal signal smoothing (EMA, adaptive Kalman), rank/z-score normalization, and horizon blends."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from v27.config import PortfolioBacktestConfig
from v27.data.bundle import V7DataBundle
from v27.portfolio.neutralization import neutralize_beta

WARMUP_DAYS = PortfolioBacktestConfig().warmup_days


def xs_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Cross-sectional z-score standardization per date (strictly contemporaneous, no lookahead)."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).clip(lower=1e-8)
    return df.sub(mu, axis=0).div(sd, axis=0).clip(-clip, clip)


def xs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional centered percentile rank in [-0.5, +0.5]."""
    return df.rank(axis=1, pct=True) - 0.5


def xs_rankgauss(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Cross-sectional rank-Gauss transformation mapping ranks to clipped N(0, 1)."""
    n_act = df.notna().sum(axis=1).clip(lower=2)
    u = (df.rank(axis=1) - 0.5).div(n_act, axis=0).clip(1e-4, 1.0 - 1e-4)
    arr = norm.ppf(u.to_numpy(dtype=np.float64))
    arr = np.where(df.notna().to_numpy(), np.clip(arr, -clip, clip), np.nan)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def causal_ema_panel(df: pd.DataFrame, span: float) -> pd.DataFrame:
    """Strictly causal per-stock EMA over active trading dates (resets cleanly when stock exits)."""
    if span <= 1.0:
        return df.copy()
    alpha = 2.0 / (span + 1.0)
    X = df.to_numpy(dtype=np.float64)
    T, N = X.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    state = np.zeros(N, dtype=np.float64)
    has_state = np.zeros(N, dtype=bool)
    for t in range(T):
        x = X[t]
        ok = np.isfinite(x)
        upd = ok & has_state
        new = ok & (~has_state)
        if upd.any():
            state[upd] = alpha * x[upd] + (1.0 - alpha) * state[upd]
        if new.any():
            state[new] = x[new]
        has_state = ok.copy()
        state[~ok] = 0.0
        out[t, ok] = state[ok]
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def causal_kalman_panel(
    df: pd.DataFrame,
    q_ratio: float = 0.30,
    phi: float = 0.88,
    adaptive_phi: bool = False,
) -> pd.DataFrame:
    """Strictly causal scalar Kalman filter separating persistent latent alpha from daily noise.

    State model for stock i:
        x_{i,t} = phi_t * x_{i,t-1} + eta_{i,t},   Var(eta) = q_ratio * R
        z_{i,t} = x_{i,t} + eps_{i,t},             Var(eps) = R = 1.0

    When `adaptive_phi=True`, estimates phi_t causally from rolling cross-sectional lag-1
    autocorrelation of z up to day t-1 (strictly causal).
    """
    X = df.to_numpy(dtype=np.float64)
    T, N = X.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    x_hat = np.zeros(N, dtype=np.float64)
    P = np.ones(N, dtype=np.float64)
    active = np.zeros(N, dtype=bool)

    phi_t = float(phi)
    ac_hist: list[float] = []

    for t in range(T):
        z = X[t]
        ok = np.isfinite(z)

        if adaptive_phi and t >= 2:
            # Update phi strictly from (t-2, t-1) cross-sectional autocorrelation
            prev1 = X[t - 1]
            prev2 = X[t - 2]
            both = np.isfinite(prev1) & np.isfinite(prev2)
            if both.sum() >= 30:
                a, b = prev1[both], prev2[both]
                sa, sb = float(np.std(a)), float(np.std(b))
                if sa > 1e-8 and sb > 1e-8:
                    ac = float(np.dot(a - a.mean(), b - b.mean()) / (len(a) * sa * sb))
                    ac_hist.append(float(np.clip(ac, 0.25, 0.96)))
                    if len(ac_hist) > 63:
                        ac_hist.pop(0)
                    phi_t = float(np.clip(np.mean(ac_hist), 0.50, 0.95))

        upd = ok & active
        new = ok & (~active)
        if upd.any():
            x_pred = phi_t * x_hat[upd]
            P_pred = (phi_t * phi_t) * P[upd] + q_ratio
            K = P_pred / (P_pred + 1.0)
            x_hat[upd] = x_pred + K * (z[upd] - x_pred)
            P[upd] = (1.0 - K) * P_pred
        if new.any():
            x_hat[new] = z[new]
            P[new] = 0.50
        active = ok.copy()
        x_hat[~ok] = 0.0
        P[~ok] = 1.0
        out[t, ok] = x_hat[ok]
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def build_signal_library(bundle: V7DataBundle) -> dict[str, pd.DataFrame]:
    """Construct standardized, smoothed, Kalman-filtered, and horizon-aligned signal panels."""
    z_raw = xs_zscore(bundle.scores["raw_t1"])
    z_r1 = xs_zscore(bundle.scores["risk_t1"])
    z_r5 = xs_zscore(bundle.scores["risk_t5"])

    rk_raw = xs_rank(bundle.scores["raw_t1"])
    rk_r1 = xs_rank(bundle.scores["risk_t1"])
    rk_r5 = xs_rank(bundle.scores["risk_t5"])

    z_r1_ema3 = causal_ema_panel(z_r1, span=3.0)
    z_r1_ema5 = causal_ema_panel(z_r1, span=5.0)
    z_raw_ema3 = causal_ema_panel(z_raw, span=3.0)
    z_raw_ema5 = causal_ema_panel(z_raw, span=5.0)
    z_r5_ema3 = causal_ema_panel(z_r5, span=3.0)

    z_r1_kal = causal_kalman_panel(z_r1, q_ratio=0.30, phi=0.88, adaptive_phi=True)
    z_raw_kal = causal_kalman_panel(z_raw, q_ratio=0.30, phi=0.88, adaptive_phi=True)
    z_r5_kal = causal_kalman_panel(z_r5, q_ratio=0.25, phi=0.92, adaptive_phi=True)

    return {
        "z_raw": z_raw,
        "z_r1": z_r1,
        "z_r5": z_r5,
        "rk_raw": rk_raw,
        "rk_r1": rk_r1,
        "rk_r5": rk_r5,
        "z_r1_ema3": z_r1_ema3,
        "z_r1_ema5": z_r1_ema5,
        "z_raw_ema3": z_raw_ema3,
        "z_raw_ema5": z_raw_ema5,
        "z_r5_ema3": z_r5_ema3,
        "z_r1_kal": z_r1_kal,
        "z_raw_kal": z_raw_kal,
        "z_r5_kal": z_r5_kal,
        # Multi-horizon blends
        "z_ew3": (z_raw + z_r1 + z_r5) / 3.0,
        "rk_ew3": (rk_raw + rk_r1 + rk_r5) / 3.0,
        "z_r5r1_ema3": 0.60 * z_r5 + 0.40 * z_r1_ema3,
        "z_all3_ema3": 0.50 * z_r5 + 0.35 * z_r1_ema3 + 0.15 * z_raw_ema3,
        "z_all3_ema5": 0.50 * z_r5 + 0.30 * z_r1_ema5 + 0.20 * z_raw_ema5,
        "z_all3_kal": 0.50 * z_r5 + 0.35 * z_r1_kal + 0.15 * z_raw_kal,
        "z_slow70_ema3": 0.70 * z_r5 + 0.20 * z_r1_ema3 + 0.10 * z_raw_ema3,
        "rk_all3_ema5": 0.50 * rk_r5
        + 0.30 * causal_ema_panel(rk_r1, span=5.0)
        + 0.20 * causal_ema_panel(rk_raw, span=5.0),
    }


def compute_signal_persistence_and_decay_table(
    bundle: V7DataBundle,
    signals: dict[str, pd.DataFrame],
    warmup: int = WARMUP_DAYS,
) -> pd.DataFrame:
    """Measure OOS cross-sectional rank autocorrelation and multi-day IC decay (t+0..t+9)."""
    r1 = bundle.ret1d_df
    eval_keys = [
        ("raw_t1 (raw)", bundle.scores["raw_t1"]),
        (
            "raw_t1 (BetaNeu)",
            neutralize_beta(bundle.scores["raw_t1"], bundle.beta60_df),
        ),
        ("risk_t1 (raw)", bundle.scores["risk_t1"]),
        (
            "risk_t1 (BetaNeu)",
            neutralize_beta(bundle.scores["risk_t1"], bundle.beta60_df),
        ),
        ("risk_t5 (raw)", bundle.scores["risk_t5"]),
        (
            "risk_t5 (BetaNeu)",
            neutralize_beta(bundle.scores["risk_t5"], bundle.beta60_df),
        ),
        ("z_r1_ema3", signals["z_r1_ema3"]),
        ("z_r1_kal", signals["z_r1_kal"]),
        ("z_ew3", signals["z_ew3"]),
        ("z_all3_ema3", signals["z_all3_ema3"]),
        ("z_all3_kal", signals["z_all3_kal"]),
        ("z_r5r1_ema3", signals["z_r5r1_ema3"]),
    ]
    rows = []
    for name, df in eval_keys:
        rk = df.rank(axis=1, pct=True)
        ac1 = float(rk.iloc[warmup:].corrwith(rk.shift(1).iloc[warmup:], axis=1).mean())
        ac2 = float(rk.iloc[warmup:].corrwith(rk.shift(2).iloc[warmup:], axis=1).mean())
        ac5 = float(rk.iloc[warmup:].corrwith(rk.shift(5).iloc[warmup:], axis=1).mean())
        row: dict[str, object] = {
            "Signal": name,
            "RankAC_1d": round(ac1, 4),
            "RankAC_2d": round(ac2, 4),
            "RankAC_5d": round(ac5, 4),
        }
        ic_vals = []
        for k in (0, 1, 2, 3, 4, 5, 9):
            ic_k = float(
                rk.shift(k)
                .iloc[warmup:]
                .corrwith(r1.iloc[warmup:], axis=1, method="spearman")
                .mean()
            )
            row[f"IC_t+{k}"] = round(ic_k, 4)
            if k <= 4:
                ic_vals.append(ic_k)
        row["Mean_IC_t0_t4"] = round(float(np.mean(ic_vals)), 4)
        rows.append(row)
    return pd.DataFrame(rows)
