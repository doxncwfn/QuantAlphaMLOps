"""Static and strictly causal walk-forward strategy allocation schedule builders."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def static_schedule(
    dates: pd.DatetimeIndex, weights_map: dict[str, float]
) -> pd.DataFrame:
    """Create constant sleeve allocation table over the evaluation calendar."""
    df = pd.DataFrame(0.0, index=dates, columns=list(weights_map.keys()))
    for k, v in weights_map.items():
        df[k] = float(v)
    return df


def causal_wf_schedule(
    ret_mat: pd.DataFrame,
    dates: pd.DatetimeIndex,
    solve_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    rebal_freq: int = 21,
    warmup: int = 252,
    purge: int = 6,
    roll_window: int | None = None,
    ema_alpha: float = 1.0,
    hyst_tau: float = 0.0,
) -> pd.DataFrame:
    """Strictly causal expanding or rolling allocation schedule with optional EMA and L1 hysteresis."""
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
                if not (
                    t > warmup
                    and hyst_tau > 0.0
                    and float(np.abs(w_blend - w_curr).sum()) < hyst_tau
                ):
                    w_curr = w_blend
        sched[t] = w_curr
    return pd.DataFrame(sched, index=dates, columns=cols)
