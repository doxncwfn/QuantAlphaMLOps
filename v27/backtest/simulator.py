"""Daily mark-to-market backtest simulator with linear TC, short borrow, financing, and sqrt impact."""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_weights_daily(
    W: np.ndarray,
    r_1d_np: np.ndarray,
    dates: pd.DatetimeIndex,
    cost_bps: float = 2.0,
    adv20_np: np.ndarray | None = None,
    sigma_np: np.ndarray | None = None,
    rf_np: np.ndarray | None = None,
    book_size: float = 20_000_000.0,
    short_borrow_ann: float = 0.0050,
    impact_eta: float = 0.10,
    ann_days: float = 252.0,
    inception_zero_trade: bool = False,
) -> dict[str, pd.Series]:
    """Execute daily mark-to-market backtest on consolidated weight panel W[t]."""
    W_c = np.nan_to_num(W.astype(np.float64), nan=0.0)
    R = np.nan_to_num(r_1d_np.astype(np.float64), nan=0.0)
    T, _ = W_c.shape

    gross = np.einsum("ti,ti->t", W_c, R)
    drifted = np.zeros_like(W_c)
    drifted[1:] = (W_c[:-1] * (1.0 + R[:-1])) / np.maximum(1.0 + gross[:-1, None], 1e-6)
    delta_w = np.abs(W_c - drifted)
    traded = delta_w.sum(axis=1)
    if inception_zero_trade:
        traded[0] = 0.0

    long_exp = np.maximum(W_c, 0.0).sum(axis=1)
    short_exp = np.maximum(-W_c, 0.0).sum(axis=1)
    gross_exp = long_exp + short_exp
    net_exp = long_exp - short_exp

    tc_linear = traded * (cost_bps / 1e4)
    net_linear = gross - tc_linear

    borrow = short_exp * (short_borrow_ann / ann_days)
    rf_arr = (
        np.zeros(T, dtype=np.float64)
        if rf_np is None
        else np.nan_to_num(rf_np, nan=0.0)
    )
    fin = np.maximum(long_exp - 1.0, 0.0) * (
        np.maximum(rf_arr, 0.0) + 0.0050 / ann_days
    )
    if adv20_np is not None and sigma_np is not None:
        ADV = np.maximum(np.nan_to_num(adv20_np, nan=50e6), 1e6)
        SIG = np.clip(np.nan_to_num(sigma_np, nan=0.02), 0.003, 0.25)
        part = np.minimum((delta_w * book_size) / ADV, 1.0)
        impact = (delta_w * impact_eta * SIG * np.sqrt(part)).sum(axis=1)
    else:
        impact = np.zeros(T, dtype=np.float64)

    net_allin = gross - tc_linear - borrow - fin - impact
    abs_w = np.abs(W_c)
    norm_abs = abs_w / np.maximum(gross_exp[:, None], 1e-12)
    hhi = (norm_abs**2).sum(axis=1)
    eff_n = np.where(hhi > 1e-12, 1.0 / hhi, 0.0)

    return {
        "gross": pd.Series(gross, index=dates),
        "net": pd.Series(net_linear, index=dates),
        "net_allin": pd.Series(net_allin, index=dates),
        "traded": pd.Series(traded, index=dates),
        "tc": pd.Series(tc_linear, index=dates),
        "borrow": pd.Series(borrow, index=dates),
        "fin": pd.Series(fin, index=dates),
        "impact": pd.Series(impact, index=dates),
        "long_exp": pd.Series(long_exp, index=dates),
        "short_exp": pd.Series(short_exp, index=dates),
        "gross_exp": pd.Series(gross_exp, index=dates),
        "net_exp": pd.Series(net_exp, index=dates),
        "eff_n": pd.Series(eff_n, index=dates),
    }
