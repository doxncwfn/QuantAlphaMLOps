"""Eight deterministic cross-sectional portfolio construction rules + causal orthogonal selector."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from v27.portfolio.neutralization import neutralize_beta, renorm_legs


def strat_rank_raw(sc_df: pd.DataFrame) -> pd.DataFrame:
    rk = sc_df.rank(axis=1)
    return renorm_legs(rk.sub(rk.mean(axis=1), axis=0).where(sc_df.notna()).fillna(0.0))


def strat_top10_raw(sc_df: pd.DataFrame, q: float = 0.10) -> pd.DataFrame:
    p = sc_df.rank(axis=1, pct=True)
    raw = (
        ((p > 1.0 - q).astype(float) - (p <= q).astype(float))
        .where(sc_df.notna())
        .fillna(0.0)
    )
    return renorm_legs(raw)


def strat_top10_invvol(
    sc_df: pd.DataFrame, inv_vol_df: pd.DataFrame, q: float = 0.10
) -> pd.DataFrame:
    p = sc_df.rank(axis=1, pct=True)
    raw = (
        ((p > 1.0 - q).astype(float) - (p <= q).astype(float))
        .where(sc_df.notna())
        .fillna(0.0)
    )
    return renorm_legs(raw * inv_vol_df.where(sc_df.notna()).fillna(0.0))


def strat_top10_buf(
    sc_df: pd.DataFrame,
    inv_vol_df: pd.DataFrame | None = None,
    q_in: float = 0.10,
    q_out: float = 0.20,
) -> pd.DataFrame:
    """10/20 hysteresis buffer portfolio construction."""
    pct = sc_df.rank(axis=1, pct=True).to_numpy(dtype=np.float64)
    act = sc_df.notna().to_numpy(dtype=bool)
    iv_arr = (
        np.ones_like(pct)
        if inv_vol_df is None
        else np.nan_to_num(inv_vol_df.to_numpy(dtype=np.float64), nan=0.0)
    )
    T, N = pct.shape
    W = np.zeros((T, N), dtype=np.float64)
    l_st = np.zeros(N, dtype=bool)
    s_st = np.zeros(N, dtype=bool)
    for t in range(T):
        p, a, iv = pct[t], act[t], iv_arr[t]
        p_ok = np.where(a & np.isfinite(p), p, 0.5)
        l_st = a & ((p_ok > 1.0 - q_in) | (l_st & (p_ok > 1.0 - q_out)))
        s_st = a & ((p_ok <= q_in) | (s_st & (p_ok <= q_out)))
        wl = np.where(l_st, iv, 0.0)
        ws = np.where(s_st, iv, 0.0)
        if wl.sum() > 1e-12:
            wl /= wl.sum()
        if ws.sum() > 1e-12:
            ws /= ws.sum()
        W[t] = wl - ws
    return pd.DataFrame(W, index=sc_df.index, columns=sc_df.columns)


def apply_overlapping_hold(
    w_target: pd.DataFrame, r_1d_df: pd.DataFrame, hold_days: int = 5
) -> pd.DataFrame:
    """Overlapping H-day tranche portfolio marked daily using realized 1-day returns."""
    W = np.nan_to_num(w_target.to_numpy(dtype=np.float64), nan=0.0)
    R = np.nan_to_num(r_1d_df.to_numpy(dtype=np.float64), nan=0.0)
    T, N = W.shape
    tranches = np.zeros((hold_days, N), dtype=np.float64)
    W_out = np.zeros((T, N), dtype=np.float64)
    for t in range(T):
        if t > 0:
            rt_p = R[t - 1]
            g = tranches @ rt_p
            tranches = (
                tranches * (1.0 + rt_p[None, :]) / np.maximum(1.0 + g[:, None], 0.25)
            )
        tranches[t % hold_days] = W[t]
        W_out[t] = tranches[: min(t + 1, hold_days)].mean(axis=0)
    return renorm_legs(
        pd.DataFrame(W_out, index=w_target.index, columns=w_target.columns)
    )


def build_8_strategies_for_score(
    sc_df: pd.DataFrame,
    r_1d_df: pd.DataFrame,
    sigma_prev_df: pd.DataFrame,
    beta60_df: pd.DataFrame,
    hold_days: int = 5,
) -> dict[str, pd.DataFrame]:
    """Construct all 8 Stock Ranking v7 portfolio strategies from a prediction score panel."""
    inv_vol = (1.0 / sigma_prev_df).where(sc_df.notna()).fillna(0.0)
    w_t10 = strat_top10_raw(sc_df)
    w_t10_iv = strat_top10_invvol(sc_df, inv_vol)
    sc_bn = neutralize_beta(sc_df, beta60_df)
    return {
        "rank_RAW": strat_rank_raw(sc_df),
        "top10_RAW": w_t10,
        "top10_InvVol": w_t10_iv,
        "top10_Buf10_20": strat_top10_buf(sc_df, None),
        "top10_InvVol_Buf": strat_top10_buf(sc_df, inv_vol),
        "top10_BetaNeu_InvVol_Buf": strat_top10_buf(sc_bn, inv_vol),
        "top10_HOLD5": apply_overlapping_hold(w_t10, r_1d_df, hold_days=hold_days),
        "top10_InvVol_HOLD5": apply_overlapping_hold(
            w_t10_iv, r_1d_df, hold_days=hold_days
        ),
    }


def select_orthogonal_subset(
    ret_net_df: pd.DataFrame,
    warmup: int = 252,
    corr_thresh: float = 0.88,
    ann_days: int = 252,
) -> list[str]:
    """Select an orthogonal candidate core strictly from causal warmup-period Sharpe & correlation."""
    from v27.config import EXPERIMENTS_V7

    sub = ret_net_df.iloc[:warmup]
    sharpes = (sub.mean() / (sub.std(ddof=1) + 1e-12)) * math.sqrt(ann_days)
    order = sharpes.sort_values(ascending=False).index.tolist()
    corr = sub.corr()
    chosen: list[str] = []
    for cid in order:
        if (
            not chosen
            or max(abs(float(corr.loc[cid, c])) for c in chosen) < corr_thresh
        ):
            chosen.append(cid)
    for exp in EXPERIMENTS_V7:
        if not any(c.startswith(f"{exp}__") for c in chosen):
            exp_cands = [c for c in order if c.startswith(f"{exp}__")]
            if exp_cands:
                chosen.append(exp_cands[0])
    return sorted(chosen)
