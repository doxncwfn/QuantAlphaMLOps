"""Equity-curve performance metrics, drawdown episode tracking, and legacy BRAIN stats."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from v27.data.bundle import V7DataBundle


def lo_hac_sharpe(r: pd.Series, ann_days: float = 252.0, max_lag: int = 5) -> float:
    """Andrew Lo (2002) autocorrelation-adjusted annualized Sharpe ratio."""
    arr = r.dropna().to_numpy(dtype=np.float64)
    n = len(arr)
    if n < 20:
        return 0.0
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd < 1e-12:
        return 0.0
    sr_1d = mu / sd
    rho_sum = 0.0
    x = arr - mu
    var = float(np.dot(x, x))
    if var > 1e-16:
        for k in range(1, max_lag + 1):
            rk = float(np.dot(x[k:], x[:-k]) / var)
            rho_sum += (1.0 - k / (max_lag + 1.0)) * rk
    denom = math.sqrt(max(1.0 + 2.0 * rho_sum, 0.25))
    return float((sr_1d * math.sqrt(ann_days)) / denom)


def compounded_dd_episodes(r: pd.Series) -> dict[str, float]:
    """Compute compounded max drawdown, longest underwater duration, max recovery, and Ulcer index."""
    arr = r.dropna().to_numpy(dtype=np.float64)
    nav = np.cumprod(1.0 + np.clip(arr, -0.95, 5.0))
    hwm = np.maximum.accumulate(np.maximum(nav, 1.0))
    dd = nav / hwm - 1.0
    mdd = float(-dd.min())
    ulcer = float(np.sqrt(np.mean((dd * 100.0) ** 2)))
    uw = dd < -1e-9
    durs, recs = [], []
    i, n = 0, len(dd)
    while i < n:
        if uw[i]:
            st = i
            while i < n and uw[i]:
                i += 1
            en = i
            tr = st + int(np.argmin(dd[st:en]))
            durs.append(en - st)
            recs.append(min(n, en) - tr)
        else:
            i += 1
    return {
        "Comp_MaxDD(%)": mdd * 100.0,
        "MaxDD_Dur(d)": float(max(durs)) if durs else 0.0,
        "MaxRecov(d)": float(max(recs)) if recs else 0.0,
        "UlcerIdx": ulcer,
    }


def compute_legacy_brain_stats(
    bt: dict[str, pd.Series], cost_bps: float = 2.0, ann_days: float = 252.0
) -> dict[str, float]:
    """Legacy arithmetic BRAIN stats matching grand_summary_v7.csv for reference parity."""
    g = bt["gross"]
    tr = bt["traded"]
    net = g - tr * (cost_bps / 1e4)
    sh = float(g.mean() / (g.std() + 1e-12) * math.sqrt(ann_days))
    ret = float(g.mean() * ann_days)
    turn = float(0.5 * tr.mean())
    fit = sh * math.sqrt(abs(ret) / max(turn, 0.125))
    cum = g.cumsum()
    dd = float((np.maximum(cum.cummax(), 0.0) - cum).max())
    mar = float(g.sum() / (tr.sum() + 1e-12) * 1e4)
    nsh = float(net.mean() / (net.std() + 1e-12) * math.sqrt(ann_days))
    return {
        "Legacy_GrossSh": sh,
        "Legacy_Turn(%)": turn * 100.0,
        "Legacy_Fitness": fit,
        "Legacy_Ret(%)": ret * 100.0,
        "Legacy_ArithDD(%)": dd * 100.0,
        "Legacy_Margin(bps)": mar,
        f"Legacy_NetSh@{cost_bps:g}bps": nsh,
    }


def compute_full_metrics_row(
    name: str,
    group: str,
    bt: dict[str, pd.Series],
    W_mat: np.ndarray | None,
    sched: pd.DataFrame | None,
    bundle: V7DataBundle,
    best_ind_ret: pd.Series,
    warmup: int = 252,
    ann_days: float = 252.0,
    cost_bps: float = 2.0,
) -> dict[str, Any]:
    """Compute complete Section 8 OOS risk, return, tail, turnover, and factor summary row."""
    sl = slice(warmup, None)
    r = bt["net"].iloc[sl]
    r_ai = bt["net_allin"].iloc[sl]
    r_g = bt["gross"].iloc[sl]
    tr = bt["traded"].iloc[sl]
    idx = r.index

    ew_b = bundle.ew_mkt_1d.reindex(idx).fillna(0.0)
    bi_b = best_ind_ret.reindex(idx).fillna(0.0)

    n_yrs = len(r) / ann_days
    nav = (1.0 + r.clip(lower=-0.95)).cumprod()
    cum_ret = float((nav.iloc[-1] - 1.0) * 100.0)
    cagr = float((nav.iloc[-1] ** (1.0 / n_yrs) - 1.0) * 100.0)
    ann_ret = float(r.mean() * ann_days * 100.0)
    ann_vol = float(r.std(ddof=1) * math.sqrt(ann_days) * 100.0)

    m_rets = (1.0 + r).resample("ME").prod() - 1.0
    y_rets = (1.0 + r).resample("YE").prod() - 1.0

    gross_sh = float(r_g.mean() / (r_g.std(ddof=1) + 1e-12) * math.sqrt(ann_days))
    net_sh = float(r.mean() / (r.std(ddof=1) + 1e-12) * math.sqrt(ann_days))
    allin_sh = float(r_ai.mean() / (r_ai.std(ddof=1) + 1e-12) * math.sqrt(ann_days))
    hac_sh = lo_hac_sharpe(r, ann_days, max_lag=5)
    ac1 = float(r.autocorr(lag=1))
    down = r[r < 0]
    down_sd_ann = math.sqrt(float((down**2).mean()) * ann_days) if len(down) else 1e-12
    sortino = float((r.mean() * ann_days) / (down_sd_ann + 1e-12))

    dd_info = compounded_dd_episodes(r)
    calmar = float((cagr / 100.0) / max(dd_info["Comp_MaxDD(%)"] / 100.0, 1e-4))
    act_ew = r - ew_b
    ir_ew = float(act_ew.mean() / (act_ew.std(ddof=1) + 1e-12) * math.sqrt(ann_days))
    act_bi = r - bi_b
    ir_bi = float(act_bi.mean() / (act_bi.std(ddof=1) + 1e-12) * math.sqrt(ann_days))
    omega = float(r[r > 0].sum() / max(float(-r[r < 0].sum()), 1e-12))

    var95 = float(r.quantile(0.05) * 100.0)
    var99 = float(r.quantile(0.01) * 100.0)
    cvar95 = float(r[r <= r.quantile(0.05)].mean() * 100.0)
    cvar99 = float(r[r <= r.quantile(0.01)].mean() * 100.0)
    roll5 = (1.0 + r).rolling(5).apply(np.prod, raw=True) - 1.0
    roll20 = (1.0 + r).rolling(20).apply(np.prod, raw=True) - 1.0

    avg_gross = float(bt["gross_exp"].iloc[sl].mean())
    avg_net = float(bt["net_exp"].iloc[sl].mean())
    daily_tvr = float(tr.mean() * 100.0)
    ann_tvr_1way = daily_tvr * 0.5 * ann_days
    tc_ann = float(bt["tc"].iloc[sl].mean() * ann_days * 100.0)
    borrow_ann = float(bt["borrow"].iloc[sl].mean() * ann_days * 100.0)

    eff_n = np.nan
    port_beta = np.nan
    if W_mat is not None:
        W_oos = W_mat[sl]
        w_sq = (W_oos**2).sum(axis=1)
        w_abs = np.abs(W_oos).sum(axis=1)
        valid_g = w_sq > 1e-12
        if valid_g.any():
            eff_n = float(np.mean((w_abs[valid_g] ** 2) / w_sq[valid_g]))
        b_oos = np.nan_to_num(bundle.beta60_df.to_numpy(dtype=np.float64)[sl], nan=1.0)
        port_beta = float(np.mean(np.einsum("ij,ij->i", W_oos, b_oos)))

    alloc_to_ann = 0.0
    if sched is not None and len(sched) > warmup + 1:
        s_oos = sched.iloc[sl].to_numpy(dtype=np.float64)
        alloc_to_ann = float(
            np.abs(np.diff(s_oos, axis=0)).sum(axis=1).mean() * 0.5 * ann_days * 100.0
        )

    legacy = compute_legacy_brain_stats(
        {"gross": r_g, "traded": tr}, cost_bps=cost_bps, ann_days=ann_days
    )
    return {
        "Strategy": name,
        "Group": group,
        "CumRet(%)": round(cum_ret, 2),
        "CAGR(%)": round(cagr, 2),
        "AnnRet(%)": round(ann_ret, 2),
        "AnnVol(%)": round(ann_vol, 2),
        "Gross_Sharpe": round(gross_sh, 3),
        "Net2_Sharpe": round(net_sh, 3),
        "AllIn_Sharpe": round(allin_sh, 3),
        "HAC_Sharpe": round(hac_sh, 3),
        "AC1": round(ac1, 3),
        "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3),
        "Omega": round(omega, 3),
        "IR_vs_EW": round(ir_ew, 3),
        "IR_vs_BestInd": round(ir_bi, 3),
        "Comp_MaxDD(%)": round(dd_info["Comp_MaxDD(%)"], 2),
        "MaxDD_Dur(d)": int(dd_info["MaxDD_Dur(d)"]),
        "MaxRecov(d)": int(dd_info["MaxRecov(d)"]),
        "UlcerIdx": round(dd_info["UlcerIdx"], 2),
        "PosMonths(%)": round(float((m_rets > 0).mean() * 100.0), 1),
        "PosYears(%)": round(float((y_rets > 0).mean() * 100.0), 1),
        "VaR95(%)": round(var95, 3),
        "VaR99(%)": round(var99, 3),
        "CVaR95(%)": round(cvar95, 3),
        "CVaR99(%)": round(cvar99, 3),
        "Skew": round(float(r.skew()), 3),
        "ExKurt": round(float(r.kurt()), 2),
        "Worst5d(%)": round(float(roll5.min() * 100.0), 2),
        "Worst20d(%)": round(float(roll20.min() * 100.0), 2),
        "AvgGross": round(avg_gross, 3),
        "AvgNet": round(avg_net, 3),
        "DailyTurn(%)": round(daily_tvr, 2),
        "AnnTurn_1Way(%)": round(ann_tvr_1way, 1),
        "TC_Ann(%)": round(tc_ann, 2),
        "Borrow_Ann(%)": round(borrow_ann, 2),
        "EffN_Stock": round(eff_n, 1) if np.isfinite(eff_n) else np.nan,
        "Port_Beta60": round(port_beta, 3) if np.isfinite(port_beta) else np.nan,
        "Alloc_TO_Ann(%)": round(alloc_to_ann, 1),
        "Legacy_BRAIN_ArithMaxDD(%)": legacy["Legacy_ArithDD(%)"],
    }
