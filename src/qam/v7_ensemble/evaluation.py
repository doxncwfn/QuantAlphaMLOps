"""Sections 8, 9 & 11: Full Equity-Curve Evaluation, Inference, Robustness, Charts & Weight Export."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.qam.v7_ensemble.combiners import WARMUP_DAYS
from src.qam.v7_ensemble.diagnostics import _compounded_dd_episodes
from src.qam.v7_ensemble.engine import (
    ANN_DAYS,
    BASELINE_COST_BPS,
    COST_GRID_BPS,
    V7DataBundle,
    backtest_weights_daily,
    compute_legacy_brain_stats,
)


# ---------------------------------------------------------------------------
# 1. HAC Inference, Ledoit-Wolf Sharpe Test & Circular Block Bootstrap
# ---------------------------------------------------------------------------
def lo_hac_sharpe(r: pd.Series, ann_days: float = ANN_DAYS, max_lag: int = 5) -> float:
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


def newey_west_ols(y: np.ndarray, X: np.ndarray, max_lag: int = 5) -> dict[str, Any]:
    n, _ = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    Xe = X * resid[:, None]
    S = (Xe.T @ Xe) / n
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        G = (Xe[lag:].T @ Xe[:-lag]) / n
        S += w * (G + G.T)
    cov_b = n * (XtX_inv @ S @ XtX_inv)
    se = np.sqrt(np.maximum(np.diag(cov_b), 1e-18))
    tstat = beta / se
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / max(ss_tot, 1e-18)
    return {"beta": beta, "se": se, "tstat": tstat, "r2": r2}


def ledoit_wolf_sharpe_test(
    r_a: np.ndarray, r_b: np.ndarray, max_lag: int = 5
) -> tuple[float, float, float]:
    a = np.asarray(r_a, dtype=np.float64)
    b = np.asarray(r_b, dtype=np.float64)
    n = len(a)
    mu_a, mu_b = float(a.mean()), float(b.mean())
    s2_a, s2_b = float(np.mean(a**2)), float(np.mean(b**2))
    var_a = max(s2_a - mu_a**2, 1e-16)
    var_b = max(s2_b - mu_b**2, 1e-16)
    sh_a, sh_b = mu_a / math.sqrt(var_a), mu_b / math.sqrt(var_b)
    delta_1d = sh_a - sh_b
    grad = np.array(
        [
            s2_a / (var_a**1.5),
            -s2_b / (var_b**1.5),
            -0.5 * mu_a / (var_a**1.5),
            0.5 * mu_b / (var_b**1.5),
        ],
        dtype=np.float64,
    )
    Y = np.column_stack([a - mu_a, b - mu_b, a**2 - s2_a, b**2 - s2_b])
    Psi = (Y.T @ Y) / n
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        G = (Y[lag:].T @ Y[:-lag]) / n
        Psi += w * (G + G.T)
    se_1d = math.sqrt(max(float(grad @ Psi @ grad) / n, 1e-18))
    z = delta_1d / se_1d
    pval = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return float(delta_1d * math.sqrt(ANN_DAYS)), float(z), pval


def circular_block_bootstrap(
    r: np.ndarray, n_boot: int = 1000, block_len: int = 21, seed: int = 42
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(r, dtype=np.float64)
    n = len(x)
    n_blocks = math.ceil(n / block_len)
    sh_b = np.empty(n_boot, dtype=np.float64)
    cagr_b = np.empty(n_boot, dtype=np.float64)
    mdd_b = np.empty(n_boot, dtype=np.float64)
    n_yrs = n / ANN_DAYS
    offsets = np.arange(block_len)[None, :]
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)[:, None]
        idx = (starts + offsets).ravel()[:n] % n
        xb = x[idx]
        sh_b[b] = xb.mean() / (xb.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS)
        nav = np.cumprod(1.0 + np.clip(xb, -0.95, 5.0))
        cagr_b[b] = (nav[-1] ** (1.0 / n_yrs) - 1.0) * 100.0
        hwm = np.maximum.accumulate(np.maximum(nav, 1.0))
        mdd_b[b] = -(nav / hwm - 1.0).min() * 100.0
    return {
        "Sh_CI_Lo": float(np.percentile(sh_b, 2.5)),
        "Sh_CI_Hi": float(np.percentile(sh_b, 97.5)),
        "P_Sh_gt_0(%)": float((sh_b > 0).mean() * 100.0),
        "CAGR_CI_Lo": float(np.percentile(cagr_b, 2.5)),
        "CAGR_CI_Hi": float(np.percentile(cagr_b, 97.5)),
        "MDD_CI_Lo": float(np.percentile(mdd_b, 2.5)),
        "MDD_CI_Hi": float(np.percentile(mdd_b, 97.5)),
    }


# ---------------------------------------------------------------------------
# 2. Full Section-8 Equity-Curve Metrics
# ---------------------------------------------------------------------------
def compute_full_metrics_row(
    name: str,
    group: str,
    bt: dict[str, pd.Series],
    W_mat: np.ndarray | None,
    sched: pd.DataFrame | None,
    bundle: V7DataBundle,
    best_ind_ret: pd.Series,
    warmup: int = WARMUP_DAYS,
) -> dict[str, Any]:
    sl = slice(warmup, None)
    r = bt["net"].iloc[sl]
    r_ai = bt["net_allin"].iloc[sl]
    r_g = bt["gross"].iloc[sl]
    tr = bt["traded"].iloc[sl]
    idx = r.index

    ew_b = bundle.ew_mkt_1d.reindex(idx).fillna(0.0)
    bi_b = best_ind_ret.reindex(idx).fillna(0.0)

    n_days = len(r)
    n_yrs = n_days / ANN_DAYS
    nav = (1.0 + r.clip(lower=-0.95)).cumprod()
    cum_ret = float((nav.iloc[-1] - 1.0) * 100.0)
    cagr = float((nav.iloc[-1] ** (1.0 / n_yrs) - 1.0) * 100.0)
    ann_ret = float(r.mean() * ANN_DAYS * 100.0)
    ann_vol = float(r.std(ddof=1) * math.sqrt(ANN_DAYS) * 100.0)

    m_rets = (1.0 + r).resample("ME").prod() - 1.0
    y_rets = (1.0 + r).resample("YE").prod() - 1.0

    gross_sh = float(r_g.mean() / (r_g.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS))
    net_sh = float(r.mean() / (r.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS))
    allin_sh = float(r_ai.mean() / (r_ai.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS))
    hac_sh = lo_hac_sharpe(r, ANN_DAYS, max_lag=5)
    ac1 = float(r.autocorr(lag=1))
    down = r[r < 0]
    down_sd_ann = math.sqrt(float((down**2).mean()) * ANN_DAYS) if len(down) else 1e-12
    sortino = float((r.mean() * ANN_DAYS) / (down_sd_ann + 1e-12))

    dd_info = _compounded_dd_episodes(r)
    calmar = float((cagr / 100.0) / max(dd_info["Comp_MaxDD(%)"] / 100.0, 1e-4))
    act_ew = r - ew_b
    ir_ew = float(act_ew.mean() / (act_ew.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS))
    act_bi = r - bi_b
    ir_bi = float(act_bi.mean() / (act_bi.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS))
    omega = float(r[r > 0].sum() / max(float(-r[r < 0].sum()), 1e-12))

    var95 = float(r.quantile(0.05) * 100.0)
    var99 = float(r.quantile(0.01) * 100.0)
    cvar95 = float(r[r <= r.quantile(0.05)].mean() * 100.0)
    cvar99 = float(r[r <= r.quantile(0.01)].mean() * 100.0)
    roll5 = (1.0 + r).rolling(5).apply(np.prod, raw=True) - 1.0
    roll20 = (1.0 + r).rolling(20).apply(np.prod, raw=True) - 1.0

    # Portfolio weights diagnostics
    avg_gross = float(bt["gross_exp"].iloc[sl].mean())
    avg_net = float(bt["net_exp"].iloc[sl].mean())
    daily_tvr = float(tr.mean() * 100.0)
    ann_tvr_1way = daily_tvr * 0.5 * ANN_DAYS
    tc_ann = float(bt["tc"].iloc[sl].mean() * ANN_DAYS * 100.0)
    borrow_ann = float(bt["borrow"].iloc[sl].mean() * ANN_DAYS * 100.0)

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
            np.abs(np.diff(s_oos, axis=0)).sum(axis=1).mean() * 0.5 * ANN_DAYS * 100.0
        )

    legacy = compute_legacy_brain_stats(
        {"gross": r_g, "traded": tr}, cost_bps=BASELINE_COST_BPS
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


# ---------------------------------------------------------------------------
# 3. Master Evaluation Runner (Tables, Significance, Costs, Folds, FF6, Plots)
# ---------------------------------------------------------------------------
def run_full_evaluation(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    cand_bts: dict[str, dict[str, pd.Series]],
    ensemble_results: list[dict[str, Any]],
    ablation_df: pd.DataFrame,
    tab_dir: Path,
    fig_dir: Path,
    wgt_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Compute master leaderboard, significance tests, cost grid, folds, FF6/sector, plots & weights."""
    # Benchmark EW Russell 1000
    ew_bt = {
        "gross": bundle.ew_mkt_1d,
        "net": bundle.ew_mkt_1d,
        "net_allin": bundle.ew_mkt_1d,
        "traded": pd.Series(0.0, index=bundle.dates),
        "tc": pd.Series(0.0, index=bundle.dates),
        "borrow": pd.Series(0.0, index=bundle.dates),
        "impact": pd.Series(0.0, index=bundle.dates),
        "gross_exp": pd.Series(1.0, index=bundle.dates),
        "net_exp": pd.Series(1.0, index=bundle.dates),
    }

    # Best Individual candidate by OOS Net2 Sharpe
    ind_sh = {
        cid: float(
            bt["net"].iloc[WARMUP_DAYS:].mean()
            / (bt["net"].iloc[WARMUP_DAYS:].std(ddof=1) + 1e-12)
            * math.sqrt(ANN_DAYS)
        )
        for cid, bt in cand_bts.items()
    }
    best_ind_id = max(ind_sh, key=ind_sh.get)
    best_ind_ret = cand_bts[best_ind_id]["net"]

    all_entries: dict[str, dict[str, Any]] = {}
    rows = []
    # Add 24 Individuals
    for cid, bt in cand_bts.items():
        all_entries[cid] = {
            "name": cid,
            "group": "Individual_v7",
            "schedule": None,
            "weights": cand_weights[cid],
            "bt": bt,
        }
        rows.append(
            compute_full_metrics_row(
                cid, "Individual_v7", bt, cand_weights[cid], None, bundle, best_ind_ret
            )
        )
    # Add EW Market benchmark
    rows.append(
        compute_full_metrics_row(
            "Bench_EW_Russell1000", "Benchmark", ew_bt, None, None, bundle, best_ind_ret
        )
    )
    # Add all Ensembles
    for res in ensemble_results:
        all_entries[res["name"]] = res
        rows.append(
            compute_full_metrics_row(
                res["name"],
                res["group"],
                res["bt"],
                res["weights"],
                res["schedule"],
                bundle,
                best_ind_ret,
            )
        )

    lb = (
        pd.DataFrame(rows)
        .sort_values("Net2_Sharpe", ascending=False)
        .reset_index(drop=True)
    )
    lb.to_csv(tab_dir / "tab08_master_leaderboard.csv", index=False)

    # Identify top finalists across categories for deep evaluation
    ew_24_name = "A1_EW_All24"
    ew_24_ret = (
        all_entries[ew_24_name]["bt"]["net"]
        .iloc[WARMUP_DAYS:]
        .to_numpy(dtype=np.float64)
    )
    bi_ret_arr = best_ind_ret.iloc[WARMUP_DAYS:].to_numpy(dtype=np.float64)

    ens_rows = lb[~lb["Group"].isin(["Individual_v7", "Benchmark"])]
    top_names = [best_ind_id, ew_24_name]
    for grp in ens_rows["Group"].unique():
        sub = ens_rows[ens_rows["Group"] == grp]
        if not sub.empty:
            top_names.append(str(sub.iloc[0]["Strategy"]))
    # Also add runner-up top overall ensembles up to 10 names
    for nm in ens_rows["Strategy"].head(6):
        if nm not in top_names:
            top_names.append(str(nm))
    top_names = list(dict.fromkeys(top_names))

    # 1. Statistical Significance + Circular Block Bootstrap
    sig_rows = []
    for nm in top_names:
        r_arr = (
            all_entries[nm]["bt"]["net"].iloc[WARMUP_DAYS:].to_numpy(dtype=np.float64)
        )
        d_ew, z_ew, p_ew = ledoit_wolf_sharpe_test(r_arr, ew_24_ret)
        d_bi, z_bi, p_bi = ledoit_wolf_sharpe_test(r_arr, bi_ret_arr)
        boot = circular_block_bootstrap(r_arr, n_boot=1000, block_len=21, seed=42)
        sig_rows.append(
            {
                "Strategy": nm,
                "Group": all_entries[nm]["group"],
                "dSh_vs_EW24": round(d_ew, 3),
                "z_vs_EW24": round(z_ew, 2),
                "p_vs_EW24": round(p_ew, 4),
                "dSh_vs_BestInd": round(d_bi, 3),
                "z_vs_BestInd": round(z_bi, 2),
                "p_vs_BestInd": round(p_bi, 4),
                **{k: round(v, 3) for k, v in boot.items()},
            }
        )
    pd.DataFrame(sig_rows).to_csv(
        tab_dir / "tab08_top_ensemble_significance.csv", index=False
    )

    # 2. Multi-Cost Sensitivity Grid (0, 2, 5, 10, 15 bps + AllIn@2)
    r1_np = bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64)
    adv_np = bundle.adv20_df.to_numpy(dtype=np.float64)
    sig_np = bundle.sigma_prev_df.to_numpy(dtype=np.float64)
    rf_np = bundle.ff6_df["rf"].to_numpy(dtype=np.float64)
    cost_rows = []
    for nm in top_names:
        W_m = all_entries[nm]["weights"]
        row_c = {"Strategy": nm, "Group": all_entries[nm]["group"]}
        for bps in COST_GRID_BPS:
            btc = backtest_weights_daily(
                W_m,
                r1_np,
                bundle.dates,
                cost_bps=bps,
                adv20_np=adv_np,
                sigma_np=sig_np,
                rf_np=rf_np,
            )
            rc = btc["net"].iloc[WARMUP_DAYS:]
            sh = float(rc.mean() / (rc.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS))
            row_c[f"Sh_{int(bps)}bps"] = round(sh, 3)
            if int(bps) in (2, 5):
                rai = btc["net_allin"].iloc[WARMUP_DAYS:]
                row_c[f"AllInSh_{int(bps)}bps"] = round(
                    float(rai.mean() / (rai.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS)),
                    3,
                )
        cost_rows.append(row_c)
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(tab_dir / "tab08_cost_sensitivity.csv", index=False)

    # 3. 14-Fold Walk-Forward Stability Table
    fold_rows = []
    folds_oos = sorted(bundle.fold_ser.iloc[WARMUP_DAYS:].dropna().unique().tolist())
    for nm in top_names:
        r_ser = all_entries[nm]["bt"]["net"]
        r_d = {"Strategy": nm}
        shs = []
        for fd in folds_oos:
            mask = (bundle.fold_ser == fd) & (
                np.arange(len(bundle.dates)) >= WARMUP_DAYS
            )
            rf_sub = r_ser.loc[mask]
            sh_f = (
                float(
                    rf_sub.mean() / (rf_sub.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS)
                )
                if len(rf_sub) > 10
                else np.nan
            )
            r_d[str(fd)] = round(sh_f, 3) if np.isfinite(sh_f) else np.nan
            if np.isfinite(sh_f):
                shs.append(sh_f)
        r_d["PosFolds"] = f"{sum(x > 0 for x in shs)}/{len(shs)}"
        r_d["MinFold_Sh"] = round(min(shs), 3) if shs else np.nan
        r_d["MeanFold_Sh"] = round(float(np.mean(shs)), 3) if shs else np.nan
        fold_rows.append(r_d)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(tab_dir / "tab08_fold_stability.csv", index=False)

    # 4. FF6 Attribution & 12-Sector Exposure
    ff_oos = bundle.ff6_df.iloc[WARMUP_DAYS:]
    X_ff = np.column_stack(
        [
            np.ones(len(ff_oos)),
            ff_oos["mktrf"].to_numpy(),
            ff_oos["smb"].to_numpy(),
            ff_oos["hml"].to_numpy(),
            ff_oos["rmw"].to_numpy(),
            ff_oos["cma"].to_numpy(),
            ff_oos["umd"].to_numpy(),
        ]
    )
    ff_names = ["Alpha_Ann(%)", "MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
    ff_rows = []
    sec_rows = []
    uniq_secs = sorted(set(bundle.sectors.tolist()))
    for nm in top_names:
        r_ex = (
            all_entries[nm]["bt"]["net"].iloc[WARMUP_DAYS:] - ff_oos["rf"]
        ).to_numpy(dtype=np.float64)
        res = newey_west_ols(r_ex, X_ff, max_lag=5)
        b, t = res["beta"], res["tstat"]
        row_ff = {"Strategy": nm, "R2": round(float(res["r2"]), 3)}
        row_ff["Alpha_Ann(%)"] = round(float(b[0] * ANN_DAYS * 100.0), 2)
        row_ff["Alpha_tstat"] = round(float(t[0]), 2)
        for j, fn in enumerate(ff_names[1:], start=1):
            row_ff[fn] = round(float(b[j]), 3)
            row_ff[f"{fn}_t"] = round(float(t[j]), 2)
        ff_rows.append(row_ff)

        W_oos = all_entries[nm]["weights"][WARMUP_DAYS:]
        s_row = {"Strategy": nm}
        for sec in uniq_secs:
            m_sec = bundle.sectors == sec
            s_row[sec] = round(float(W_oos[:, m_sec].sum(axis=1).mean() * 100.0), 2)
        sec_rows.append(s_row)

    ff_table = pd.DataFrame(ff_rows)
    ff_table.to_csv(tab_dir / "tab08_factor_attribution_ff6.csv", index=False)
    pd.DataFrame(sec_rows).to_csv(tab_dir / "tab08_sector_exposure.csv", index=False)

    # 5. Allocation Weight Diagnostics for scheduled ensembles
    alloc_rows = []
    for res in ensemble_results:
        sch = res.get("schedule")
        if sch is None:
            continue
        s_oos = sch.iloc[WARMUP_DAYS:]
        for col in s_oos.columns:
            col_s = s_oos[col]
            if col_s.max() > 1e-4:
                alloc_rows.append(
                    {
                        "Ensemble": res["name"],
                        "Constituent": col,
                        "Mean_W(%)": round(float(col_s.mean() * 100.0), 2),
                        "Min_W(%)": round(float(col_s.min() * 100.0), 2),
                        "Max_W(%)": round(float(col_s.max() * 100.0), 2),
                    }
                )
    pd.DataFrame(alloc_rows).to_csv(
        tab_dir / "tab08_allocation_weights_summary.csv", index=False
    )

    # 6. Export top consolidated weights (long sparse format + schedules)
    best_overall = str(ens_rows.iloc[0]["Strategy"])
    best_dyn = (
        str(ens_rows[ens_rows["Group"] == "Method_E_Dynamic"].iloc[0]["Strategy"])
        if (ens_rows["Group"] == "Method_E_Dynamic").any()
        else best_overall
    )
    best_hier = (
        str(ens_rows[ens_rows["Group"] == "Method_F_Hierarchical"].iloc[0]["Strategy"])
        if (ens_rows["Group"] == "Method_F_Hierarchical").any()
        else best_overall
    )
    best_overlay = (
        str(ens_rows[ens_rows["Group"] == "Method_G_RiskOverlay"].iloc[0]["Strategy"])
        if (ens_rows["Group"] == "Method_G_RiskOverlay").any()
        else best_overall
    )

    for export_nm in dict.fromkeys(
        [best_overall, best_hier, best_dyn, best_overlay, ew_24_name]
    ):
        W_exp = all_entries[export_nm]["weights"]
        r_idx, c_idx = np.nonzero(np.abs(W_exp) > 1e-6)
        df_w = pd.DataFrame(
            {
                "date": bundle.dates[r_idx],
                "security_id": bundle.security_ids[c_idx],
                "weight": W_exp[r_idx, c_idx].astype(np.float32),
            }
        )
        df_w.to_parquet(wgt_dir / f"weights_{export_nm}.parquet", index=False)
        if all_entries[export_nm]["schedule"] is not None:
            all_entries[export_nm]["schedule"].to_csv(
                wgt_dir / f"schedule_{export_nm}.csv"
            )

    # -----------------------------------------------------------------------
    # 7. Charts (Section 8 & Section 10)
    # -----------------------------------------------------------------------
    oos_dates = bundle.dates[WARMUP_DAYS:]
    plot_cands = list(
        dict.fromkeys(
            [best_ind_id, ew_24_name, best_overall, best_hier, best_dyn, best_overlay]
        )
    )

    # Fig 08_01: Compounded Net @ 2bps NAV & Underwater Drawdown
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    for nm in plot_cands:
        r_s = all_entries[nm]["bt"]["net"].iloc[WARMUP_DAYS:]
        nav = (1.0 + r_s).cumprod()
        dd = (nav / np.maximum.accumulate(np.maximum(nav, 1.0)) - 1.0) * 100.0
        lw = 2.4 if nm in (best_overall, best_overlay) else 1.4
        ax1.plot(
            oos_dates,
            nav,
            label=f"{nm} (Sh={lb.set_index('Strategy').loc[nm, 'Net2_Sharpe']:.2f})",
            linewidth=lw,
        )
        ax2.plot(oos_dates, dd, linewidth=lw * 0.85, label=nm)
    nav_ew = (1.0 + bundle.ew_mkt_1d.iloc[WARMUP_DAYS:]).cumprod()
    ax1.plot(
        oos_dates,
        nav_ew,
        color="black",
        linestyle="--",
        alpha=0.65,
        linewidth=1.2,
        label="Bench_EW_Russell1000",
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("Compounded Net NAV (log, 2 bps)")
    ax1.set_title("v7 Ensemble vs Best Individual & Benchmark (Strict OOS 2011-2024)")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(alpha=0.25)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig08_01_cum_nav_and_dd.png", dpi=160)
    plt.close()

    # Fig 08_02: Rolling 252d Sharpe & Rolling 63d Volatility
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    for nm in plot_cands[:5]:
        r_s = all_entries[nm]["bt"]["net"].iloc[WARMUP_DAYS:]
        r_sh = (r_s.rolling(252).mean() / r_s.rolling(252).std()) * math.sqrt(ANN_DAYS)
        r_vol = r_s.rolling(63).std() * math.sqrt(ANN_DAYS) * 100.0
        ax1.plot(oos_dates, r_sh, label=nm, linewidth=1.6)
        ax2.plot(oos_dates, r_vol, label=nm, linewidth=1.4)
    ax1.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Rolling 252d Net Sharpe")
    ax1.set_title(
        "Rolling Risk-Adjusted Performance & Volatility Stability (OOS 2011-2024)"
    )
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(alpha=0.25)
    ax2.set_ylabel("Rolling 63d Ann. Vol (%)")
    ax2.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig08_02_rolling_sharpe_and_vol.png", dpi=160)
    plt.close()

    # Fig 08_03: Risk-Return Frontier (CAGR vs MaxDD) & Transaction Cost Decay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ind_df = lb[lb["Group"] == "Individual_v7"]
    ens_df = lb[~lb["Group"].isin(["Individual_v7", "Benchmark"])]
    ax1.scatter(
        ind_df["Comp_MaxDD(%)"],
        ind_df["CAGR(%)"],
        s=45,
        alpha=0.65,
        color="#7f8c8d",
        label="24 Individual v7 Strategies",
    )
    ax1.scatter(
        ens_df["Comp_MaxDD(%)"],
        ens_df["CAGR(%)"],
        s=65,
        alpha=0.85,
        color="#1f77b4",
        label="Combined Ensembles (A-G)",
    )
    for nm in [best_ind_id, ew_24_name, best_overall, best_overlay]:
        row = lb.set_index("Strategy").loc[nm]
        ax1.scatter(
            [row["Comp_MaxDD(%)"]],
            [row["CAGR(%)"]],
            s=120,
            edgecolors="black",
            zorder=5,
        )
        ax1.annotate(
            nm,
            (row["Comp_MaxDD(%)"], row["CAGR(%)"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
        )
    ax1.set_xlabel("Compounded Max Drawdown (%)")
    ax1.set_ylabel("OOS Net CAGR (%) @ 2 bps")
    ax1.set_title("Risk-Return Pareto Frontier (Top-Left = Dominates)")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    bps_cols = [f"Sh_{int(b)}bps" for b in COST_GRID_BPS]
    for _, rc in cost_df.head(7).iterrows():
        ax2.plot(
            list(COST_GRID_BPS),
            [rc[c] for c in bps_cols],
            marker="o",
            linewidth=1.6,
            label=rc["Strategy"],
        )
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=0.9)
    ax2.set_xlabel("One-Way Transaction Cost (bps)")
    ax2.set_ylabel("OOS Net Sharpe")
    ax2.set_title("Transaction Cost Decay Across 0-15 bps/side")
    ax2.legend(fontsize=7.5)
    ax2.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig08_03_risk_return_frontier_and_cost_decay.png", dpi=160)
    plt.close()

    # Fig 08_04: Dynamic/Hierarchical Allocation Weights Over Time
    sch_plot = (
        all_entries[best_hier]["schedule"]
        if all_entries[best_hier]["schedule"] is not None
        else all_entries[best_dyn]["schedule"]
    )
    fig, ax = plt.subplots(figsize=(13, 5.5))
    if sch_plot is not None:
        s_sub = sch_plot.iloc[WARMUP_DAYS:]
        top_cols = s_sub.mean().sort_values(ascending=False).head(8).index.tolist()
        s_top = s_sub[top_cols].copy()
        if len(s_sub.columns) > 8:
            s_top["Other_Sleeves"] = s_sub[
                [c for c in s_sub.columns if c not in top_cols]
            ].sum(axis=1)
        ax.stackplot(
            oos_dates,
            [s_top[c].to_numpy() * 100.0 for c in s_top.columns],
            labels=list(s_top.columns),
            alpha=0.85,
        )
        ax.set_ylabel("Allocation Weight (%)")
        ax.set_ylim(0, 100)
        ax.set_title(
            f"Walk-Forward Strategy Allocation History ({best_hier if all_entries[best_hier]['schedule'] is not None else best_dyn})"
        )
        ax.legend(loc="upper left", fontsize=7.5, ncol=3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig08_04_allocation_weights.png", dpi=160)
    plt.close()

    # Fig 08_05: Fold Stability & FF6 Exposure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    f_cols = [c for c in fold_df.columns if c.startswith("fold_")]
    f_mat = fold_df.set_index("Strategy")[f_cols].to_numpy(dtype=np.float64)
    im = ax1.imshow(f_mat, cmap="RdYlGn", vmin=-0.5, vmax=3.0, aspect="auto")
    ax1.set_xticks(range(len(f_cols)))
    ax1.set_xticklabels(
        [c.replace("fold_", "F") for c in f_cols], rotation=45, fontsize=8
    )
    ax1.set_yticks(range(len(fold_df)))
    ax1.set_yticklabels(fold_df["Strategy"], fontsize=7.5)
    for i in range(f_mat.shape[0]):
        for j in range(f_mat.shape[1]):
            if np.isfinite(f_mat[i, j]):
                ax1.text(
                    j, i, f"{f_mat[i, j]:.1f}", ha="center", va="center", fontsize=6.5
                )
    ax1.set_title("OOS Walk-Forward Fold Sharpe Matrix")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    ff_sub = ff_table.set_index("Strategy")[
        ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
    ].head(6)
    ff_sub.plot(kind="barh", ax=ax2, width=0.78)
    ax2.axvline(0.0, color="black", linewidth=0.8)
    ax2.set_title("FF6 Factor Exposures (OOS Daily Returns)")
    ax2.set_xlabel("Factor Beta")
    ax2.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig08_05_fold_stability_and_ff6.png", dpi=160)
    plt.close()

    # Fig 10_01: Ablation Impact Chart
    if not ablation_df.empty:
        ab = ablation_df.iloc[1:].copy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
        cols_c = ["#d62728" if x < 0 else "#2ca02c" for x in ab["Delta_Sharpe"]]
        ax1.barh(ab["Ablation"], ab["Delta_Sharpe"], color=cols_c, alpha=0.85)
        ax1.axvline(0.0, color="black", linewidth=0.9)
        ax1.set_title("Ablation Impact on OOS Net2 Sharpe vs ERC_All24")
        ax1.set_xlabel("Delta Sharpe")
        ax1.grid(alpha=0.25)

        cols_d = ["#2ca02c" if x < 0 else "#d62728" for x in ab["Delta_MaxDD(%)"]]
        ax2.barh(ab["Ablation"], ab["Delta_MaxDD(%)"], color=cols_d, alpha=0.85)
        ax2.axvline(0.0, color="black", linewidth=0.9)
        ax2.set_title("Ablation Impact on Max Drawdown (%) vs ERC_All24")
        ax2.set_xlabel("Delta MaxDD (%) [Negative = Improvement]")
        ax2.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig10_01_ablations.png", dpi=160)
        plt.close()

    return lb, all_entries
