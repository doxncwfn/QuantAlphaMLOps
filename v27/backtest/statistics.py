"""Statistical inference, cost-grid sensitivity, fold stability, leaderboard & Polars weight exporter."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

from v27.backtest.metrics import compute_full_metrics_row
from v27.backtest.simulator import backtest_weights_daily
from v27.config import PortfolioBacktestConfig
from v27.data.bundle import V7DataBundle
from v27.explainability.factor_exposure import newey_west_ols


def ledoit_wolf_sharpe_test(
    r_a: np.ndarray, r_b: np.ndarray, max_lag: int = 5, ann_days: float = 252.0
) -> tuple[float, float, float]:
    """Ledoit-Wolf (2008) HAC test for annualized Sharpe difference (r_a - r_b)."""
    a = np.asarray(r_a, dtype=np.float64)
    b = np.asarray(r_b, dtype=np.float64)
    n = len(a)
    mu_a, mu_b = float(a.mean()), float(b.mean())
    s2_a, s2_b = float(np.mean(a**2)), float(np.mean(b**2))
    var_a, var_b = max(s2_a - mu_a**2, 1e-16), max(s2_b - mu_b**2, 1e-16)
    delta_1d = (mu_a / math.sqrt(var_a)) - (mu_b / math.sqrt(var_b))
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
    return (
        float(delta_1d * math.sqrt(ann_days)),
        float(z),
        float(2.0 * (1.0 - norm.cdf(abs(z)))),
    )


def circular_block_bootstrap(
    r: np.ndarray,
    n_boot: int = 1000,
    block_len: int = 21,
    seed: int = 42,
    ann_days: float = 252.0,
) -> dict[str, float]:
    """Circular block bootstrap 95% confidence intervals for Sharpe, CAGR, and MaxDD."""
    rng = np.random.default_rng(seed)
    x = np.asarray(r, dtype=np.float64)
    n = len(x)
    n_blocks = math.ceil(n / block_len)
    sh_b = np.empty(n_boot, dtype=np.float64)
    cagr_b = np.empty(n_boot, dtype=np.float64)
    mdd_b = np.empty(n_boot, dtype=np.float64)
    n_yrs = n / ann_days
    offsets = np.arange(block_len)[None, :]
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)[:, None]
        xb = x[(starts + offsets).ravel()[:n] % n]
        sh_b[b] = xb.mean() / (xb.std(ddof=1) + 1e-12) * math.sqrt(ann_days)
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


def evaluate_and_export_tables(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    cand_bts: dict[str, dict[str, pd.Series]],
    ensemble_results: list[dict[str, Any]],
    tab_dir: Path,
    wgt_dir: Path,
    bt_cfg: PortfolioBacktestConfig | None = None,
) -> dict[str, Any]:
    """Build leaderboard, significance tests, cost grid, WF folds, FF6/sector tables, and Polars sparse weights."""
    bt_cfg = bt_cfg or PortfolioBacktestConfig()
    wu, ann = bt_cfg.warmup_days, bt_cfg.ann_days
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

    ind_sh = {
        cid: float(
            bt["net"].iloc[wu:].mean()
            / (bt["net"].iloc[wu:].std(ddof=1) + 1e-12)
            * math.sqrt(ann)
        )
        for cid, bt in cand_bts.items()
    }
    best_ind_id = max(ind_sh, key=ind_sh.get)  # type: ignore[arg-type]
    best_ind_ret = cand_bts[best_ind_id]["net"]

    all_entries: dict[str, dict[str, Any]] = {}
    rows = []
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
                cid,
                "Individual_v7",
                bt,
                cand_weights[cid],
                None,
                bundle,
                best_ind_ret,
                wu,
                ann,
                bt_cfg.baseline_cost_bps,
            )
        )
    rows.append(
        compute_full_metrics_row(
            "Bench_EW_Russell1000",
            "Benchmark",
            ew_bt,
            None,
            None,
            bundle,
            best_ind_ret,
            wu,
            ann,
            bt_cfg.baseline_cost_bps,
        )
    )
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
                wu,
                ann,
                bt_cfg.baseline_cost_bps,
            )
        )

    lb = (
        pd.DataFrame(rows)
        .sort_values("Net2_Sharpe", ascending=False)
        .reset_index(drop=True)
    )
    pl.from_pandas(lb).write_csv(tab_dir / "tab08_master_leaderboard.csv")

    ew_24_name = "A1_EW_All24"
    ew_24_ret = (
        all_entries[ew_24_name]["bt"]["net"].iloc[wu:].to_numpy(dtype=np.float64)
    )
    bi_ret_arr = best_ind_ret.iloc[wu:].to_numpy(dtype=np.float64)
    ens_rows = lb[~lb["Group"].isin(["Individual_v7", "Benchmark"])]

    top_names = [best_ind_id, ew_24_name]
    for grp in ens_rows["Group"].unique():
        sub = ens_rows[ens_rows["Group"] == grp]
        if not sub.empty:
            top_names.append(str(sub.iloc[0]["Strategy"]))
    for nm in ens_rows["Strategy"].head(6):
        if nm not in top_names:
            top_names.append(str(nm))
    top_names = list(dict.fromkeys(top_names))

    sig_rows = []
    for nm in top_names:
        r_arr = all_entries[nm]["bt"]["net"].iloc[wu:].to_numpy(dtype=np.float64)
        d_ew, z_ew, p_ew = ledoit_wolf_sharpe_test(r_arr, ew_24_ret, ann_days=ann)
        d_bi, z_bi, p_bi = ledoit_wolf_sharpe_test(r_arr, bi_ret_arr, ann_days=ann)
        boot = circular_block_bootstrap(
            r_arr, n_boot=1000, block_len=21, seed=42, ann_days=ann
        )
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
    sig_df = pd.DataFrame(sig_rows)
    pl.DataFrame(sig_rows).write_csv(tab_dir / "tab08_top_ensemble_significance.csv")

    r1_np = bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64)
    adv_np = bundle.adv20_df.to_numpy(dtype=np.float64)
    sig_np = bundle.sigma_prev_df.to_numpy(dtype=np.float64)
    rf_np = bundle.ff6_df["rf"].to_numpy(dtype=np.float64)
    cost_rows = []
    for nm in top_names:
        W_m = all_entries[nm]["weights"]
        row_c: dict[str, Any] = {"Strategy": nm, "Group": all_entries[nm]["group"]}
        for bps in bt_cfg.cost_grid_bps:
            btc = backtest_weights_daily(
                W_m,
                r1_np,
                bundle.dates,
                cost_bps=bps,
                adv20_np=adv_np,
                sigma_np=sig_np,
                rf_np=rf_np,
                book_size=bt_cfg.book_size,
                short_borrow_ann=bt_cfg.short_borrow_ann,
                impact_eta=bt_cfg.impact_coeff,
                ann_days=ann,
            )
            rc = btc["net"].iloc[wu:]
            row_c[f"Sh_{int(bps)}bps"] = round(
                float(rc.mean() / (rc.std(ddof=1) + 1e-12) * math.sqrt(ann)), 3
            )
            if int(bps) in (2, 5):
                rai = btc["net_allin"].iloc[wu:]
                row_c[f"AllInSh_{int(bps)}bps"] = round(
                    float(rai.mean() / (rai.std(ddof=1) + 1e-12) * math.sqrt(ann)), 3
                )
        cost_rows.append(row_c)
    cost_df = pd.DataFrame(cost_rows)
    pl.DataFrame(cost_rows).write_csv(tab_dir / "tab08_cost_sensitivity.csv")

    fold_rows = []
    folds_oos = sorted(bundle.fold_ser.iloc[wu:].dropna().unique().tolist())
    for nm in top_names:
        r_ser = all_entries[nm]["bt"]["net"]
        r_d: dict[str, Any] = {"Strategy": nm}
        shs = []
        for fd in folds_oos:
            mask = (bundle.fold_ser == fd) & (np.arange(len(bundle.dates)) >= wu)
            rf_sub = r_ser.loc[mask]
            sh_f = (
                float(rf_sub.mean() / (rf_sub.std(ddof=1) + 1e-12) * math.sqrt(ann))
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
    pl.DataFrame(fold_rows).write_csv(tab_dir / "tab08_fold_stability.csv")

    ff_oos = bundle.ff6_df.iloc[wu:]
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
    ff_rows, sec_rows = [], []
    uniq_secs = sorted(set(bundle.sectors.tolist()))
    for nm in top_names:
        r_ex = (all_entries[nm]["bt"]["net"].iloc[wu:] - ff_oos["rf"]).to_numpy(
            dtype=np.float64
        )
        res = newey_west_ols(r_ex, X_ff, max_lag=5)
        b, t = res["beta"], res["tstat"]
        row_ff: dict[str, Any] = {
            "Strategy": nm,
            "R2": round(float(res["r2"]), 3),
            "Alpha_Ann(%)": round(float(b[0] * ann * 100.0), 2),
            "Alpha_tstat": round(float(t[0]), 2),
        }
        for j, fn in enumerate(ff_names[1:], start=1):
            row_ff[fn] = round(float(b[j]), 3)
            row_ff[f"{fn}_t"] = round(float(t[j]), 2)
        ff_rows.append(row_ff)
        W_oos = all_entries[nm]["weights"][wu:]
        s_row: dict[str, Any] = {"Strategy": nm}
        for sec in uniq_secs:
            s_row[sec] = round(
                float(W_oos[:, bundle.sectors == sec].sum(axis=1).mean() * 100.0), 2
            )
        sec_rows.append(s_row)
    ff_table = pd.DataFrame(ff_rows)
    pl.DataFrame(ff_rows).write_csv(tab_dir / "tab08_factor_attribution_ff6.csv")
    pl.DataFrame(sec_rows).write_csv(tab_dir / "tab08_sector_exposure.csv")

    alloc_rows = []
    for res in ensemble_results:
        sch = res.get("schedule")
        if sch is None:
            continue
        s_oos = sch.iloc[wu:]
        for col in s_oos.columns:
            if s_oos[col].max() > 1e-4:
                alloc_rows.append(
                    {
                        "Ensemble": res["name"],
                        "Constituent": col,
                        "Mean_W(%)": round(float(s_oos[col].mean() * 100.0), 2),
                        "Min_W(%)": round(float(s_oos[col].min() * 100.0), 2),
                        "Max_W(%)": round(float(s_oos[col].max() * 100.0), 2),
                    }
                )
    pl.DataFrame(alloc_rows).write_csv(tab_dir / "tab08_allocation_weights_summary.csv")

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

    dates_np = bundle.dates.to_numpy()
    for export_nm in dict.fromkeys(
        [best_overall, best_hier, best_dyn, best_overlay, ew_24_name]
    ):
        W_exp = all_entries[export_nm]["weights"]
        r_idx, c_idx = np.nonzero(np.abs(W_exp) > 1e-6)
        pl.DataFrame(
            {
                "date": dates_np[r_idx],
                "security_id": bundle.security_ids[c_idx],
                "weight": W_exp[r_idx, c_idx].astype(np.float32),
            }
        ).write_parquet(wgt_dir / f"weights_{export_nm}.parquet")
        if all_entries[export_nm]["schedule"] is not None:
            all_entries[export_nm]["schedule"].to_csv(
                wgt_dir / f"schedule_{export_nm}.csv"
            )

    return {
        "leaderboard": lb,
        "all_entries": all_entries,
        "cost_df": cost_df,
        "fold_df": fold_df,
        "ff_table": ff_table,
        "sig_df": sig_df,
        "best_ind_id": best_ind_id,
        "ew_24_name": ew_24_name,
        "best_overall": best_overall,
        "best_hier": best_hier,
        "best_dyn": best_dyn,
        "best_overlay": best_overlay,
    }
