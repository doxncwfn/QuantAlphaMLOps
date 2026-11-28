"""Method B: Prediction-level score blending & score-vs-construction decomposition."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v27.backtest.simulator import backtest_weights_daily
from v27.config import STRATEGIES_V7, PortfolioBacktestConfig
from v27.data.bundle import V7DataBundle
from v27.portfolio.constructors import build_8_strategies_for_score, strat_top10_buf
from v27.portfolio.neutralization import neutralize_beta


def build_method_b_score_ensembles(
    bundle: V7DataBundle,
    tab_dir: Path | None = None,
    bt_cfg: PortfolioBacktestConfig | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Combine normalized scores prior to portfolio construction and audit score vs rule gains."""
    bt_cfg = bt_cfg or PortfolioBacktestConfig()
    dates = bundle.dates
    r1_df = bundle.ret1d_df
    r1_np = r1_df.fillna(0.0).to_numpy(dtype=np.float64)
    sig_df = bundle.sigma_prev_df
    beta_df = bundle.beta60_df
    adv_np = bundle.adv20_df.to_numpy(dtype=np.float64)
    sig_np = sig_df.to_numpy(dtype=np.float64)
    rf_np = bundle.ff6_df["rf"].to_numpy(dtype=np.float64)

    def _xs_z(df: pd.DataFrame) -> pd.DataFrame:
        return df.sub(df.mean(axis=1), axis=0).div(
            df.std(axis=1).clip(lower=1e-8), axis=0
        )

    def _xs_rank(df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, pct=True) - 0.5

    z_raw, z_risk1, z_risk5 = (
        _xs_z(bundle.scores["raw_t1"]),
        _xs_z(bundle.scores["risk_t1"]),
        _xs_z(bundle.scores["risk_t5"]),
    )
    rk_raw, rk_risk1, rk_risk5 = (
        _xs_rank(bundle.scores["raw_t1"]),
        _xs_rank(bundle.scores["risk_t1"]),
        _xs_rank(bundle.scores["risk_t5"]),
    )

    lam_grid = [0.0, 0.25, 0.40, 0.50, 0.60, 0.75, 1.0]
    grid_rets: dict[float, np.ndarray] = {}
    for lam in lam_grid:
        sc_g = lam * z_raw + (1.0 - lam) * z_risk1
        inv_vol_g = (1.0 / sig_df).where(sc_g.notna()).fillna(0.0)
        w_g = strat_top10_buf(neutralize_beta(sc_g, beta_df), inv_vol_g).to_numpy(
            dtype=np.float64
        )
        bt_g = backtest_weights_daily(
            w_g, r1_np, dates, cost_bps=bt_cfg.baseline_cost_bps
        )
        grid_rets[lam] = bt_g["net"].to_numpy(dtype=np.float64)

    T = len(dates)
    lam_series = np.full(T, 0.50, dtype=np.float64)
    for t in range(bt_cfg.warmup_days, T):
        if t == bt_cfg.warmup_days or ((t - bt_cfg.warmup_days) % 63 == 0):
            end_i = max(t - bt_cfg.purge_days, 20)
            best_lam, best_sh = 0.50, -1e9
            for lam in lam_grid:
                seg = grid_rets[lam][:end_i]
                sh = float(np.mean(seg) / (np.std(seg, ddof=1) + 1e-12))
                if sh > best_sh:
                    best_sh, best_lam = sh, lam
            lam_series[t:] = best_lam

    sc_wf_opt = z_raw.mul(lam_series, axis=0) + z_risk1.mul(1.0 - lam_series, axis=0)
    blended_scores = {
        "B1_ScoreEW_Rank_1d": 0.5 * rk_raw + 0.5 * rk_risk1,
        "B2_ScoreEW_Z_1d": 0.5 * z_raw + 0.5 * z_risk1,
        "B3_ScoreWF_Opt_1d": sc_wf_opt,
        "B4_ScoreEW_Z_All3": (z_raw + z_risk1 + z_risk5) / 3.0,
        "B5_ScoreEW_Rank_All3": (rk_raw + rk_risk1 + rk_risk5) / 3.0,
    }
    all_score_sets: dict[str, pd.DataFrame] = {
        "Standalone_raw_t1": bundle.scores["raw_t1"],
        "Standalone_risk_t1": bundle.scores["risk_t1"],
        "Standalone_risk_t5": bundle.scores["risk_t5"],
        **blended_scores,
    }

    decomp_rows = []
    promoted: list[dict[str, Any]] = []
    key_constructions = (
        "top10_InvVol",
        "top10_InvVol_Buf",
        "top10_BetaNeu_InvVol_Buf",
        "top10_InvVol_HOLD5",
    )

    for sc_name, sc_df in all_score_sets.items():
        w_dict = build_8_strategies_for_score(
            sc_df, r1_df, sig_df, beta_df, hold_days=bt_cfg.hold_days
        )
        for sname in STRATEGIES_V7:
            W_np = w_dict[sname].to_numpy(dtype=np.float64)
            bt = backtest_weights_daily(
                W_np,
                r1_np,
                dates,
                cost_bps=bt_cfg.baseline_cost_bps,
                adv20_np=adv_np,
                sigma_np=sig_np,
                rf_np=rf_np,
            )
            r_oos = bt["net"].iloc[bt_cfg.warmup_days :]
            nav = (1.0 + r_oos).cumprod()
            mdd = float(
                -(nav / np.maximum.accumulate(np.maximum(nav, 1.0)) - 1.0).min() * 100.0
            )
            cagr = float((nav.iloc[-1] ** (bt_cfg.ann_days / len(r_oos)) - 1.0) * 100.0)
            sh = float(
                r_oos.mean() / (r_oos.std(ddof=1) + 1e-12) * math.sqrt(bt_cfg.ann_days)
            )
            tvr = float(bt["traded"].iloc[bt_cfg.warmup_days :].mean() * 100.0)
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
    if tab_dir is not None:
        decomp_df.to_csv(
            tab_dir / "tab05_methodB_score_vs_construction_decomposition.csv",
            index=False,
        )
    return promoted, decomp_df
