"""Method E: Causal dynamic rolling allocators with EMA smoothing & L1 hysteresis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from v27.config import PortfolioBacktestConfig
from v27.data.bundle import V7DataBundle
from v27.experiments.common import run_consolidated_strategy
from v27.optimization.estimators import ledoit_wolf_cov
from v27.optimization.schedule import causal_wf_schedule
from v27.optimization.solvers import (
    project_simplex_bounded,
    solve_erc,
    solve_inv_vol,
    solve_min_var,
)


def build_method_e_dynamic(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    ret_net_df: pd.DataFrame,
    ortho_cols: list[str],
    bt_cfg: PortfolioBacktestConfig | None = None,
) -> list[dict[str, Any]]:
    """Run Method E dynamic allocators (E1..E7) over orthogonal candidate sleeves."""
    bt_cfg = bt_cfg or PortfolioBacktestConfig()
    dates = bundle.dates
    R_df = ret_net_df[ortho_cols]
    N = len(ortho_cols)
    out: list[dict[str, Any]] = []
    wu, pg, ann = bt_cfg.warmup_days, bt_cfg.purge_days, bt_cfg.ann_days

    ew_mkt = bundle.ew_mkt_1d.reindex(dates).fillna(0.0).to_numpy(dtype=np.float64)
    vol20_mkt = pd.Series(ew_mkt).rolling(20).std().to_numpy(
        dtype=np.float64
    ) * math.sqrt(ann)
    xs_disp = (
        bundle.ret1d_df.std(axis=1, skipna=True)
        .rolling(20)
        .mean()
        .to_numpy(dtype=np.float64)
    )

    sched_riv = causal_wf_schedule(
        R_df,
        dates,
        lambda X, wp, t: solve_inv_vol(X, w_max=0.35),
        warmup=wu,
        purge=pg,
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
            bt_cfg=bt_cfg,
        )
    )

    sched_rerc = causal_wf_schedule(
        R_df,
        dates,
        lambda X, wp, t: solve_erc(ledoit_wolf_cov(X), w_max=0.35),
        warmup=wu,
        purge=pg,
        roll_window=126,
        ema_alpha=0.30,
        hyst_tau=0.015,
    )
    out.append(
        run_consolidated_strategy(
            "E2_RollERC_126d_Hyst",
            "Method_E_Dynamic",
            sched_rerc,
            cand_weights,
            bundle,
            bt_cfg=bt_cfg,
        )
    )

    sched_rmv = causal_wf_schedule(
        R_df,
        dates,
        lambda X, wp, t: solve_min_var(ledoit_wolf_cov(X), w_max=0.35, l2_reg=0.15),
        warmup=wu,
        purge=pg,
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
            bt_cfg=bt_cfg,
        )
    )

    def _roll_perf_solve(X: np.ndarray, wp: np.ndarray, t: int) -> np.ndarray:
        sh = np.mean(X, axis=0) / (np.std(X, axis=0, ddof=1) + 1e-12) * math.sqrt(ann)
        iv = solve_inv_vol(X, w_max=0.35)
        tilt = np.exp(0.35 * np.clip(sh - np.median(sh), -2.0, 2.0))
        return project_simplex_bounded(iv * tilt, w_max=0.35, w_min=0.02)

    sched_perf = causal_wf_schedule(
        R_df,
        dates,
        _roll_perf_solve,
        warmup=wu,
        purge=pg,
        roll_window=63,
        ema_alpha=0.30,
        hyst_tau=0.02,
    )
    out.append(
        run_consolidated_strategy(
            "E4_RollPerfTilt_63d",
            "Method_E_Dynamic",
            sched_perf,
            cand_weights,
            bundle,
            bt_cfg=bt_cfg,
        )
    )

    def _dd_aware_solve(X: np.ndarray, wp: np.ndarray, t: int) -> np.ndarray:
        base = solve_erc(ledoit_wolf_cov(X), w_max=0.35)
        nav = np.cumprod(1.0 + np.clip(X, -0.5, 1.0), axis=0)
        hwm = np.maximum.accumulate(np.maximum(nav, 1.0), axis=0)
        cur_dd = -(nav[-1] / hwm[-1] - 1.0)
        pen = np.exp(-8.0 * np.clip(cur_dd, 0.0, 0.25))
        return project_simplex_bounded(base * pen, w_max=0.35, w_min=0.02)

    sched_dd = causal_wf_schedule(
        R_df,
        dates,
        _dd_aware_solve,
        warmup=wu,
        purge=pg,
        roll_window=126,
        ema_alpha=0.30,
        hyst_tau=0.015,
    )
    out.append(
        run_consolidated_strategy(
            "E5_DrawdownAware_ERC",
            "Method_E_Dynamic",
            sched_dd,
            cand_weights,
            bundle,
            bt_cfg=bt_cfg,
        )
    )

    low_turn_mask = np.array(
        ["HOLD5" in c or "Buf" in c or "rank_RAW" in c for c in ortho_cols], dtype=float
    )

    def _regime_solve(X: np.ndarray, wp: np.ndarray, t: int) -> np.ndarray:
        base = solve_erc(ledoit_wolf_cov(X), w_max=0.35)
        v_hist, d_hist = vol20_mkt[: max(t - 1, 20)], xs_disp[: max(t - 1, 20)]
        v_now, d_now = vol20_mkt[t - 1], xs_disp[t - 1]
        hi_vol = np.isfinite(v_now) and v_now > np.nanpercentile(v_hist, 75)
        hi_disp = np.isfinite(d_now) and d_now > np.nanpercentile(d_hist, 65)
        tilt = np.ones(N, dtype=np.float64)
        if hi_vol:
            tilt = np.where(low_turn_mask > 0.5, 1.25, 0.80)
        elif hi_disp:
            tilt = np.where(low_turn_mask < 0.5, 1.20, 0.85)
        return project_simplex_bounded(base * tilt, w_max=0.35, w_min=0.02)

    sched_reg = causal_wf_schedule(
        R_df,
        dates,
        _regime_solve,
        warmup=wu,
        purge=pg,
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
            bt_cfg=bt_cfg,
        )
    )

    def _corr_aware_solve(X: np.ndarray, wp: np.ndarray, t: int) -> np.ndarray:
        cov = ledoit_wolf_cov(X)
        base = solve_erc(cov, w_max=0.35)
        sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        C = np.clip(cov / np.outer(sd, sd), -0.99, 0.99)
        avg_c = (C.sum(axis=1) - 1.0) / max(N - 1, 1)
        pen = np.exp(-0.75 * np.maximum(avg_c, 0.0))
        return project_simplex_bounded(base * pen, w_max=0.35, w_min=0.02)

    sched_corr = causal_wf_schedule(
        R_df,
        dates,
        _corr_aware_solve,
        warmup=wu,
        purge=pg,
        roll_window=126,
        ema_alpha=0.30,
        hyst_tau=0.015,
    )
    out.append(
        run_consolidated_strategy(
            "E7_CorrAware_ERC",
            "Method_E_Dynamic",
            sched_corr,
            cand_weights,
            bundle,
            bt_cfg=bt_cfg,
        )
    )
    return out
