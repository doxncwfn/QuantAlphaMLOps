"""Method F: Two-level hierarchical allocation (Prediction, Construction, Dual-Horizon 1d vs 5d)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from v27.config import (
    CANDIDATES_1D_16,
    CANDIDATES_24,
    EXPERIMENTS_V7,
    STRATEGIES_V7,
    PortfolioBacktestConfig,
)
from v27.data.bundle import V7DataBundle
from v27.experiments.common import run_consolidated_strategy
from v27.optimization.estimators import ledoit_wolf_cov
from v27.optimization.schedule import causal_wf_schedule
from v27.optimization.solvers import solve_erc


def build_method_f_hierarchical(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    ret_net_df: pd.DataFrame,
    bt_cfg: PortfolioBacktestConfig | None = None,
) -> list[dict[str, Any]]:
    """Run Method F two-level hierarchical ERC allocators (F1..F4)."""
    bt_cfg = bt_cfg or PortfolioBacktestConfig()
    dates = bundle.dates
    out: list[dict[str, Any]] = []
    wu, pg = bt_cfg.warmup_days, bt_cfg.purge_days

    def _build_two_level(
        groups: dict[str, list[str]], l1_erc: bool, name: str
    ) -> dict[str, Any]:
        l2_scheds: dict[str, pd.DataFrame] = {}
        grp_rets = {}
        for gname, cols in groups.items():
            sch = causal_wf_schedule(
                ret_net_df[cols],
                dates,
                lambda X, wp, t, m=0.45: solve_erc(ledoit_wolf_cov(X), w_max=m),
                warmup=wu,
                purge=pg,
                ema_alpha=0.50,
            )
            l2_scheds[gname] = sch
            grp_rets[gname] = (
                ret_net_df[cols].to_numpy(dtype=np.float64)
                * sch.to_numpy(dtype=np.float64)
            ).sum(axis=1)

        grp_ret_df = pd.DataFrame(grp_rets, index=dates)
        if l1_erc:
            l1_sched = causal_wf_schedule(
                grp_ret_df,
                dates,
                lambda X, wp, t: solve_erc(ledoit_wolf_cov(X), w_max=0.60),
                warmup=wu,
                purge=pg,
                ema_alpha=0.50,
            )
        else:
            l1_sched = pd.DataFrame(
                1.0 / len(groups), index=dates, columns=list(groups.keys())
            )

        full_sched = pd.DataFrame(0.0, index=dates, columns=list(CANDIDATES_24))
        for gname, cols in groups.items():
            g_w = l1_sched[gname].to_numpy(dtype=np.float64)[:, None]
            full_sched[cols] += g_w * l2_scheds[gname][cols].to_numpy(dtype=np.float64)
        return run_consolidated_strategy(
            name,
            "Method_F_Hierarchical",
            full_sched,
            cand_weights,
            bundle,
            bt_cfg=bt_cfg,
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
