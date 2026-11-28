"""Method A: Static Equal-Weight, Within/Across-Family, Priority Pairs & Corrected 5d Rebalance."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from v27.backtest.simulator import backtest_weights_daily
from v27.config import (
    CANDIDATES_1D_16,
    CANDIDATES_24,
    EXPERIMENTS_V7,
    STRATEGIES_V7,
    PortfolioBacktestConfig,
)
from v27.data.bundle import V7DataBundle
from v27.experiments.common import run_consolidated_strategy
from v27.optimization.schedule import static_schedule
from v27.risk.constraints import apply_portfolio_constraints
from v27.signal_analysis.horizon_audit import non_overlapping_rebal_weights


def build_method_a_static(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    ret_net_df: pd.DataFrame,
    bt_cfg: PortfolioBacktestConfig | None = None,
) -> list[dict[str, Any]]:
    """Evaluate static candidate allocations and matched 5-day non-overlapping rebalance."""
    bt_cfg = bt_cfg or PortfolioBacktestConfig()
    dates = bundle.dates
    out: list[dict[str, Any]] = []

    def add_static(name: str, group: str, wmap: dict[str, float]) -> None:
        sched = static_schedule(dates, wmap)
        out.append(
            run_consolidated_strategy(
                name, group, sched, cand_weights, bundle, bt_cfg=bt_cfg
            )
        )

    add_static(
        "A1_EW_All24",
        "Method_A_Static",
        {c: 1.0 / len(CANDIDATES_24) for c in CANDIDATES_24},
    )
    add_static(
        "A2_EW_1d16",
        "Method_A_Static",
        {c: 1.0 / len(CANDIDATES_1D_16) for c in CANDIDATES_1D_16},
    )

    for exp in EXPERIMENTS_V7:
        cols = [f"{exp}__{s}" for s in STRATEGIES_V7]
        add_static(
            f"A3_EW_Exp_{exp}", "Method_A_Static", {c: 1.0 / len(cols) for c in cols}
        )

    for fam in (
        "rank_RAW",
        "top10_InvVol",
        "top10_InvVol_Buf",
        "top10_BetaNeu_InvVol_Buf",
        "top10_HOLD5",
        "top10_InvVol_HOLD5",
    ):
        cols = [f"{e}__{fam}" for e in EXPERIMENTS_V7]
        add_static(
            f"A4_EW_Fam_{fam}", "Method_A_Static", {c: 1.0 / len(cols) for c in cols}
        )

    add_static(
        "A5_Pair_rawInvVol_riskBetaNeu",
        "Method_A_Static",
        {"raw_t1__top10_InvVol": 0.50, "risk_t1__top10_BetaNeu_InvVol_Buf": 0.50},
    )
    add_static(
        "A6_Pair_riskBetaNeu_risk5HOLD5",
        "Method_A_Static",
        {
            "risk_t1__top10_BetaNeu_InvVol_Buf": 0.50,
            "risk_t5__top10_InvVol_HOLD5": 0.50,
        },
    )
    add_static(
        "A7_Pair_rawBuf_riskBetaNeu",
        "Method_A_Static",
        {"raw_t1__top10_Buf10_20": 0.50, "risk_t1__top10_BetaNeu_InvVol_Buf": 0.50},
    )

    w_ret = ret_net_df.iloc[: bt_cfg.warmup_days]
    w_sh = (w_ret.mean() / (w_ret.std(ddof=1) + 1e-12)) * math.sqrt(bt_cfg.ann_days)
    top_pool = w_sh[w_sh > 0.6].index.tolist() or list(CANDIDATES_24)
    cmat = w_ret[top_pool].corr()
    best_pair = ("raw_t1__top10_BetaNeu_InvVol_Buf", "risk_t5__top10_InvVol_HOLD5")
    best_score = -1e9
    for i, c1 in enumerate(top_pool):
        for c2 in top_pool[i + 1 :]:
            if c1.split("__")[0] == c2.split("__")[0]:
                continue
            sc = 0.5 * (float(w_sh[c1]) + float(w_sh[c2])) - 1.5 * float(
                cmat.loc[c1, c2]
            )
            if sc > best_score:
                best_score = sc
                best_pair = (c1, c2)
    add_static(
        "A8_Discovered_LowCorr_Pair",
        "Method_A_Static",
        {best_pair[0]: 0.50, best_pair[1]: 0.50},
    )

    W_5d_t5 = non_overlapping_rebal_weights(
        pd.DataFrame(
            cand_weights["risk_t5__top10_InvVol_HOLD5"],
            index=dates,
            columns=bundle.security_ids,
        ),
        bundle.ret1d_df,
        stride=5,
    ).to_numpy(dtype=np.float64)
    W_5d_t1 = non_overlapping_rebal_weights(
        pd.DataFrame(
            cand_weights["risk_t1__top10_BetaNeu_InvVol_Buf"],
            index=dates,
            columns=bundle.security_ids,
        ),
        bundle.ret1d_df,
        stride=5,
    ).to_numpy(dtype=np.float64)
    W_5d_blend = apply_portfolio_constraints(
        0.50 * W_5d_t5 + 0.50 * W_5d_t1, max_single_w=0.05
    )
    bt_5d = backtest_weights_daily(
        W_5d_blend,
        bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64),
        dates,
        cost_bps=bt_cfg.baseline_cost_bps,
        adv20_np=bundle.adv20_df.to_numpy(dtype=np.float64),
        sigma_np=bundle.sigma_prev_df.to_numpy(dtype=np.float64),
        rf_np=bundle.ff6_df["rf"].to_numpy(dtype=np.float64),
        book_size=bt_cfg.book_size,
        short_borrow_ann=bt_cfg.short_borrow_ann,
        impact_eta=bt_cfg.impact_coeff,
        ann_days=bt_cfg.ann_days,
    )
    out.append(
        {
            "name": "A9_Corr5dRebal_Pair_t5_t1",
            "group": "Method_A_5dRebal",
            "schedule": None,
            "weights": W_5d_blend,
            "bt": bt_5d,
        }
    )
    return out
