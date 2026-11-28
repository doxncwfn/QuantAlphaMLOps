"""Shared consolidated execution helper for strategy experiments and optimizers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from v27.backtest.simulator import backtest_weights_daily
from v27.config import PortfolioBacktestConfig, RiskOverlayConfig
from v27.data.bundle import V7DataBundle
from v27.portfolio.consolidation import consolidate_positions
from v27.risk.constraints import apply_portfolio_constraints


def run_consolidated_strategy(
    name: str,
    group: str,
    alloc_schedule: pd.DataFrame,
    cand_weights: dict[str, np.ndarray],
    bundle: V7DataBundle,
    max_single_w: float | None = 0.05,
    cost_bps: float | None = None,
    apply_full_constraints: bool = False,
    bt_cfg: PortfolioBacktestConfig | None = None,
    risk_cfg: RiskOverlayConfig | None = None,
) -> dict[str, Any]:
    """Consolidate sleeve weights daily, net internal trades, apply optional constraints, and backtest."""
    bt_cfg = bt_cfg or PortfolioBacktestConfig()
    risk_cfg = risk_cfg or RiskOverlayConfig()

    W = consolidate_positions(alloc_schedule, cand_weights)
    if apply_full_constraints:
        W = apply_portfolio_constraints(
            W,
            beta60_np=bundle.beta60_df.to_numpy(dtype=np.float64),
            sectors=bundle.sectors,
            adv20_np=bundle.adv20_df.to_numpy(dtype=np.float64),
            r1d_np=bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64),
            max_single_w=risk_cfg.max_single_w,
            max_gross=risk_cfg.max_gross,
            max_abs_beta=risk_cfg.max_abs_beta,
            max_sector_net=risk_cfg.max_sector_net,
            max_adv_frac=risk_cfg.max_adv_frac,
            book_size=bt_cfg.book_size,
        )
    elif max_single_w is not None:
        W = apply_portfolio_constraints(
            W, max_single_w=max_single_w, book_size=bt_cfg.book_size
        )

    eff_cost = bt_cfg.baseline_cost_bps if cost_bps is None else cost_bps
    bt = backtest_weights_daily(
        W,
        bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64),
        bundle.dates,
        cost_bps=eff_cost,
        adv20_np=bundle.adv20_df.to_numpy(dtype=np.float64),
        sigma_np=bundle.sigma_prev_df.to_numpy(dtype=np.float64),
        rf_np=bundle.ff6_df["rf"].to_numpy(dtype=np.float64),
        book_size=bt_cfg.book_size,
        short_borrow_ann=bt_cfg.short_borrow_ann,
        impact_eta=bt_cfg.impact_coeff,
        ann_days=bt_cfg.ann_days,
        inception_zero_trade=False,
    )
    return {
        "name": name,
        "group": group,
        "schedule": alloc_schedule,
        "weights": W,
        "bt": bt,
    }
