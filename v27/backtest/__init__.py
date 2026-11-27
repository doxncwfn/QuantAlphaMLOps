"""Backtester simulator, equity metrics, and statistical test modules for v27."""

from __future__ import annotations

from v27.backtest.metrics import (
    compounded_dd_episodes,
    compute_full_metrics_row,
    compute_legacy_brain_stats,
    lo_hac_sharpe,
)
from v27.backtest.simulator import backtest_weights_daily
from v27.backtest.statistics import (
    circular_block_bootstrap,
    evaluate_and_export_tables,
    ledoit_wolf_sharpe_test,
)

__all__ = [
    "backtest_weights_daily",
    "circular_block_bootstrap",
    "compounded_dd_episodes",
    "compute_full_metrics_row",
    "compute_legacy_brain_stats",
    "evaluate_and_export_tables",
    "ledoit_wolf_sharpe_test",
    "lo_hac_sharpe",
]
