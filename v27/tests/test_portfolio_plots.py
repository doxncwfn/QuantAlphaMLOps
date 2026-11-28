"""Smoke tests for portfolio_plots time series generation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from v27.config import PortfolioBacktestConfig
from v27.visualization.portfolio_plots import plot_portfolio_evaluation_suite


def test_plot_portfolio_evaluation_suite_writes_sharpe_and_turnover(tmp_path: Path):
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    r_s = pd.Series(rng.normal(0.0005, 0.01, size=len(dates)), index=dates)
    to_s = pd.Series(rng.uniform(0.01, 0.05, size=len(dates)), index=dates)
    bt = {"net": r_s, "traded": to_s}

    strat = "S1"
    all_entries = {
        strat: {
            "bt": bt,
            "weights": np.zeros((len(dates), 2)),
            "schedule": pd.DataFrame({"A": np.ones(len(dates))}, index=dates),
        }
    }
    lb = pd.DataFrame(
        {
            "Strategy": [strat],
            "Group": ["Ensemble"],
            "Net2_Sharpe": [1.0],
            "Comp_MaxDD(%)": [5.0],
            "CAGR(%)": [10.0],
        }
    )
    cost_df = pd.DataFrame(
        {
            "Strategy": [strat],
            "Sh_0bps": [1.1],
            "Sh_2bps": [1.0],
            "Sh_5bps": [0.9],
            "Sh_10bps": [0.7],
            "Sh_15bps": [0.5],
        }
    )
    fold_df = pd.DataFrame({"Strategy": [strat], "fold_1": [1.0]})
    ff_table = pd.DataFrame(
        {
            "Strategy": [strat],
            "MKT": [0.0],
            "SMB": [0.0],
            "HML": [0.0],
            "RMW": [0.0],
            "CMA": [0.0],
            "MOM": [0.0],
        }
    )
    eval_out = {
        "leaderboard": lb,
        "all_entries": all_entries,
        "cost_df": cost_df,
        "fold_df": fold_df,
        "ff_table": ff_table,
        "best_ind_id": strat,
        "ew_24_name": strat,
        "best_overall": strat,
        "best_hier": strat,
        "best_dyn": strat,
        "best_overlay": strat,
        "dyn_weights_df": pd.DataFrame(),
        "hier_weights_df": pd.DataFrame(),
    }
    bundle = SimpleNamespace(dates=dates, ew_mkt_1d=r_s)
    bt_cfg = PortfolioBacktestConfig(warmup_days=10, ann_days=252.0)

    plot_portfolio_evaluation_suite(bundle, eval_out, pd.DataFrame(), tmp_path, bt_cfg)

    assert (tmp_path / "fig08_08_sharpe_and_daily_turnover.png").exists()
