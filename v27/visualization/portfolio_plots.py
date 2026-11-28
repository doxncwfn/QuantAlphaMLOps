"""Portfolio NAV, drawdown, rolling Sharpe/vol, frontier/cost decay, weights, and ablation charts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v27.config import PortfolioBacktestConfig
from v27.data.bundle import V7DataBundle
from v27.visualization.style import save_fig, shade_crisis_eras


def plot_portfolio_evaluation_suite(
    bundle: V7DataBundle,
    eval_out: dict[str, Any],
    ablation_df: pd.DataFrame,
    fig_dir: Path,
    bt_cfg: PortfolioBacktestConfig | None = None,
) -> None:
    """Generate fig08_01..fig08_05 and fig10_01 diagnostic portfolio charts."""
    bt_cfg = bt_cfg or PortfolioBacktestConfig()
    wu, ann = bt_cfg.warmup_days, bt_cfg.ann_days
    lb = eval_out["leaderboard"]
    all_entries = eval_out["all_entries"]
    cost_df = eval_out["cost_df"]
    fold_df = eval_out["fold_df"]
    ff_table = eval_out["ff_table"]
    best_ind_id = eval_out["best_ind_id"]
    ew_24_name = eval_out["ew_24_name"]
    best_overall = eval_out["best_overall"]
    best_hier = eval_out["best_hier"]
    best_dyn = eval_out["best_dyn"]
    best_overlay = eval_out["best_overlay"]

    oos_dates = bundle.dates[wu:]
    plot_cands = list(
        dict.fromkeys(
            [best_ind_id, ew_24_name, best_overall, best_hier, best_dyn, best_overlay]
        )
    )

    # Fig 08_01: Compounded Net NAV & Underwater Drawdown
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    lb_idx = lb.set_index("Strategy")
    for nm in plot_cands:
        r_s = all_entries[nm]["bt"]["net"].iloc[wu:]
        nav = (1.0 + r_s).cumprod()
        dd = (nav / np.maximum.accumulate(np.maximum(nav, 1.0)) - 1.0) * 100.0
        lw = 2.4 if nm in (best_overall, best_overlay) else 1.4
        ax1.plot(
            oos_dates,
            nav,
            label=f"{nm} (Sh={lb_idx.loc[nm, 'Net2_Sharpe']:.2f})",
            linewidth=lw,
        )
        ax2.plot(oos_dates, dd, linewidth=lw * 0.85, label=nm)
    nav_ew = (1.0 + bundle.ew_mkt_1d.iloc[wu:]).cumprod()
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
    ax1.set_title("v27 Ensemble vs Best Individual & Benchmark (Strict OOS 2011-2024)")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(alpha=0.25)
    shade_crisis_eras(ax1, add_labels=True)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.25)
    shade_crisis_eras(ax2)
    save_fig(fig, fig_dir / "fig08_01_cum_nav_and_dd.png")

    # Fig 08_02: Rolling 252d Sharpe & Rolling 63d Vol
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    for nm in plot_cands[:5]:
        r_s = all_entries[nm]["bt"]["net"].iloc[wu:]
        ax1.plot(
            oos_dates,
            (r_s.rolling(252).mean() / r_s.rolling(252).std()) * math.sqrt(ann),
            label=nm,
            linewidth=1.6,
        )
        ax2.plot(
            oos_dates,
            r_s.rolling(63).std() * math.sqrt(ann) * 100.0,
            label=nm,
            linewidth=1.4,
        )
    ax1.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Rolling 252d Net Sharpe")
    ax1.set_title(
        "Rolling Risk-Adjusted Performance & Volatility Stability (OOS 2011-2024)"
    )
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(alpha=0.25)
    shade_crisis_eras(ax1, add_labels=True)
    ax2.set_ylabel("Rolling 63d Ann. Vol (%)")
    ax2.grid(alpha=0.25)
    shade_crisis_eras(ax2)
    save_fig(fig, fig_dir / "fig08_02_rolling_sharpe_and_vol.png")

    # Fig 08_08: Rolling 252d Sharpe & Daily Turnover Time Series (best strategy)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    bt_nm = all_entries[best_overall]["bt"]
    r_s = bt_nm["net"].iloc[wu:]
    to_s = bt_nm["traded"].iloc[wu:] * 100.0
    ax1.plot(
        oos_dates,
        (r_s.rolling(252).mean() / r_s.rolling(252).std()) * math.sqrt(ann),
        label=best_overall,
        linewidth=0.8,
    )
    ax2.plot(
        oos_dates,
        to_s,
        label=best_overall,
        linewidth=0.6,
        alpha=0.85,
    )
    ax1.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Rolling 252d Net Sharpe")
    ax1.set_title("Rolling Sharpe & Daily Turnover Time Series")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.25)
    shade_crisis_eras(ax1, add_labels=True)
    ax2.set_ylabel("Daily Turnover (%)")
    ax2.grid(alpha=0.25)
    shade_crisis_eras(ax2)
    save_fig(fig, fig_dir / "fig08_08_sharpe_and_daily_turnover.png")

    # Fig 08_03: Frontier & Transaction Cost Decay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ind_df = lb[lb["Group"] == "Individual_v7"]
    ens_df = lb[~lb["Group"].isin(["Individual_v7", "Benchmark"])]
    ax1.scatter(
        ind_df["Comp_MaxDD(%)"],
        ind_df["CAGR(%)"],
        s=45,
        alpha=0.65,
        color="#7f8c8d",
        label="24 Individual Strategies",
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
        row = lb_idx.loc[nm]
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
    ax1.set_title("Risk-Return Pareto Frontier (Top-Left Dominates)")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    cost_grid: Sequence[float] = bt_cfg.cost_grid_bps
    bps_cols = [f"Sh_{int(b)}bps" for b in cost_grid]
    for _, rc in cost_df.head(7).iterrows():
        ax2.plot(
            list(cost_grid),
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
    save_fig(fig, fig_dir / "fig08_03_risk_return_frontier_and_cost_decay.png")

    # Fig 08_04: Allocation Stack
    sch_plot = (
        all_entries[best_hier]["schedule"]
        if all_entries[best_hier]["schedule"] is not None
        else all_entries[best_dyn]["schedule"]
    )
    fig, ax = plt.subplots(figsize=(13, 5.5))
    if sch_plot is not None:
        s_sub = sch_plot.iloc[wu:]
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
            f"Walk-Forward Allocation History ({best_hier if all_entries[best_hier]['schedule'] is not None else best_dyn})"
        )
        ax.legend(loc="upper left", fontsize=7.5, ncol=3)
        shade_crisis_eras(ax, add_labels=True)
    save_fig(fig, fig_dir / "fig08_04_allocation_weights.png")

    # Fig 08_05: Fold Stability & FF6 Exposures
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

    ff_table.set_index("Strategy")[["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]].head(
        6
    ).plot(kind="barh", ax=ax2, width=0.78)
    ax2.axvline(0.0, color="black", linewidth=0.8)
    ax2.set_title("FF6 Factor Exposures (OOS Daily Returns)")
    ax2.set_xlabel("Factor Beta")
    ax2.grid(alpha=0.25)
    save_fig(fig, fig_dir / "fig08_05_fold_stability_and_ff6.png")

    # Fig 10_01: Ablation Impact Chart
    if not ablation_df.empty:
        ab = ablation_df.iloc[1:].copy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
        ax1.barh(
            ab["Ablation"],
            ab["Delta_Sharpe"],
            color=["#d62728" if x < 0 else "#2ca02c" for x in ab["Delta_Sharpe"]],
            alpha=0.85,
        )
        ax1.axvline(0.0, color="black", linewidth=0.9)
        ax1.set_title("Ablation Impact on OOS Net2 Sharpe vs ERC_All24")
        ax1.set_xlabel("Delta Sharpe")
        ax1.grid(alpha=0.25)

        ax2.barh(
            ab["Ablation"],
            ab["Delta_MaxDD(%)"],
            color=["#2ca02c" if x < 0 else "#d62728" for x in ab["Delta_MaxDD(%)"]],
            alpha=0.85,
        )
        ax2.axvline(0.0, color="black", linewidth=0.9)
        ax2.set_title("Ablation Impact on Max Drawdown (%) vs ERC_All24")
        ax2.set_xlabel("Delta MaxDD (%) [Negative = Improvement]")
        ax2.grid(alpha=0.25)
        save_fig(fig, fig_dir / "fig10_01_ablations.png")
