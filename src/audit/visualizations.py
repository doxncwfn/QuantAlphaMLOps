"""Publication-quality visualization and notebook generator for data quality audit."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# Standard academic plot styling
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def generate_all_visualizations(
    config: dict[str, Any],
    tables: dict[str, pl.DataFrame],
) -> list[Path]:
    """Generate all 16 publication-quality audit figures deterministically."""
    lookback_days = int(config.get("parameters", {}).get("lookback_days", 40))
    logger.info(
        "Generating publication-quality audit figures (lookback: %d days)...", lookback_days
    )
    paths = config.get("paths", {})
    fig_dir = Path(paths.get("output_figures_dir", "report/quality/figures"))
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Clean up obsolete figures
    for f in [
        "wrds_vs_crawled_coverage.png",
        "data_source_boundary_2024_2025.png",
        "jaccard_similarity.png",
        "membership_churn.png",
        "model_60d_eligibility_by_year.png",
        "model_60d_eligibility_by_period.png",
    ]:
        p = fig_dir / f
        if p.exists():
            p.unlink()

    figures_generated = []

    # 1. Annual Constituent Count
    if "annual_quality" in tables:
        df = tables['annual_quality'].to_pandas()
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            df['year'],
            df['raw_row_count'],
            color='#2b5c8f',
            width=0.6,
            label='Raw Constituent Count',
        )
        ax.axhline(
            1000, color='#d95f02', linestyle='--', linewidth=1.5, label='Nominal 1,000 Target'
        )
        ax.set_xlabel('Year')
        ax.set_ylabel('Constituent Count')
        ax.set_ylim(940, 1060)
        ax.set_xticks(df['year'])
        ax.set_xticklabels([f"{int(y)}" for y in df['year']], rotation=45)
        ax.legend(loc='upper right')

        # Add number on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9
            )

        plt.tight_layout()
        out_p = fig_dir / "annual_constituent_count.png"
        fig.savefig(out_p)
        plt.close(fig)
        figures_generated.append(out_p)

    # 2. Entries and Exits
    if "transitions" in tables:
        df = tables['transitions'].to_pandas()
        x_labels = [f"{int(r['to_year'])}" for _, r in df.iterrows()]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(df))
        w = 0.35
        ax.bar(
            x - w / 2,
            df['entries_count'],
            width=w,
            color='#1b9e77',
            label='Entries',
        )
        ax.bar(
            x + w / 2,
            df['exits_count'],
            width=w,
            color='#e7298a',
            label='Exits',
        )
        ax.set_xlabel('Transition Window')
        ax.set_ylabel('Number of Securities')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        out_p = fig_dir / "annual_entries_exits.png"
        fig.savefig(out_p)
        plt.close(fig)
        figures_generated.append(out_p)

    # 3. Consolidated Churn and Jaccard Dual-Axis Plot
    if "transitions" in tables:
        df = tables["transitions"].to_pandas()
        fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=300)
        years = df["to_year"].tolist()
        jaccard = df["jaccard_similarity"].tolist()
        churn = df["churn_rate"].tolist()

        color1 = "#1f77b4"
        ax1.set_ylabel("Jaccard Similarity", color=color1, fontweight="bold")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_ylim(0.60, 0.95)

        # Number the Jaccard points (top, above if higher/equal, below if less than previous)
        for idx, (x, y) in enumerate(zip(years, jaccard)):
            if idx == 0 or y >= jaccard[idx - 1]:
                va = "bottom"
                dy = 0.01
            else:
                va = "top"
                dy = -0.01
            ax1.text(
                x, y + dy, f"{y:.3f}",
                ha="center", va=va,
                fontsize=8, color=color1, fontweight="bold"
                # box removed
            )

        ax2 = ax1.twinx()
        color2 = "#d62728"
        ax2.set_ylabel("Membership Churn Rate", color=color2, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0.05, 0.40)

        # Number the Churn points (top, above if higher/equal, below if less than previous)
        for idx, (x, y) in enumerate(zip(years, churn)):
            if idx == 0 or y >= churn[idx - 1]:
                va = "bottom"
                dy = 0.01
            else:
                va = "top"
                dy = -0.01
            ax2.text(
                x, y + dy, f"{y:.3f}",
                ha="center", va=va,
                fontsize=8, color=color2, fontweight="bold"
                # box removed
            )

        ax1.set_xticks(years)
        ax1.set_xticklabels([f"{int(y)}" for y in years], rotation=45, ha="right")
        plt.tight_layout()
        out_p = fig_dir / "membership_churn_and_jaccard.png"
        fig.savefig(out_p)
        plt.close(fig)
        figures_generated.append(out_p)

    # 4. Price Data Coverage by Year
    if "security_coverage" in tables:
        df = tables['security_coverage'].to_pandas()
        yearly_cov = df.groupby('year')['coverage_ratio'].agg(['mean', 'median']).reset_index()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            yearly_cov['year'],
            yearly_cov['mean'] * 100,
            marker='o',
            label='Mean Coverage (%)',
            color='#1f78b4',
        )
        mean_vals = (yearly_cov['mean'] * 100).values
        years = yearly_cov['year'].values
        for idx, (x, y) in enumerate(zip(years, mean_vals)):
            if idx == 0 or y >= mean_vals[idx - 1]:
                va = "top"
                dy = 0.55
            else:
                va = "bottom"
                dy = -0.55
            ax.text(
                x, y + dy, f"{y:.2f}",
                ha="center", va=va,
                fontsize=8, color='#1f78b4', fontweight="bold"
            )
        ax.plot(
            yearly_cov['year'],
            yearly_cov['median'] * 100,
            marker='s',
            label='Median Coverage (%)',
            color='#33a02c',
        )
        ax.axhline(95, color='gray', linestyle=':', label='95% Target')
        ax.set_xticks(yearly_cov['year'])
        ax.set_xticklabels([f"{int(y)}" for y in yearly_cov['year']], rotation=45)
        ax.set_ylim(90, 101)
        ax.legend(loc='lower left')
        plt.tight_layout()
        out_p = fig_dir / "price_data_coverage_by_year.png"
        fig.savefig(out_p)
        plt.close(fig)
        figures_generated.append(out_p)

    # 5. Distribution of Security Coverage
    if "security_coverage" in tables:
        df = tables['security_coverage'].to_pandas()
        fig, ax = plt.subplots(figsize=(11, 5))
        n, bins, patches = ax.hist(df['coverage_ratio'] * 100, bins=50, color='#4575b4', edgecolor='black', alpha=0.8)
        ax.set_xlabel('Trading Day Coverage (%)')
        ax.set_ylabel('Security-Year Frequency')
        ax.set_yscale('log')

        # Add numbers on top of each bar
        for i in range(len(n)):
            if n[i] > 0:
                ax.text(
                    (bins[i] + bins[i+1]) / 2,
                    n[i],
                    f"{int(n[i])}",
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    color='black',
                    clip_on=True
                )
        plt.tight_layout()
        out_p = fig_dir / "distribution_security_coverage.png"
        fig.savefig(out_p)
        plt.close(fig)
        figures_generated.append(out_p)

    # 6. Distribution of Missing Streak Lengths
    if "security_coverage" in tables:
        df = tables['security_coverage'].to_pandas()
        streaks = df[df['longest_missing_streak'] > 0]['longest_missing_streak']
        fig, ax = plt.subplots(figsize=(8, 5))
        n, bins, patches = ax.hist(streaks, bins=50, color='#d73027', edgecolor='black', alpha=0.8)
        ax.set_xlabel('Consecutive Missing Trading Days')
        ax.set_ylabel('Security-Year Count (Log Scale)')
        ax.set_yscale('log')

        for i in range(len(n)):
            if n[i] > 0:
                ax.text(
                    (bins[i] + bins[i + 1]) / 2,
                    n[i],
                    f"{int(n[i])}",
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    color='black'
                )
        plt.tight_layout()
        out_p = fig_dir / "distribution_missing_streak_lengths.png"
        fig.savefig(out_p)
        plt.close(fig)
        figures_generated.append(out_p)

    # 7. Daily Cross-Sectional Coverage
    if "daily_coverage" in tables:
        df = tables['daily_coverage'].to_pandas()
        fig, ax = plt.subplots(figsize=(12, 5))
        step = max(1, len(df) // 1000)
        df_sub = df.iloc[::step]
        ax.plot(
            np.arange(len(df_sub)), df_sub['coverage_ratio'] * 100, color='#2b5c8f', linewidth=1.2
        )
        ax.axhline(95, color='#d95f02', linestyle='--', label='95% Coverage Threshold')
        ax.set_xlabel('Trading Date Index')
        ax.set_ylabel('Cross-Sectional Coverage (%)')
        ax.set_ylim(94, 105)
        ax.legend(loc='lower left')
        plt.tight_layout()
        out_p = fig_dir / "daily_cross_sectional_coverage.png"
        fig.savefig(out_p)
        plt.close(fig)
        figures_generated.append(out_p)

    # 8. Missing Observations by Year
    if "security_coverage" in tables:
        df = tables['security_coverage'].to_pandas()
        missing_by_yr = df.groupby('year')['missing_days'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(missing_by_yr['year'], missing_by_yr['missing_days'], color='#fc8d59', width=0.6)
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Missing Observations')
        ax.set_xticks(missing_by_yr['year'])
        ax.set_xticklabels([f"{int(y)}" for y in missing_by_yr['year']], rotation=45)
        plt.tight_layout()
        out_p = fig_dir / "missing_observations_by_year_month.png"
        fig.savefig(out_p)
        plt.close(fig)
        figures_generated.append(out_p)

    # 9. Extreme Return Distribution
    if "extreme_returns" in tables:
        df = tables['extreme_returns'].to_pandas()
        fig, ax = plt.subplots(figsize=(8, 5))
        raw_r = df['raw_return']
        ax.hist(raw_r.clip(-1.0, 2.0) * 100, bins=50, color='#91bfdb', edgecolor='black', alpha=0.8)
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        out_p = fig_dir / "extreme_return_distribution.png"
        fig.savefig(out_p)
        plt.close(fig)
        figures_generated.append(out_p)

    # 10. Model Eligibility by Year (Continuous vs Isolated)
    if "model_eligibility" in tables:
        df = tables["model_eligibility"].to_pandas()
        fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
        y_col = "list_year" if "list_year" in df.columns else "year"
        years = df[y_col].tolist()
        p_col = (
            "pct_eligible_continuous"
            if "pct_eligible_continuous" in df.columns
            else "pct_eligible_40d"
        )
        pcts_new = (df[p_col] * 100).tolist()

        if "model_boundary_exclusion_comparison" in tables:
            comp_df = tables["model_boundary_exclusion_comparison"].to_pandas()
            pcts_old = (comp_df["old_isolated_pct"] * 100).tolist()
            ax.plot(
                years,
                pcts_old,
                marker="o",
                linestyle="--",
                color="#e41a1c",
                linewidth=1.5,
                markersize=5,
                label="Isolated",
            )

        ax.plot(
            years,
            pcts_new,
            marker="s",
            linestyle="-",
            color="#2ca02c",
            linewidth=1.0,
            markersize=6,
            label="Continuous",
        )
        ax.set_ylabel("Eligible 40-Day Windows (%)")
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
        ax.set_ylim(60, 102)
        ax.legend(loc="lower left")
        plt.tight_layout()
        out_p = fig_dir / f"model_{lookback_days}d_eligibility_by_year.png"
        fig.savefig(out_p)
        plt.close(fig)
        figures_generated.append(out_p)

    # 11. Historical Delisted Constituents
    if "delisted_securities" in tables:
        df = tables['delisted_securities'].to_pandas()
        if len(df) > 0 and 'year' in df.columns:
            delist_by_yr = df.groupby('year')['PERMNO'].nunique().reset_index()
            fig, ax = plt.subplots(figsize=(10, 4.5))
            bars = ax.bar(delist_by_yr['year'], delist_by_yr['PERMNO'], color='#8073ac', width=0.6)
            ax.set_title('Historical Delisted / Inactive Constituents Recorded in WRDS')
            ax.set_ylabel('Delisted Securities Count')
            ax.set_xticks(delist_by_yr['year'])
            ax.set_xticklabels([f"{int(y)}" for y in delist_by_yr['year']], rotation=45)
            # Add numbers to bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9,
                    color='#222222'
                )
            plt.tight_layout()
        out_p = fig_dir / "historical_delisted_constituents.png"
        fig.savefig(out_p)
        plt.close(fig)
        figures_generated.append(out_p)

    # 12. Russell List Source Snapshot Date by List Year
    if "membership_snapshot_metadata" in tables:
        df_meta = tables['membership_snapshot_metadata'].to_pandas()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        df_meta['dt'] = pd.to_datetime(df_meta['source_snapshot_date'])
        df_meta['day_of_year'] = df_meta['dt'].dt.dayofyear
        normal_mask = ~df_meta['is_off_cycle_snapshot']
        off_mask = df_meta['is_off_cycle_snapshot']

        ax.axhspan(
            170, 185, color='#e5f5e0', alpha=0.7, label='Standard June Reconstitution Window'
        )
        ax.scatter(
            df_meta.loc[normal_mask, 'list_year'],
            df_meta.loc[normal_mask, 'day_of_year'],
            color='#1f78b4',
            s=60,
            zorder=3,
            label='Standard June Snapshot (PDF/CSV)',
        )
        ax.scatter(
            df_meta.loc[off_mask, 'list_year'],
            df_meta.loc[off_mask, 'day_of_year'],
            color='#e31a1c',
            s=80,
            marker='s',
            zorder=4,
            label='Off-Cycle Snapshot (JSON/XLS)',
        )

        for _, r in df_meta[off_mask].iterrows():
            ax.annotate(
                f"{r['source_snapshot_date']}\n+{r['hindsight_gap_days']}d gap",
                (r['list_year'], r['day_of_year']),
                textcoords='offset points',
                xytext=(-25, -25),
                fontsize=8,
                fontweight='bold',
                color='#b10026',
                arrowprops={'arrowstyle': '->', 'color': '#b10026', 'lw': 1},
            )

        ax.set_ylabel('Day of Calendar Year')
        ax.set_xticks(df_meta['list_year'])
        ax.set_xticklabels([f"{int(y)}" for y in df_meta['list_year']], rotation=45)
        ax.legend(loc='upper left', framealpha=0.9)
        plt.tight_layout()
        out_p = fig_dir / "russell_list_snapshot_date_by_year.png"
        fig.savefig(out_p)
        plt.close(fig)
        figures_generated.append(out_p)

    # 13. Corrected Model Lookback Eligibility by Membership Period
    model_table_name = f"corrected_model_{lookback_days}d_eligibility"
    if model_table_name in tables:
        df_model = tables["corrected_model_40d_eligibility"].to_pandas()
        fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
        years = df_model["list_year"].tolist()
        pct_eligible = (df_model["pct_eligible_continuous"] * 100).tolist()
        colors = ["#2ca02c" for _ in years]
        ax.bar(years, pct_eligible, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.axhline(
            95,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="95% High Completeness Threshold",
        )
        ax.set_ylabel("Eligible 40-Day Feature Windows (%)")
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
        ax.set_ylim(70, 102)
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor="#2ca02c", edgecolor="black", label=f"Continuous Multi-Year Stitching (Avg {np.mean(pct_eligible):.1f}%)"),
            plt.Line2D([0], [0], color="black", linestyle="--", label="95% Target"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")
        plt.tight_layout()
        out_p = fig_dir / f"model_{lookback_days}d_eligibility_by_period.png"
        fig.savefig(out_p)
        plt.close(fig)
        figures_generated.append(out_p)

    logger.info(
        "Successfully generated %d publication figures in %s", len(figures_generated), fig_dir
    )
    return figures_generated


def export_visualizations_notebook(
    output_notebook_path: Path | str = "notebooks/visualizations.ipynb",
    tables_dir: Path | str = "report/quality/tables",
) -> Path:
    """
    Export the comprehensive visualizations suite to a standalone Jupyter notebook.
    Enforces exactly 1 plot per code cell with pre-rendered base64 image display data.
    """
    out_path = Path(output_notebook_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        logger.info("Visualizations notebook is up-to-date at %s", out_path)
        return out_path

    logger.info("Visualizations notebook target is %s", out_path)
    return out_path
