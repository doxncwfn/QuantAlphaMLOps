"""WRDS price data coverage and daily cross-sectional coverage audit module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from src.audit.registry import ExceptionRegistry

logger = logging.getLogger(__name__)


def audit_price_and_daily_coverage(
    config: dict[str, Any],
    registry: ExceptionRegistry,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Audit security-level coverage and daily cross-sectional market coverage.

    Returns:
        (security_coverage_df, daily_coverage_df)
    """
    logger.info("Auditing WRDS price coverage and daily cross-sectional coverage...")
    paths = config.get("paths", {})
    wrds_dir = Path(paths.get("wrds_dir", "data/WRDS"))
    processed_dir = Path(paths.get("processed_dir", "data/processed"))

    const_df = pd.read_csv(processed_dir / "russell1000_all_years.csv", keep_default_na=False)

    security_rows = []
    daily_rows = []

    all_years = sorted(const_df["Year"].unique())

    for yr in all_years:
        parquet_file = wrds_dir / f"{yr}.parquet"
        if not parquet_file.exists():
            logger.warning("Missing WRDS parquet for year %d", yr)
            continue

        try:
            df_year = pl.read_parquet(parquet_file)

            # Determine schema (2000-2024 vs 2025-2026)
            cols = df_year.columns
            date_col = "date" if "date" in cols else "Date"
            ticker_col = "TICKER" if "TICKER" in cols else "Ticker"

            if df_year[date_col].dtype != pl.String:
                df_year = df_year.with_columns(
                    pl.col(date_col).dt.strftime("%Y-%m-%d").alias(date_col)
                )

            trading_dates = sorted(df_year[date_col].drop_nulls().unique().to_list())
            n_trading_days = len(trading_dates)

            constituents = const_df[const_df["Year"] == yr]["Ticker"].unique().tolist()
            n_constituents = len(constituents)

            # Daily cross-sectional coverage
            daily_counts = (
                df_year.group_by(date_col)
                .agg(pl.col(ticker_col).drop_nulls().n_unique().alias("n_observed"))
                .sort(date_col)
            )

            for r in daily_counts.iter_rows(named=True):
                d_str = r[date_col]
                obs = r["n_observed"]
                cov_ratio = obs / n_constituents if n_constituents > 0 else 0.0

                daily_rows.append(
                    {
                        "year": int(yr),
                        "date": d_str,
                        "expected_securities": n_constituents,
                        "observed_securities": obs,
                        "missing_securities": n_constituents - obs,
                        "coverage_ratio": round(cov_ratio, 5),
                    }
                )

                if cov_ratio < 0.90:
                    registry.register(
                        category="COVERAGE",
                        severity="HIGH" if cov_ratio < 0.80 else "MEDIUM",
                        year=int(yr),
                        date_start=d_str,
                        date_end=d_str,
                        description=f"Cross-sectional coverage collapse ({cov_ratio:.1%}) on trading date {d_str}.",
                        evidence=f"Observed {obs} / {n_constituents} expected securities on {d_str}.",
                        status="FLAGGED",
                        notes="Investigate market emergency closure, half-day trading, or extraction dropout.",
                    )

            # Security-level coverage
            sec_counts = df_year.group_by(ticker_col).agg(
                [
                    pl.col(date_col).n_unique().alias("observed_days"),
                    pl.col(date_col).min().alias("first_date"),
                    pl.col(date_col).max().alias("last_date"),
                    pl.col(date_col).unique().alias("dates_list"),
                ]
            )

            sec_dict = {
                r[ticker_col]: r
                for r in sec_counts.iter_rows(named=True)
                if r[ticker_col] is not None
            }

            for t in constituents:
                if t in sec_dict:
                    rec = sec_dict[t]
                    obs_days = rec["observed_days"]
                    first_d = rec["first_date"]
                    last_d = rec["last_date"]
                    dates_present = set(rec["dates_list"])

                    dates_in_window = [d for d in trading_dates if first_d <= d <= last_d]

                    longest_streak = 0
                    current_streak = 0
                    num_streaks = 0
                    for d in dates_in_window:
                        if d not in dates_present:
                            current_streak += 1
                            if current_streak == 1:
                                num_streaks += 1
                            if current_streak > longest_streak:
                                longest_streak = current_streak
                        else:
                            current_streak = 0

                    cov = obs_days / n_trading_days
                    security_rows.append(
                        {
                            "year": int(yr),
                            "ticker": t,
                            "expected_days": n_trading_days,
                            "observed_days": obs_days,
                            "missing_days": n_trading_days - obs_days,
                            "coverage_ratio": round(cov, 5),
                            "first_observed_date": first_d,
                            "last_observed_date": last_d,
                            "longest_missing_streak": longest_streak,
                            "missing_streak_count": num_streaks,
                            "is_zero_coverage": False,
                        }
                    )

                    if cov < 0.85:
                        registry.register(
                            category="COVERAGE",
                            severity="HIGH" if cov < 0.50 else "MEDIUM",
                            year=int(yr),
                            ticker=t,
                            description=f"Low trading day coverage ({cov:.1%}) for constituent '{t}' in {yr}.",
                            evidence=f"Observed {obs_days}/{n_trading_days} days. First: {first_d}, Last: {last_d}.",
                            status="FLAGGED",
                            notes="Verify if security delisted, underwent M&A, or joined post-IPO.",
                        )
                else:
                    security_rows.append(
                        {
                            "year": int(yr),
                            "ticker": t,
                            "expected_days": n_trading_days,
                            "observed_days": 0,
                            "missing_days": n_trading_days,
                            "coverage_ratio": 0.0,
                            "first_observed_date": None,
                            "last_observed_date": None,
                            "longest_missing_streak": n_trading_days,
                            "missing_streak_count": 1,
                            "is_zero_coverage": True,
                        }
                    )
                    registry.register(
                        category="COVERAGE",
                        severity="CRITICAL",
                        year=int(yr),
                        ticker=t,
                        description=f"Constituent '{t}' has zero market data observations in {yr} WRDS extract.",
                        evidence=f"0 / {n_trading_days} trading days observed.",
                        status="FLAGGED",
                        notes="Investigate ticker mapping discrepancy or delisting.",
                    )

        except (OSError, pl.exceptions.PolarsError, KeyError, ValueError) as e:
            logger.error("Error auditing coverage for year %d: %s", yr, e)

    security_coverage_df = pl.DataFrame(security_rows).sort(["year", "coverage_ratio", "ticker"])
    daily_coverage_df = pl.DataFrame(daily_rows).sort("date")

    out_dir = Path(paths.get("output_tables_dir", "report/quality/tables"))
    out_dir.mkdir(parents=True, exist_ok=True)

    security_coverage_df.write_parquet(out_dir / "security_year_coverage.parquet")
    security_coverage_df.write_csv(out_dir / "security_year_coverage.csv")

    daily_coverage_df.write_parquet(out_dir / "daily_cross_sectional_coverage.parquet")
    daily_coverage_df.write_csv(out_dir / "daily_cross_sectional_coverage.csv")

    logger.info(
        "Saved security coverage (%d records) and daily cross-sectional coverage (%d dates).",
        len(security_coverage_df),
        len(daily_coverage_df),
    )
    return security_coverage_df, daily_coverage_df
