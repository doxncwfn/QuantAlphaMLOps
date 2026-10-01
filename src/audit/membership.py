"""Russell 1000 constituent membership integrity, transition, and ticker formatting audit."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from src.audit.registry import ExceptionRegistry

logger = logging.getLogger(__name__)


def audit_membership_integrity(
    config: dict[str, Any],
    registry: ExceptionRegistry,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Audit Russell 1000 membership consistency across all years and transitions.

    Returns:
        (annual_quality_df, transitions_df)
    """
    logger.info("Auditing Russell 1000 membership integrity...")
    paths = config.get("paths", {})
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    csv_file = processed_dir / "russell1000_all_years.csv"

    # Always use keep_default_na=False to avoid interpreting 'NA' (Nabisco) as NaN
    df_raw = pd.read_csv(csv_file, keep_default_na=False)

    # Derive source map from configured snapshots, with standard fallbacks
    snapshots = config.get("snapshots", [])
    source_map: dict[int, tuple[str, str]] = {}
    for s in snapshots:
        source_map[int(s["list_year"])] = (
            s.get("source_format", "UNKNOWN"),
            s.get("source_file", "unknown"),
        )

    # Fallback source mapping if snapshots not present in config
    if not source_map:
        for y in range(2000, 2027):
            if y in (
                2000,
                2001,
                2002,
                2003,
                2004,
                2005,
                2006,
                2007,
                2009,
                2010,
                2011,
                2012,
                2013,
                2014,
                2015,
                2016,
                2017,
                2018,
                2020,
                2021,
                2022,
            ):
                source_map[y] = ("PDF", f"{y}.pdf")
            elif y == 2008:
                source_map[y] = ("CSV", "2008.csv")
            elif y in (2019, 2023, 2024):
                source_map[y] = ("JSON", f"{y}.json")
            elif y in (2025, 2026):
                source_map[y] = ("XLS", f"{y}.xls")

    years = sorted(df_raw["Year"].unique())
    quality_rows = []
    yearly_constituent_sets: dict[int, set[str]] = {}

    for year in years:
        year_df = df_raw[df_raw["Year"] == year]
        raw_count = len(year_df)
        raw_tickers = year_df["Ticker"].tolist()

        empty_count = sum(1 for t in raw_tickers if not t.strip())
        unique_tickers = set(raw_tickers)
        dup_count = raw_count - len(unique_tickers)
        malformed_tickers = [t for t in unique_tickers if not t.isalnum()]
        malformed_count = len(malformed_tickers)

        normalized_tickers = {
            t.strip().upper().replace(".", "").replace("-", "").replace("/", "")
            for t in unique_tickers
        }
        normalized_count = len(normalized_tickers)

        fmt, src_file = source_map.get(year, ("UNKNOWN", "unknown"))

        quality_rows.append(
            {
                "year": int(year),
                "raw_row_count": raw_count,
                "unique_ticker_count": len(unique_tickers),
                "duplicate_count": dup_count,
                "null_empty_count": empty_count,
                "malformed_ticker_count": malformed_count,
                "normalized_ticker_count": normalized_count,
                "source_format": fmt,
                "source_filename": src_file,
                "flag_unusual_size": bool(raw_count < 970 or raw_count > 1040),
            }
        )

        yearly_constituent_sets[year] = unique_tickers

        if empty_count > 0:
            registry.register(
                category="MEMBERSHIP",
                severity="HIGH",
                year=int(year),
                description=f"Year {year} has {empty_count} null or empty ticker records.",
                evidence=f"Empty ticker count = {empty_count}",
                status="FLAGGED",
                notes="Check CSV parser and source file.",
            )

        if raw_count < 975:
            registry.register(
                category="MEMBERSHIP",
                severity="MEDIUM",
                year=int(year),
                description=f"Year {year} constituent count ({raw_count}) is unusually low (< 975).",
                evidence=f"Raw rows = {raw_count}, Source = {src_file}",
                status="INVESTIGATE",
                notes="Possible constituent extraction truncation in source PDF/file.",
            )

        if raw_count > 1030:
            registry.register(
                category="MEMBERSHIP",
                severity="MEDIUM",
                year=int(year),
                description=f"Year {year} constituent count ({raw_count}) is unusually high (> 1030).",
                evidence=f"Raw rows = {raw_count}, Source = {src_file}",
                status="INVESTIGATE",
                notes="Possible dual share classes or intra-year ETF holdings additions.",
            )

    annual_quality_df = pl.DataFrame(quality_rows).sort("year")

    # Record any missing years across the configured evaluation range
    start_year = int(config.get("years", {}).get("start_year", 2000))
    end_year = int(config.get("years", {}).get("end_year", 2026))
    for expected_year in range(start_year, end_year + 1):
        if expected_year not in years:
            registry.register(
                category="MEMBERSHIP",
                severity="HIGH",
                year=expected_year,
                description=f"Year {expected_year} Russell 1000 constituent list is completely missing from dataset.",
                evidence=f"{expected_year} file absent in data/processed and data/raw.",
                status="FLAGGED",
                notes="Year-over-year transitions across this year will be discontinuous.",
            )

    # 2. Year-over-Year Transition Matrix
    transition_rows = []
    sorted_years = sorted(yearly_constituent_sets.keys())

    for i in range(len(sorted_years) - 1):
        y_from = sorted_years[i]
        y_to = sorted_years[i + 1]
        elapsed = y_to - y_from
        s_from = yearly_constituent_sets[y_from]
        s_to = yearly_constituent_sets[y_to]

        entries = s_to - s_from
        exits = s_from - s_to
        intersection = s_from & s_to
        union = s_from | s_to

        jaccard = len(intersection) / len(union) if union else 0.0
        churn_rate = (len(entries) + len(exits)) / len(union) if union else 0.0

        is_multi_year = elapsed > 1
        transition_rows.append(
            {
                "from_year": int(y_from),
                "to_year": int(y_to),
                "elapsed_years": int(elapsed),
                "is_multi_year_jump": is_multi_year,
                "u_prev_size": len(s_from),
                "u_curr_size": len(s_to),
                "entries_count": len(entries),
                "exits_count": len(exits),
                "intersection_count": len(intersection),
                "union_count": len(union),
                "jaccard_similarity": round(jaccard, 5),
                "churn_rate": round(churn_rate, 5),
            }
        )

        if is_multi_year:
            registry.register(
                category="MEMBERSHIP",
                severity="MEDIUM",
                year=int(y_from),
                description=f"Multi-year gap in constituent transitions: {y_from} -> {y_to} ({elapsed} years).",
                evidence=f"Elapsed years = {elapsed}, Churn = {churn_rate:.2%}",
                status="FLAGGED",
                notes="Expected turnover is ~2x normal single-year turnover (~20-25%).",
            )

        if not is_multi_year and churn_rate > 0.20:
            registry.register(
                category="MEMBERSHIP",
                severity="MEDIUM",
                year=int(y_to),
                description=f"High constituent churn ({churn_rate:.1%}) between {y_from} and {y_to}.",
                evidence=f"Entries: {len(entries)}, Exits: {len(exits)}, Churn: {churn_rate:.3f}",
                status="INVESTIGATE",
                notes="Normal index rebalance churn is 8-15%. Check for market disruption or methodology shift.",
            )

    transitions_df = pl.DataFrame(transition_rows).sort(["from_year", "to_year"])
    return annual_quality_df, transitions_df


def audit_ticker_formats_and_duplicates(
    config: dict[str, Any],
    registry: ExceptionRegistry,
) -> pl.DataFrame:
    """Audit duplicate tickers, formatting anomalies, and dual share class representations."""
    logger.info("Auditing ticker formatting, duplicates, and symbol anomalies...")
    paths = config.get("paths", {})
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    csv_file = processed_dir / "russell1000_all_years.csv"

    df_raw = pd.read_csv(csv_file, keep_default_na=False)
    anomaly_records = []

    # 1. Intra-year duplicate check
    for year, group in df_raw.groupby("Year"):
        tickers = group["Ticker"].tolist()
        seen = set()
        duplicates = set()
        for t in tickers:
            if t in seen:
                duplicates.add(t)
            seen.add(t)

        for d in duplicates:
            count = tickers.count(d)
            anomaly_records.append(
                {
                    "year": int(year),
                    "raw_ticker": d,
                    "normalized_ticker": d.replace(".", "").replace("-", "").replace("/", ""),
                    "anomaly_type": "EXACT_DUPLICATE",
                    "classification": "formatting_duplicate",
                    "severity": "HIGH",
                    "description": f"Ticker '{d}' appears {count} times in year {year} constituent list.",
                    "action_recommendation": "Deduplicate constituent rows preserving first valid record.",
                }
            )
            registry.register(
                category="DUPLICATE",
                severity="HIGH",
                year=int(year),
                ticker=d,
                description=f"Duplicate constituent '{d}' in year {year} ({count} occurrences).",
                evidence=f"Count: {count}",
                status="FLAGGED",
                resolution="Deduplicate during PiT universe construction.",
            )

    # 2. Punctuation and share-class syntax analysis
    unique_holdings = df_raw[["Ticker", "Name", "Year"]].drop_duplicates()
    for _, row in unique_holdings.iterrows():
        t = row["Ticker"]
        name = row["Name"]
        yr = int(row["Year"])

        has_dot = "." in t
        has_dash = "-" in t
        has_slash = "/" in t
        has_space = " " in t

        if has_dot or has_dash or has_slash or has_space:
            # Distinguish dual-class (e.g., BRK.B, BF.B, JW.A) from formatting errors
            is_likely_class = False
            for sep in [".", "-", "/"]:
                if sep in t:
                    parts = t.split(sep)
                    if len(parts) == 2 and len(parts[1]) in (1, 2) and parts[1].isalpha():
                        is_likely_class = True
                        break

            classification = "dual_share_class" if is_likely_class else "formatting_artifact"
            severity = "LOW" if is_likely_class else "MEDIUM"

            anomaly_records.append(
                {
                    "year": yr,
                    "raw_ticker": t,
                    "normalized_ticker": t.replace(".", "")
                    .replace("-", "")
                    .replace("/", "")
                    .strip(),
                    "anomaly_type": "PUNCTUATION_SYNTAX",
                    "classification": classification,
                    "severity": severity,
                    "description": f"Ticker '{t}' contains special characters ({name}). Class: {classification}.",
                    "action_recommendation": (
                        "Map to canonical CRSP format (e.g., 'BRK.B' -> 'BRKB') for price joins."
                        if is_likely_class
                        else "Clean punctuation and verify against exchange master."
                    ),
                }
            )

            registry.register(
                category="TICKER",
                severity=severity,
                year=yr,
                ticker=t,
                description=f"Ticker symbol '{t}' has non-standard syntax ({classification}).",
                evidence=f"Ticker: {t}, Company: {name}",
                status="FLAGGED",
                resolution="Harmonize ticker syntax across CRSP and crawled data.",
            )

    df_anomalies = (
        pl.DataFrame(anomaly_records).sort(["year", "raw_ticker"])
        if anomaly_records
        else pl.DataFrame(
            schema={
                "year": pl.Int32,
                "raw_ticker": pl.String,
                "normalized_ticker": pl.String,
                "anomaly_type": pl.String,
                "classification": pl.String,
                "severity": pl.String,
                "description": pl.String,
                "action_recommendation": pl.String,
            }
        )
    )

    logger.info("Identified %d ticker anomalies and duplicates.", len(df_anomalies))
    return df_anomalies
