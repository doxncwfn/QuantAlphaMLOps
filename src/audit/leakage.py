"""Look-ahead, survivorship, selection, and ticker leakage risk audit module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from src.audit.registry import ExceptionRegistry

logger = logging.getLogger(__name__)


def audit_leakage_and_survivorship_risks(
    config: dict[str, Any],
    registry: ExceptionRegistry,
) -> pl.DataFrame:
    """
    Audit potential survivorship bias, look-ahead bias, selection leakage, and ticker leakage mechanisms.

    Returns:
        potential_leakage_df
    """
    logger.info("Auditing look-ahead, survivorship, selection, and ticker leakage risks...")
    paths = config.get("paths", {})
    lookback_days = int(config.get("parameters", {}).get("lookback_days", 40))

    leakage_cases = [
        {
            "leakage_category": "SURVIVORSHIP_BIAS",
            "risk_mechanism": "Excluding delisted / bankrupt / acquired securities from historical panel.",
            "affected_scope": "All historical constituent lists (2000-2024), especially ~1,200 delisted securities in WRDS.",
            "severity": "CRITICAL",
            "evidence": "WRDS data contains extensive DLSTCD records (codes 200-580). If any filter removes incomplete price series, only surviving firms remain.",
            "resolution_requirement": "PiT universe must retain every constituent up to its actual delisting/cease date. Delisting returns (DLRET) must be integrated.",
        },
        {
            "leakage_category": "LOOK_AHEAD_BIAS",
            "risk_mechanism": "Using annual constituent list published in late June (or April/May ETF snapshots) retroactively from January 1.",
            "affected_scope": "Reconstitution timing across all 26 years.",
            "severity": "CRITICAL",
            "evidence": "Russell reconstitution occurs the last Friday of June. If an annual list is applied to January-June of the same calendar year, it leaks future index additions.",
            "resolution_requirement": "The reconstitution cycle runs from late June (t) to late June (t+1). Never apply Year t constituents to prior months (Jan-June of Year t).",
        },
        {
            "leakage_category": "LOOK_AHEAD_BIAS_ETF_SNAPSHOT",
            "risk_mechanism": "Intra-year ETF holdings snapshots (e.g. 2015-04-29, 2016-05-13, 2021-01-13) used as whole-year membership proxies.",
            "affected_scope": "Years 2015, 2016, 2020, 2021, 2023, 2025, 2026.",
            "severity": "HIGH",
            "evidence": "data/raw/2015.csv contains firms delisted in May/June 2015. 2021.xls dated January 2021 reflects corporate actions of late 2020.",
            "resolution_requirement": "Explicitly document snapshot timestamps. When building daily eligibility, verify security was actually active on trading date d.",
        },
        {
            "leakage_category": "SELECTION_LEAKAGE",
            "risk_mechanism": "Filtering universe by requiring complete, uninterrupted history over entire training horizon (e.g. strict intersection).",
            "affected_scope": "Model dataset construction (e.g. earlier residualizer requiring max_valid_days).",
            "severity": "CRITICAL",
            "evidence": "Filtering to stocks with 100% complete days creates perfect foresight bias, conditioning on future survival.",
            "resolution_requirement": f"Eligibility must be evaluated dynamically at each date t using only trailing information (t - {lookback_days} to t). Never condition on t > 0 survival.",
        },
        {
            "leakage_category": "TICKER_LEAKAGE",
            "risk_mechanism": "Using a modern / terminal ticker symbol to query historical prices without resolving ticker changes or ticker reuse.",
            "affected_scope": "At least 20 documented recycled tickers and >150 ticker rebranding events.",
            "severity": "CRITICAL",
            "evidence": "Multiple companies shared tickers across different decades. Modern query for 'C' returns Citigroup, missing Chrysler in earlier eras.",
            "resolution_requirement": "Link all securities via permanent numerical IDs (PERMNO or CUSIP) before joining price series.",
        },
    ]

    for lc in leakage_cases:
        registry.register(
            category="LEAKAGE",
            severity=lc["severity"],
            description=f"Risk: {lc['risk_mechanism']}",
            evidence=lc["evidence"],
            status="FLAGGED",
            resolution=lc["resolution_requirement"],
            notes=f"Category: {lc['leakage_category']}. Scope: {lc['affected_scope']}",
        )

    df_leakage = pl.DataFrame(leakage_cases).sort(["severity", "leakage_category"])
    out_dir = Path(paths.get("output_tables_dir", "report/quality/tables"))
    out_dir.mkdir(parents=True, exist_ok=True)
    df_leakage.write_parquet(out_dir / "potential_leakage_cases.parquet")
    df_leakage.write_csv(out_dir / "potential_leakage_cases.csv")

    logger.info("Saved %d potential leakage mechanisms to registry.", len(df_leakage))
    return df_leakage
