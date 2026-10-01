"""Price, OHLC bounds, returns, corporate action, and identity integrity audit module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from src.audit.registry import ExceptionRegistry

logger = logging.getLogger(__name__)


def audit_price_ohlc_integrity(
    config: dict[str, Any],
    registry: ExceptionRegistry,
) -> pl.DataFrame:
    """Audit OHLC mathematical consistency, negative quotes, zero prices, and volume anomalies."""
    logger.info("Auditing price and OHLC integrity across all years...")
    paths = config.get("paths", {})
    wrds_dir = Path(paths.get("wrds_dir", "data/WRDS"))

    issues = []

    for yr in range(2000, 2027):
        p = wrds_dir / f"{yr}.parquet"
        if not p.exists():
            continue

        try:
            df = pl.read_parquet(p)
            cols = df.columns

            if "PRC" in cols:
                # WRDS CRSP format (2000-2024)
                date_col = "date"
                ticker_col = "TICKER"
                permno_col = "PERMNO"

                # Check duplicate (PERMNO, date)
                dups = (
                    df.group_by([permno_col, date_col])
                    .agg(pl.len().alias("count"))
                    .filter(pl.col("count") > 1)
                    .sort([permno_col, date_col])
                )
                dup_count = len(dups)
                if dup_count > 0:
                    sample_dup = dups.head(2).to_dicts()
                    issues.append(
                        {
                            "year": yr,
                            "check_name": "DUPLICATE_PERMNO_DATE_ROWS",
                            "severity": "CRITICAL",
                            "affected_count": dup_count,
                            "description": f"Found {dup_count} duplicate (PERMNO, date) pairs in {yr}.",
                            "evidence": f"Sample duplicates: {sample_dup}",
                        }
                    )
                    registry.register(
                        category="PRICE",
                        severity="CRITICAL",
                        year=yr,
                        description=f"Duplicate primary key (PERMNO, date) detected ({dup_count} instances) in {yr}.",
                        evidence=str(sample_dup),
                        status="FLAGGED",
                        notes="Requires deduplication prior to time series modeling.",
                    )

                # Check negative prices (CRSP bid/ask midpoint convention)
                neg_prc = df.filter(pl.col("PRC") < 0)
                neg_count = len(neg_prc)
                if neg_count > 0:
                    pct = neg_count / len(df) * 100
                    issues.append(
                        {
                            "year": yr,
                            "check_name": "CRSP_NEGATIVE_PRICE_QUOTES",
                            "severity": "INFO",
                            "affected_count": neg_count,
                            "description": f"{neg_count} records ({pct:.2f}%) have negative PRC (CRSP convention: bid/ask midpoint average due to no trade).",
                            "evidence": f"Negative PRC count: {neg_count} / {len(df)}",
                        }
                    )

                # Check true zero prices (no price and no quote available)
                zero_prc = df.filter(pl.col("PRC") == 0)
                zero_count = len(zero_prc)
                if zero_count > 0:
                    issues.append(
                        {
                            "year": yr,
                            "check_name": "ZERO_PRICE_OBSERVATIONS",
                            "severity": "HIGH",
                            "affected_count": zero_count,
                            "description": f"{zero_count} observations have PRC == 0.0 (neither trade price nor quote available).",
                            "evidence": f"Zero PRC count: {zero_count}",
                        }
                    )
                    registry.register(
                        category="PRICE",
                        severity="HIGH",
                        year=yr,
                        description=f"Zero price observations ({zero_count} rows) in {yr} WRDS data.",
                        evidence=f"Zero PRC count = {zero_count}",
                        status="FLAGGED",
                        notes="Zero price indicates total absence of market quotes; must not be used as valid price denominator.",
                    )

                # Check negative volume
                neg_vol = df.filter(pl.col("VOL") < 0)
                neg_vol_count = len(neg_vol)
                if neg_vol_count > 0:
                    issues.append(
                        {
                            "year": yr,
                            "check_name": "NEGATIVE_VOLUME",
                            "severity": "CRITICAL",
                            "affected_count": neg_vol_count,
                            "description": f"{neg_vol_count} records have negative trading volume (VOL < 0).",
                            "evidence": f"VOL < 0 count: {neg_vol_count}",
                        }
                    )
                    registry.register(
                        category="PRICE",
                        severity="CRITICAL",
                        year=yr,
                        description=f"Negative volume detected ({neg_vol_count} rows) in {yr}.",
                        evidence=f"Negative volume count = {neg_vol_count}",
                        status="FLAGGED",
                    )

                # Check zero volume
                zero_vol = df.filter(pl.col("VOL") == 0)
                zero_vol_count = len(zero_vol)
                if zero_vol_count > 0:
                    pct_vol0 = zero_vol_count / len(df) * 100
                    issues.append(
                        {
                            "year": yr,
                            "check_name": "ZERO_VOLUME_OBSERVATIONS",
                            "severity": "LOW",
                            "affected_count": zero_vol_count,
                            "description": f"{zero_vol_count} records ({pct_vol0:.2f}%) have VOL == 0 (non-trading days or quote-only days).",
                            "evidence": f"VOL == 0 count: {zero_vol_count}",
                        }
                    )

                # Check OHLC bounds: High >= max(Open, Close), Low <= min(Open, Close), High >= Low
                valid_ohlc = df.filter(
                    pl.col("PRC").is_not_null()
                    & pl.col("OPENPRC").is_not_null()
                    & pl.col("ASKHI").is_not_null()
                    & pl.col("BIDLO").is_not_null()
                    & (pl.col("OPENPRC") != 0)
                    & (pl.col("PRC") != 0)
                )

                if len(valid_ohlc) > 0:
                    ohlc_check = valid_ohlc.with_columns(
                        [
                            pl.col("PRC").abs().alias("p_abs"),
                            pl.col("OPENPRC").abs().alias("open_abs"),
                            pl.col("ASKHI").abs().alias("high_abs"),
                            pl.col("BIDLO").abs().alias("low_abs"),
                        ]
                    ).with_columns(
                        [
                            (
                                pl.col("high_abs") < pl.max_horizontal("open_abs", "p_abs") - 1e-4
                            ).alias("high_violation"),
                            (
                                pl.col("low_abs") > pl.min_horizontal("open_abs", "p_abs") + 1e-4
                            ).alias("low_violation"),
                            (pl.col("high_abs") < pl.col("low_abs")).alias("spread_violation"),
                        ]
                    )

                    high_viols = ohlc_check.filter(pl.col("high_violation"))
                    low_viols = ohlc_check.filter(pl.col("low_violation"))
                    spread_viols = ohlc_check.filter(pl.col("spread_violation"))

                    if len(high_viols) > 0:
                        sample_high = (
                            high_viols.sort([permno_col, date_col])
                            .select([permno_col, date_col, "high_abs", "open_abs", "p_abs"])
                            .head(2)
                            .to_dicts()
                        )
                        issues.append(
                            {
                                "year": yr,
                                "check_name": "HIGH_LESS_THAN_MAX_OPEN_CLOSE",
                                "severity": "HIGH",
                                "affected_count": len(high_viols),
                                "description": f"{len(high_viols)} rows where High < max(Open, Close).",
                                "evidence": f"Sample: {sample_high}",
                            }
                        )
                        registry.register(
                            category="PRICE",
                            severity="HIGH",
                            year=yr,
                            description=f"OHLC violation: High price lower than max(Open, Close) in {len(high_viols)} rows in {yr}.",
                            evidence=f"Violations: {len(high_viols)}",
                            status="INVESTIGATE",
                            notes="Examine CRSP ask quotes versus executed trade prices.",
                        )

                    if len(low_viols) > 0:
                        sample_low = (
                            low_viols.sort([permno_col, date_col])
                            .select([permno_col, date_col, "low_abs", "open_abs", "p_abs"])
                            .head(2)
                            .to_dicts()
                        )
                        issues.append(
                            {
                                "year": yr,
                                "check_name": "LOW_GREATER_THAN_MIN_OPEN_CLOSE",
                                "severity": "HIGH",
                                "affected_count": len(low_viols),
                                "description": f"{len(low_viols)} rows where Low > min(Open, Close).",
                                "evidence": f"Sample: {sample_low}",
                            }
                        )
                        registry.register(
                            category="PRICE",
                            severity="HIGH",
                            year=yr,
                            description=f"OHLC violation: Low price higher than min(Open, Close) in {len(low_viols)} rows in {yr}.",
                            evidence=f"Violations: {len(low_viols)}",
                            status="INVESTIGATE",
                            notes="Examine CRSP bid quotes versus executed trade prices.",
                        )

                    if len(spread_viols) > 0:
                        sample_spread = (
                            spread_viols.sort([permno_col, date_col])
                            .select([permno_col, date_col, "high_abs", "low_abs"])
                            .head(2)
                            .to_dicts()
                        )
                        issues.append(
                            {
                                "year": yr,
                                "check_name": "HIGH_LESS_THAN_LOW",
                                "severity": "CRITICAL",
                                "affected_count": len(spread_viols),
                                "description": f"{len(spread_viols)} rows where High < Low.",
                                "evidence": f"Sample: {sample_spread}",
                            }
                        )
                        registry.register(
                            category="PRICE",
                            severity="CRITICAL",
                            year=yr,
                            description=f"OHLC violation: High < Low in {len(spread_viols)} rows in {yr}.",
                            evidence=f"Violations: {len(spread_viols)}",
                            status="FLAGGED",
                        )

            elif "Close" in cols:
                # Crawled continuation format (2025-2026)
                date_col = "Date"
                ticker_col = "Ticker"

                # Check duplicate (Ticker, Date)
                dups = (
                    df.group_by([ticker_col, date_col])
                    .agg(pl.len().alias("count"))
                    .filter(pl.col("count") > 1)
                )
                if len(dups) > 0:
                    issues.append(
                        {
                            "year": yr,
                            "check_name": "DUPLICATE_TICKER_DATE_ROWS",
                            "severity": "CRITICAL",
                            "affected_count": len(dups),
                            "description": f"Found {len(dups)} duplicate (Ticker, Date) pairs in crawled {yr}.",
                            "evidence": f"Sample: {dups.head(2).to_dicts()}",
                        }
                    )
                    registry.register(
                        category="PRICE",
                        severity="CRITICAL",
                        year=yr,
                        description=f"Duplicate primary key (Ticker, Date) in crawled {yr} ({len(dups)} instances).",
                        evidence=str(dups.head(2).to_dicts()),
                        status="FLAGGED",
                    )

                # Check zero or negative Close
                bad_close = df.filter(pl.col("Close") <= 0)
                if len(bad_close) > 0:
                    issues.append(
                        {
                            "year": yr,
                            "check_name": "ZERO_OR_NEGATIVE_CLOSE",
                            "severity": "HIGH",
                            "affected_count": len(bad_close),
                            "description": f"{len(bad_close)} observations have Close <= 0 in {yr}.",
                            "evidence": f"Count: {len(bad_close)}",
                        }
                    )

                # Check OHLC bounds
                ohlc_check = df.filter(
                    pl.col("High").is_not_null()
                    & pl.col("Low").is_not_null()
                    & pl.col("Open").is_not_null()
                    & pl.col("Close").is_not_null()
                ).with_columns(
                    [
                        (pl.col("High") < pl.max_horizontal("Open", "Close") - 1e-4).alias(
                            "high_violation"
                        ),
                        (pl.col("Low") > pl.min_horizontal("Open", "Close") + 1e-4).alias(
                            "low_violation"
                        ),
                        (pl.col("High") < pl.col("Low")).alias("spread_violation"),
                    ]
                )

                h_viols = len(ohlc_check.filter(pl.col("high_violation")))
                l_viols = len(ohlc_check.filter(pl.col("low_violation")))
                s_viols = len(ohlc_check.filter(pl.col("spread_violation")))

                if h_viols > 0 or l_viols > 0 or s_viols > 0:
                    issues.append(
                        {
                            "year": yr,
                            "check_name": "CRAWLED_OHLC_BOUND_VIOLATION",
                            "severity": "MEDIUM",
                            "affected_count": h_viols + l_viols + s_viols,
                            "description": f"Crawled OHLC bound violations in {yr}: High<{h_viols}, Low>{l_viols}, High<Low:{s_viols}.",
                            "evidence": f"High: {h_viols}, Low: {l_viols}, Spread: {s_viols}",
                        }
                    )

        except (OSError, pl.exceptions.PolarsError, KeyError, ValueError) as e:
            logger.error("Error auditing OHLC integrity for year %d: %s", yr, e)

    df_issues = (
        pl.DataFrame(issues).sort(["year", "severity"])
        if issues
        else pl.DataFrame(
            schema={
                "year": pl.Int32,
                "check_name": pl.String,
                "severity": pl.String,
                "affected_count": pl.Int64,
                "description": pl.String,
                "evidence": pl.String,
            }
        )
    )

    out_dir = Path(paths.get("output_tables_dir", "report/quality/tables"))
    out_dir.mkdir(parents=True, exist_ok=True)
    df_issues.write_parquet(out_dir / "ohlc_integrity_issues.parquet")
    df_issues.write_csv(out_dir / "ohlc_integrity_issues.csv")

    logger.info("Saved OHLC integrity audit table with %d issue records.", len(df_issues))
    return df_issues


def audit_returns_and_corporate_actions(
    config: dict[str, Any],
    registry: ExceptionRegistry,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Audit extreme price movements, stock splits, distributions, and CRSP adjustment factors.

    Returns:
        (extreme_returns_df, corporate_actions_df)
    """
    logger.info("Auditing return sanity, extreme moves, and corporate actions...")
    paths = config.get("paths", {})
    wrds_dir = Path(paths.get("wrds_dir", "data/WRDS"))

    extreme_returns_list = []
    corp_action_list = []

    for yr in range(2000, 2027):
        p = wrds_dir / f"{yr}.parquet"
        if not p.exists():
            continue

        try:
            df = pl.read_parquet(p)
            cols = df.columns

            if "PRC" in cols:
                # WRDS CRSP format (2000-2024)
                cols_to_select = [
                    "PERMNO",
                    "date",
                    "TICKER",
                    "COMNAM",
                    "PRC",
                    "CFACPR",
                    "FACPR",
                    "DISTCD",
                    "DIVAMT",
                ]
                if "RET" in cols:
                    cols_to_select.append("RET")
                if "DLRET" in cols:
                    cols_to_select.append("DLRET")

                df_sub = df.select(cols_to_select).sort(["PERMNO", "date"])

                df_calc = df_sub.with_columns(
                    [
                        pl.col("PRC").abs().alias("p_abs"),
                        pl.col("CFACPR").alias("cfacpr"),
                    ]
                ).with_columns(
                    [
                        pl.col("p_abs").shift(1).over("PERMNO").alias("prev_p_abs"),
                        pl.col("cfacpr").shift(1).over("PERMNO").alias("prev_cfacpr"),
                        pl.col("date").shift(1).over("PERMNO").alias("prev_date"),
                    ]
                )

                df_calc = df_calc.filter(
                    pl.col("prev_p_abs").is_not_null()
                    & (pl.col("prev_p_abs") > 0)
                    & (pl.col("p_abs") > 0)
                    & (pl.col("cfacpr") > 0)
                    & (pl.col("prev_cfacpr") > 0)
                ).with_columns(
                    [
                        ((pl.col("p_abs") / pl.col("prev_p_abs")) - 1.0).alias("raw_ret"),
                        (
                            (
                                (pl.col("p_abs") / pl.col("cfacpr"))
                                / (pl.col("prev_p_abs") / pl.col("prev_cfacpr"))
                            )
                            - 1.0
                        ).alias("adj_ret"),
                        (pl.col("cfacpr") != pl.col("prev_cfacpr")).alias("is_split_date"),
                    ]
                )

                # Scan extreme raw returns (|raw_ret| > 0.50)
                extremes = df_calc.filter(pl.col("raw_ret").abs() > 0.50)
                for r in extremes.iter_rows(named=True):
                    is_split = r["is_split_date"]
                    raw_r = r["raw_ret"]
                    adj_r = r["adj_ret"]
                    t = r["TICKER"]
                    d = r["date"]
                    pno = r["PERMNO"]

                    if is_split and abs(adj_r) < 0.25:
                        reason = "STOCK_SPLIT_OR_CONSOLIDATION"
                    elif is_split:
                        reason = "SPLIT_PLUS_PRICE_SHOCK"
                    elif abs(adj_r) > 1.0:
                        reason = "EXTREME_VOLATILITY_OR_DATA_ERROR"
                    else:
                        reason = "CORPORATE_ACTION_UNADJUSTED"

                    extreme_returns_list.append(
                        {
                            "year": yr,
                            "date": d,
                            "ticker": t,
                            "security_id": pno,
                            "raw_return": round(raw_r, 4),
                            "split_adjusted_return": round(adj_r, 4),
                            "cfacpr_today": r["cfacpr"],
                            "cfacpr_prev": r["prev_cfacpr"],
                            "split_flag": is_split,
                            "classification": reason,
                        }
                    )

                    if reason == "EXTREME_VOLATILITY_OR_DATA_ERROR":
                        registry.register(
                            category="PRICE",
                            severity="HIGH",
                            year=yr,
                            ticker=t,
                            security_id=pno,
                            date_start=d,
                            description=f"Extreme daily return ({raw_r:.1%}) for {t} on {d} not resolved by CFACPR factor (adj ret: {adj_r:.1%}).",
                            evidence=f"Raw P: {r['prev_p_abs']} -> {r['p_abs']}, CFACPR: {r['prev_cfacpr']} -> {r['cfacpr']}",
                            status="INVESTIGATE",
                            notes="Examine CRSP distribution codes, news, or potential data transcription error.",
                        )

                # Scan corporate action distributions
                actions = df_calc.filter(
                    (pl.col("is_split_date")) | (pl.col("DISTCD").is_not_null())
                )
                if len(actions) > 0:
                    summary_actions = (
                        actions.group_by("DISTCD")
                        .agg(
                            [
                                pl.len().alias("count"),
                                pl.col("DIVAMT").drop_nulls().mean().alias("avg_dividend"),
                            ]
                        )
                        .sort("DISTCD")
                        .with_columns(pl.lit(yr).alias("year"))
                    )
                    corp_action_list.append(summary_actions)

            elif "Close" in cols:
                # Crawled continuation format (2025-2026)
                df_calc = (
                    df.sort(["Ticker", "Date"])
                    .with_columns(
                        [
                            pl.col("Close").shift(1).over("Ticker").alias("prev_close"),
                            pl.col("Adj_Close").shift(1).over("Ticker").alias("prev_adj_close"),
                        ]
                    )
                    .filter(
                        pl.col("prev_close").is_not_null()
                        & (pl.col("prev_close") > 0)
                        & (pl.col("Close") > 0)
                    )
                    .with_columns(
                        [
                            ((pl.col("Close") / pl.col("prev_close")) - 1.0).alias("raw_ret"),
                            ((pl.col("Adj_Close") / pl.col("prev_adj_close")) - 1.0).alias(
                                "adj_ret"
                            ),
                        ]
                    )
                )

                extremes = df_calc.filter(pl.col("raw_ret").abs() > 0.50)
                for r in extremes.iter_rows(named=True):
                    d_str = str(r["Date"])[:10]
                    t = r["Ticker"]
                    raw_r = r["raw_ret"]
                    adj_r = r["adj_ret"]

                    extreme_returns_list.append(
                        {
                            "year": yr,
                            "date": d_str,
                            "ticker": t,
                            "security_id": None,
                            "raw_return": round(raw_r, 4),
                            "split_adjusted_return": round(adj_r, 4),
                            "cfacpr_today": 1.0,
                            "cfacpr_prev": 1.0,
                            "split_flag": False,
                            "classification": (
                                "EXTREME_VOLATILITY_OR_DATA_ERROR"
                                if abs(adj_r) > 0.50
                                else "SPLIT_ADJUSTED"
                            ),
                        }
                    )

        except (OSError, pl.exceptions.PolarsError, KeyError, ValueError) as e:
            logger.error("Error auditing returns for year %d: %s", yr, e)

    df_extremes = (
        pl.DataFrame(extreme_returns_list).sort(["year", "date", "ticker", "raw_return"])
        if extreme_returns_list
        else pl.DataFrame()
    )

    df_corps = (
        pl.concat(corp_action_list).sort(["year", "DISTCD"]) if corp_action_list else pl.DataFrame()
    )

    out_dir = Path(paths.get("output_tables_dir", "report/quality/tables"))
    out_dir.mkdir(parents=True, exist_ok=True)
    df_extremes.write_parquet(out_dir / "extreme_returns.parquet")
    df_extremes.write_csv(out_dir / "extreme_returns.csv")
    df_corps.write_parquet(out_dir / "corporate_action_flags.parquet")
    df_corps.write_csv(out_dir / "corporate_action_flags.csv")

    logger.info(
        "Saved extreme returns (%d records) and corporate actions (%d records).",
        len(df_extremes),
        len(df_corps),
    )
    return df_extremes, df_corps


def audit_identity_and_continuity(
    config: dict[str, Any],
    registry: ExceptionRegistry,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Audit security identity mapping between TICKER and PERMNO across WRDS data.

    Returns:
        (identity_anomalies_df, delisted_securities_df)
    """
    logger.info("Auditing historical identity, ticker continuity, and delistings...")
    paths = config.get("paths", {})
    wrds_dir = Path(paths.get("wrds_dir", "data/WRDS"))

    identity_dfs = []
    delist_records = []

    for yr in range(2000, 2025):
        p = wrds_dir / f"{yr}.parquet"
        if not p.exists():
            continue

        try:
            df = (
                pl.scan_parquet(p)
                .select(
                    [
                        "PERMNO",
                        "date",
                        "TICKER",
                        "COMNAM",
                        "CUSIP",
                        "NCUSIP",
                        "PERMCO",
                        "SHRCD",
                        "EXCHCD",
                        "DLSTCD",
                        "DLPRC",
                        "DLRET",
                        "NWPERM",
                    ]
                )
                .with_columns(
                    [
                        pl.col("PERMNO").cast(pl.Int64),
                        pl.col("date").cast(pl.String),
                        pl.col("TICKER").cast(pl.String),
                        pl.col("COMNAM").cast(pl.String),
                        pl.col("CUSIP").cast(pl.String),
                        pl.col("NCUSIP").cast(pl.String),
                        pl.col("PERMCO").cast(pl.Float64),
                        pl.col("SHRCD").cast(pl.Float64),
                        pl.col("EXCHCD").cast(pl.Float64),
                        pl.col("DLSTCD").cast(pl.String),
                        pl.col("DLPRC").cast(pl.String),
                        pl.col("DLRET").cast(pl.String),
                        pl.col("NWPERM").cast(pl.String),
                    ]
                )
                .collect()
            )

            df_year = df.with_columns(pl.lit(yr).alias("year"))
            identity_dfs.append(df_year)

            delisted = df.filter(pl.col("DLSTCD").is_not_null())
            if len(delisted) > 0:
                delist_summary = (
                    delisted.group_by(["PERMNO", "TICKER", "COMNAM", "DLSTCD"])
                    .agg(
                        [
                            pl.col("date").max().alias("delist_date"),
                            pl.col("DLPRC").drop_nulls().first().alias("dl_price"),
                            pl.col("DLRET").drop_nulls().first().alias("dl_return"),
                            pl.col("NWPERM").drop_nulls().first().alias("nw_permno"),
                        ]
                    )
                    .with_columns(pl.lit(yr).alias("year"))
                )
                delist_records.append(delist_summary)
        except (OSError, pl.exceptions.PolarsError, KeyError, ValueError) as e:
            logger.warning("Error reading identity fields from %s: %s", p.name, e)

    master_identity = pl.concat(identity_dfs).sort(["year", "PERMNO", "TICKER"])

    # 1. Tickers mapped to multiple PERMNOs (Ticker Reuse)
    ticker_to_permnos = (
        master_identity.group_by("TICKER")
        .agg(
            [
                pl.col("PERMNO").n_unique().alias("permno_count"),
                pl.col("PERMNO").unique().sort().alias("permnos"),
                pl.col("COMNAM").drop_nulls().unique().sort().alias("companies"),
                pl.col("year").unique().sort().alias("years_active"),
            ]
        )
        .filter(pl.col("permno_count") > 1)
        .sort(["permno_count", "TICKER"], descending=[True, False])
    )

    anomalies = []

    for row in ticker_to_permnos.iter_rows(named=True):
        t = row["TICKER"]
        permnos = row["permnos"]
        companies = [str(c) for c in row["companies"] if c is not None]
        years = row["years_active"]
        anomalies.append(
            {
                "ticker": str(t),
                "security_id": str(permnos),
                "anomaly_type": "TICKER_REUSE_MULTIPLE_PERMNOS",
                "count": row["permno_count"],
                "years_active": str(years),
                "companies_involved": "; ".join(companies[:4]),
                "severity": "CRITICAL",
                "description": f"Ticker '{t}' was reused across {row['permno_count']} distinct permanent securities (PERMNOs: {permnos}).",
                "recommendation": "DO NOT query or merge historical panels by TICKER alone. Always use PERMNO or CUSIP.",
            }
        )
        registry.register(
            category="IDENTITY",
            severity="CRITICAL",
            ticker=str(t),
            security_id=str(permnos),
            description=f"Ticker symbol '{t}' reused across {row['permno_count']} distinct companies/PERMNOs ({permnos}).",
            evidence=f"Companies: {companies[:3]} across years {years}",
            status="FLAGGED",
            notes="Requires PERMNO-based identity resolution to prevent conflating unrelated entities.",
        )

    # 2. PERMNOs mapped to multiple TICKERs (Ticker Renaming)
    permno_to_tickers = (
        master_identity.group_by("PERMNO")
        .agg(
            [
                pl.col("TICKER").drop_nulls().n_unique().alias("ticker_count"),
                pl.col("TICKER").drop_nulls().unique().sort().alias("tickers"),
                pl.col("COMNAM").drop_nulls().unique().sort().alias("company_names"),
                pl.col("year").unique().sort().alias("years_active"),
            ]
        )
        .filter(pl.col("ticker_count") > 1)
        .sort(["ticker_count", "PERMNO"], descending=[True, False])
    )

    for row in permno_to_tickers.iter_rows(named=True):
        pno = row["PERMNO"]
        tickers = [str(t) for t in row["tickers"] if t is not None]
        names = [str(n) for n in row["company_names"] if n is not None]
        years = row["years_active"]
        if not tickers:
            continue
        anomalies.append(
            {
                "ticker": "; ".join(tickers[:4]),
                "security_id": str(pno),
                "anomaly_type": "PERMNO_TICKER_RENAMING",
                "count": row["ticker_count"],
                "years_active": str(years),
                "companies_involved": "; ".join(names[:4]),
                "severity": "HIGH",
                "description": f"PERMNO {pno} changed exchange ticker across {row['ticker_count']} symbols: {tickers}.",
                "recommendation": "Track security continuity via PERMNO. Map old and new tickers to same time series.",
            }
        )
        registry.register(
            category="IDENTITY",
            severity="HIGH",
            security_id=pno,
            ticker="/".join(tickers[:3]),
            description=f"PERMNO {pno} changed ticker symbol across {len(tickers)} identifiers: {tickers}.",
            evidence=f"Ticker progression: {tickers}, Names: {names[:2]}",
            status="FLAGGED",
            notes="Security continuity maintained via PERMNO; historical ticker queries will fail if using modern symbol.",
        )

    # 3. Share classification check (SHRCD)
    non_common_shares = (
        master_identity.filter(~pl.col("SHRCD").is_in([10, 11]))
        .select(["PERMNO", "TICKER", "COMNAM", "SHRCD", "year"])
        .unique(["PERMNO", "SHRCD"])
        .sort(["SHRCD", "PERMNO"])
    )

    for row in non_common_shares.iter_rows(named=True):
        shrcd = row["SHRCD"]
        t = row["TICKER"]
        pno = row["PERMNO"]
        name = row["COMNAM"]
        desc = (
            "Foreign/ADR"
            if shrcd == 12
            else ("REIT" if shrcd in (18, 48) else f"Special Share Code {shrcd}")
        )
        anomalies.append(
            {
                "ticker": t,
                "security_id": str(pno),
                "anomaly_type": "NON_STANDARD_SHARE_CODE",
                "count": 1,
                "years_active": str(row["year"]),
                "companies_involved": name,
                "severity": "MEDIUM",
                "description": f"Security {t} (PERMNO {pno}) has CRSP share code {shrcd} ({desc}).",
                "recommendation": "Evaluate if REITs, ADRs, or non-ordinary shares are eligible under project universe rules.",
            }
        )
        registry.register(
            category="IDENTITY",
            severity="MEDIUM",
            ticker=t,
            security_id=pno,
            description=f"Non-standard equity classification: SHRCD={shrcd} ({desc}) for {t} ({name}).",
            evidence=f"PERMNO: {pno}, SHRCD: {shrcd}",
            status="INVESTIGATE",
            notes="Determine whether index rules permit foreign shares, REITs, or units in ML panel.",
        )

    # 4. Delisted securities summary
    master_delist = (
        pl.concat(delist_records).sort(["year", "delist_date", "PERMNO", "TICKER"])
        if delist_records
        else pl.DataFrame()
    )

    for row in master_delist.iter_rows(named=True):
        registry.register(
            category="DELISTING",
            severity="INFO",
            year=row["year"],
            ticker=row["TICKER"],
            security_id=row["PERMNO"],
            date_end=row["delist_date"],
            description=f"Security {row['TICKER']} (PERMNO {row['PERMNO']}) delisted on {row['delist_date']} (DLSTCD={row['DLSTCD']}).",
            evidence=f"DLPRC={row['dl_price']}, DLRET={row['dl_return']}, NWPERM={row['nw_permno']}",
            status="RESOLVED",
            notes="Delisted security must be retained up to its delisting date to prevent survivorship bias.",
        )

    identity_anomalies_df = pl.DataFrame(anomalies).sort(
        ["anomaly_type", "severity", "ticker", "security_id"]
    )

    out_dir = Path(paths.get("output_tables_dir", "report/quality/tables"))
    out_dir.mkdir(parents=True, exist_ok=True)

    identity_anomalies_df.write_parquet(out_dir / "identity_anomalies.parquet")
    identity_anomalies_df.write_csv(out_dir / "identity_anomalies.csv")
    master_delist.write_parquet(out_dir / "delisted_securities.parquet")
    master_delist.write_csv(out_dir / "delisted_securities.csv")

    logger.info(
        "Saved %d identity anomalies and %d delisting records.",
        len(identity_anomalies_df),
        len(master_delist),
    )
    return identity_anomalies_df, master_delist
