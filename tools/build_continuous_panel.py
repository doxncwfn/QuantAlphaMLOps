#!/usr/bin/env python3
"""
build_continuous_panel.py

Constructs a single continuous per-stock panel and point-in-time universe mask
by unifying CRSP WRDS (2000-2024), Yahoo (2025-2026), and backfilling pre-index
history from US_history.parquet.

Outputs:
- data/processed/continuous_panel.parquet
- data/processed/universe_mask.parquet
- data/processed/panel_validation_metrics.json
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_continuous_panel")

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    wrds_dir = data_dir / "WRDS"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    us_hist_path = data_dir / "US_history.parquet"
    if not us_hist_path.exists():
        logger.error("US_history.parquet not found at %s", us_hist_path)
        sys.exit(1)

    t0 = time.time()
    logger.info("Step 1: Identifying Russell target securities across all sources...")

    # 1. Collect all PERMNOs and Tickers from annual WRDS CRSP (2000-2024)
    wrds_crsp_permnos = set()
    wrds_ticker_history = []
    for yr in range(2000, 2025):
        f = wrds_dir / f"{yr}.parquet"
        df = pl.read_parquet(f, columns=["PERMNO", "TICKER", "date"])
        for row in df.iter_rows():
            p, t, d = row[0], row[1], str(row[2])[:10]
            if p is not None:
                wrds_crsp_permnos.add(p)
                if t is not None:
                    wrds_ticker_history.append((t, p, d))

    logger.info("Found %d unique PERMNOs in WRDS CRSP (2000-2024)", len(wrds_crsp_permnos))

    # 2. Collect Yahoo tickers (2025-2026)
    yahoo_tickers = set()
    for yr in [2025, 2026]:
        f = wrds_dir / f"{yr}.parquet"
        df = pl.read_parquet(f, columns=["Ticker"])
        yahoo_tickers.update(df["Ticker"].drop_nulls().unique().to_list())
    logger.info("Found %d unique Tickers in Yahoo era (2025-2026)", len(yahoo_tickers))

    # 3. Collect Russell membership list tickers from csv
    uni_path = processed_dir / "russell1000_all_years.csv"
    uni_df = pl.read_csv(uni_path)
    uni_tickers = set(uni_df["Ticker"].drop_nulls().unique().to_list())
    logger.info("Found %d unique Tickers in %s", len(uni_tickers), uni_path.name)

    # Add Baxalta (15401) and Silver Standard (49509) from Russell historical membership
    extra_russell_permnos = {15401, 49509}
    all_target_permnos = wrds_crsp_permnos | extra_russell_permnos
    logger.info("Total target PERMNOs for continuous panel: %d", len(all_target_permnos))

    # Build Master Ticker -> PERMNO mapping from historical WRDS (latest date per ticker)
    # This avoids look-ahead bias and handles ticker changes
    ticker_hist_df = pl.DataFrame(
        wrds_ticker_history, schema=["TICKER", "PERMNO", "date"], orient="row"
    ).sort("date")
    last_mapping_df = ticker_hist_df.group_by("TICKER").last()
    master_ticker_to_permno = dict(
        zip(
            last_mapping_df["TICKER"].to_list(),
            last_mapping_df["PERMNO"].to_list(),
        )
    )
    # Add extra mappings
    master_ticker_to_permno["BXLT"] = 15401
    master_ticker_to_permno["SS"] = 49509

    logger.info("Step 2: Processing and normalizing annual WRDS files...")
    # CRSP normalization expressions matching v1.3 §10.3
    crsp_frames = []

    crsp_cols_needed = [
        "date", "TICKER", "PERMNO", "PRC", "OPENPRC", "ASKHI", "BIDLO",
        "VOL", "RET", "RETX", "CFACPR", "CFACSHR", "DLSTCD", "DLRET", "SHROUT"
    ]

    for yr in range(2000, 2025):
        f = wrds_dir / f"{yr}.parquet"
        df_raw = pl.read_parquet(f, columns=crsp_cols_needed)

        # Normalize
        cfacpr_clean = pl.when(pl.col("CFACPR") == 0).then(None).otherwise(pl.col("CFACPR"))

        norm = df_raw.select([
            pl.col("date").cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d").alias("date"),
            pl.col("TICKER").alias("ticker"),
            pl.col("PERMNO").cast(pl.Int64).alias("permno"),
            pl.col("PRC").abs().cast(pl.Float64).alias("close_raw"),
            pl.col("OPENPRC").abs().cast(pl.Float64).alias("open_raw"),
            pl.col("ASKHI").cast(pl.Float64).alias("high_raw"),
            pl.col("BIDLO").cast(pl.Float64).alias("low_raw"),
            pl.col("VOL").cast(pl.Int64).alias("volume"),
            (pl.col("PRC").abs() / cfacpr_clean).cast(pl.Float64).alias("split_adj_close"),
            (pl.col("OPENPRC").abs() / cfacpr_clean).cast(pl.Float64).alias("split_adj_open"),
            (pl.col("ASKHI") / cfacpr_clean).cast(pl.Float64).alias("split_adj_high"),
            (pl.col("BIDLO") / cfacpr_clean).cast(pl.Float64).alias("split_adj_low"),
            pl.col("RETX").cast(pl.Float64, strict=False).log1p().alias("ret_price"),
            pl.col("RET").cast(pl.Float64, strict=False).log1p().alias("ret_total"),
            pl.col("CFACPR").cast(pl.Float64).alias("cfacpr"),
            pl.col("CFACSHR").cast(pl.Float64).alias("cfacshr"),
            pl.lit(None, dtype=pl.Float64).alias("div_adj_close"),
            pl.col("DLSTCD").cast(pl.Int64, strict=False).alias("dlstcd"),
            pl.col("DLRET").cast(pl.Float64, strict=False).alias("dlret"),
            pl.col("SHROUT").cast(pl.Int64, strict=False).alias("shrout"),
            pl.lit("CRSP").alias("source"),
        ])
        crsp_frames.append(norm)

    wrds_crsp_all = pl.concat(crsp_frames)
    raw_wrds_crsp_len = len(wrds_crsp_all)
    logger.info("Loaded WRDS CRSP (2000-2024): %d raw rows", raw_wrds_crsp_len)

    # Deduplicate WRDS CRSP on (permno, date)
    wrds_crsp_dedup = wrds_crsp_all.unique(subset=["permno", "date"], keep="first")
    annual_overlap_count = raw_wrds_crsp_len - len(wrds_crsp_dedup)
    logger.info(
        "WRDS CRSP deduplication dropped %d overlapping boundary rows (retaining %d rows)",
        annual_overlap_count, len(wrds_crsp_dedup)
    )

    # Process Yahoo files (2025-2026)
    yahoo_frames = []
    for yr in [2025, 2026]:
        f = wrds_dir / f"{yr}.parquet"
        df_raw = pl.read_parquet(f)

        # Base normalization matching v1.3 §10.3
        norm = df_raw.select([
            pl.col("Date").cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d").alias("date"),
            pl.col("Ticker").alias("ticker"),
            pl.col("Ticker").replace_strict(master_ticker_to_permno, default=None).cast(pl.Int64).alias("permno"),
            pl.col("Close").cast(pl.Float64).alias("close_raw"),
            pl.col("Open").cast(pl.Float64).alias("open_raw"),
            pl.col("High").cast(pl.Float64).alias("high_raw"),
            pl.col("Low").cast(pl.Float64).alias("low_raw"),
            pl.col("Volume").cast(pl.Int64).alias("volume"),
            pl.col("Close").cast(pl.Float64).alias("split_adj_close"),
            pl.col("Open").cast(pl.Float64).alias("split_adj_open"),
            pl.col("High").cast(pl.Float64).alias("split_adj_high"),
            pl.col("Low").cast(pl.Float64).alias("split_adj_low"),
            pl.lit(None, dtype=pl.Float64).alias("ret_price"),
            pl.lit(None, dtype=pl.Float64).alias("ret_total"),
            pl.lit(1.0, dtype=pl.Float64).alias("cfacpr"),
            pl.lit(1.0, dtype=pl.Float64).alias("cfacshr"),
            pl.col("Adj_Close").cast(pl.Float64).alias("div_adj_close"),
            pl.lit(None, dtype=pl.Int64).alias("dlstcd"),
            pl.lit(None, dtype=pl.Float64).alias("dlret"),
            pl.lit(None, dtype=pl.Int64).alias("shrout"),
            pl.lit("Yahoo").alias("source"),
        ])
        yahoo_frames.append(norm)

    yahoo_all = pl.concat(yahoo_frames).sort(["ticker", "date"])
    # Compute returns via percent change
    yahoo_all = yahoo_all.with_columns([
        (pl.col("split_adj_close") / pl.col("split_adj_close").shift(1).over("ticker")).log().alias("ret_price"),
        (pl.col("div_adj_close") / pl.col("div_adj_close").shift(1).over("ticker")).log().alias("ret_total"),
    ])
    logger.info("Loaded WRDS Yahoo (2025-2026): %d rows", len(yahoo_all))

    # Combine annual WRDS panel
    annual_panel = pl.concat([wrds_crsp_dedup, yahoo_all])
    logger.info("Total annual panel rows (2000-2026): %d", len(annual_panel))

    logger.info("Step 3: Ingesting US_history.parquet for target PERMNOs...")
    lf_us = pl.scan_parquet(us_hist_path)
    us_data = (
        lf_us.filter(pl.col("PERMNO").is_in(list(all_target_permnos)))
        .select(crsp_cols_needed)
        .collect()
    )
    logger.info("Extracted %d rows from US_history.parquet for target PERMNOs", len(us_data))

    # Normalize US_history schema identically to CRSP
    cfacpr_us = pl.when(pl.col("CFACPR") == 0).then(None).otherwise(pl.col("CFACPR"))
    us_norm = us_data.select([
        pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d").alias("date"),
        pl.col("TICKER").alias("ticker"),
        pl.col("PERMNO").cast(pl.Int64).alias("permno"),
        pl.col("PRC").abs().cast(pl.Float64).alias("close_raw"),
        pl.col("OPENPRC").abs().cast(pl.Float64).alias("open_raw"),
        pl.col("ASKHI").cast(pl.Float64).alias("high_raw"),
        pl.col("BIDLO").cast(pl.Float64).alias("low_raw"),
        pl.col("VOL").cast(pl.Int64).alias("volume"),
        (pl.col("PRC").abs() / cfacpr_us).cast(pl.Float64).alias("split_adj_close"),
        (pl.col("OPENPRC").abs() / cfacpr_us).cast(pl.Float64).alias("split_adj_open"),
        (pl.col("ASKHI") / cfacpr_us).cast(pl.Float64).alias("split_adj_high"),
        (pl.col("BIDLO") / cfacpr_us).cast(pl.Float64).alias("split_adj_low"),
        pl.col("RETX").cast(pl.Float64, strict=False).log1p().alias("ret_price"),
        pl.col("RET").cast(pl.Float64, strict=False).log1p().alias("ret_total"),
        pl.col("CFACPR").cast(pl.Float64).alias("cfacpr"),
        pl.col("CFACSHR").cast(pl.Float64).alias("cfacshr"),
        pl.lit(None, dtype=pl.Float64).alias("div_adj_close"),
        pl.col("DLSTCD").cast(pl.Int64, strict=False).alias("dlstcd"),
        pl.col("DLRET").cast(pl.Float64, strict=False).alias("dlret"),
        pl.col("SHROUT").cast(pl.Int64, strict=False).alias("shrout"),
        pl.lit("US_history").alias("source"),
    ])

    # Deduplicate US_history internal multi-distribution rows
    us_norm_dedup = us_norm.unique(subset=["permno", "date"], keep="first")
    logger.info(
        "US_history deduplication: %d multi-distribution duplicates dropped, retaining %d rows",
        len(us_norm) - len(us_norm_dedup), len(us_norm_dedup)
    )

    logger.info("Step 4: Merging Annual Panel with US_history backfill...")
    # Pre-filter US_history to keep ONLY backfilled keys (since annual WRDS rows take precedence for Russell share-class ticker formatting)
    us_backfill_only = us_norm_dedup.join(
        wrds_crsp_dedup.select(["permno", "date"]),
        on=["permno", "date"],
        how="anti"
    )
    logger.info("US_history rows to add to panel (backfilled sessions): %d", len(us_backfill_only))

    # Combine annual panel and backfill
    merged_panel = pl.concat([annual_panel, us_backfill_only])

    # Step 4.3: Deduplicate again to ensure strict uniqueness
    p_not_null = merged_panel.filter(pl.col("permno").is_not_null()).unique(subset=["permno", "date"], keep="first")
    p_null = merged_panel.filter(pl.col("permno").is_null()).unique(subset=["ticker", "date"], keep="first")
    merged_panel = pl.concat([p_not_null, p_null])
    logger.info("Merged panel deduplicated total rows: %d", len(merged_panel))

    # Sort deterministically
    # Order: by security (permno if not null else ticker), then date
    merged_panel = merged_panel.with_columns(
        pl.when(pl.col("permno").is_not_null())
        .then(pl.col("permno").cast(pl.Utf8))
        .otherwise(pl.col("ticker"))
        .alias("entity_id")
    ).sort(["entity_id", "date"])

    logger.info("Step 5: Computing cumulative session counts and features_ready...")
    # Cumulative sessions per entity (1-based count)
    merged_panel = merged_panel.with_columns([
        pl.cum_count("date").over("entity_id").alias("session_idx"),
    ])
    merged_panel = merged_panel.with_columns([
        (pl.col("session_idx") >= 332).alias("features_ready")
    ])

    ready_count = merged_panel.filter(pl.col("features_ready")).height
    logger.info(
        "Features ready count: %d / %d (%.2f%%)",
        ready_count, len(merged_panel), (ready_count / len(merged_panel)) * 100
    )

    final_panel = merged_panel.drop(["entity_id", "session_idx"])

    # Step 6: Construct Point-in-Time Universe Mask
    logger.info("Step 6: Constructing Point-in-Time Universe Mask...")
    # Pre-map constituents per year: Year -> set of Tickers and set of PERMNOs
    uni_by_year = {}
    for yr, group in uni_df.group_by("Year"):
        t_set = set(group["Ticker"].drop_nulls().unique().to_list())
        p_set = {master_ticker_to_permno[t] for t in t_set if t in master_ticker_to_permno and master_ticker_to_permno[t] is not None}
        uni_by_year[yr[0] if isinstance(yr, tuple) else yr] = (t_set, p_set)

    # For every row in panel: determine if in_universe
    # recon_year = date.year if date.month >= 7 else date.year - 1
    mask_df = final_panel.select(["date", "permno", "ticker"]).with_columns([
        pl.when(pl.col("date").dt.month() >= 7)
        .then(pl.col("date").dt.year())
        .otherwise(pl.col("date").dt.year() - 1)
        .alias("recon_year")
    ])

    lookup_rows = []
    for yr, (t_set, p_set) in uni_by_year.items():
        for p in p_set:
            lookup_rows.append({"recon_year": yr, "permno": p, "in_universe_p": True})
    uni_lookup_p = pl.DataFrame(lookup_rows)

    lookup_t_rows = []
    for yr, (t_set, p_set) in uni_by_year.items():
        for t in t_set:
            lookup_t_rows.append({"recon_year": yr, "ticker": t, "in_universe_t": True})
    uni_lookup_t = pl.DataFrame(lookup_t_rows)

    mask_joined = mask_df.join(
        uni_lookup_p, on=["recon_year", "permno"], how="left"
    ).join(
        uni_lookup_t, on=["recon_year", "ticker"], how="left"
    ).with_columns([
        (pl.col("in_universe_p").fill_null(False) | pl.col("in_universe_t").fill_null(False)).alias("in_universe")
    ]).select(["date", "permno", "in_universe"])

    universe_mask = mask_joined.unique(subset=["date", "permno"]).sort(["date", "permno"])
    logger.info("Constructed Universe Mask with %d rows", len(universe_mask))

    # Step 7: Saving Deliverables
    logger.info("Step 7: Saving Parquet Deliverables...")
    out_panel_path = processed_dir / "continuous_panel.parquet"
    out_mask_path = processed_dir / "universe_mask.parquet"

    final_panel.write_parquet(out_panel_path, compression="zstd")
    universe_mask.write_parquet(out_mask_path, compression="zstd")

    logger.info("Saved continuous panel to %s (size: %.1f MB)", out_panel_path, out_panel_path.stat().st_size / (1024*1024))
    logger.info("Saved universe mask to %s (size: %.1f MB)", out_mask_path, out_mask_path.stat().st_size / (1024*1024))

    # Step 8: Generating Detailed Statistics for Validation Report
    logger.info("Step 8: Computing Validation Report Metrics...")

    # Backfill statistics
    stocks_with_backfill = final_panel.filter(pl.col("source") == "US_history").select("permno").n_unique()
    backfilled_rows_per_stock = (
        final_panel.filter(pl.col("source") == "US_history")
        .group_by("permno")
        .len()
    )
    avg_sessions_gained = backfilled_rows_per_stock["len"].mean()
    median_sessions_gained = backfilled_rows_per_stock["len"].median()
    max_sessions_gained = backfilled_rows_per_stock["len"].max()

    # Pre-2000-06-30 statistics
    import datetime
    pre_2000_stocks = final_panel.filter(pl.col("date") < datetime.date(2000, 6, 30)).select("permno").n_unique()
    pre_2000_rows = final_panel.filter(pl.col("date") < datetime.date(2000, 6, 30)).height

    # Stock counts
    n_total_stocks = final_panel.select(
        pl.when(pl.col("permno").is_not_null())
        .then(pl.col("permno").cast(pl.Utf8))
        .otherwise(pl.col("ticker"))
        .alias("id")
    ).n_unique()
    n_unique_permnos = final_panel.select("permno").drop_nulls().n_unique()
    n_unique_tickers = final_panel.select("ticker").n_unique()
    min_date = final_panel.select("date").min().item()
    max_date = final_panel.select("date").max().item()

    # History length distribution
    hist_lengths = final_panel.group_by(
        pl.when(pl.col("permno").is_not_null()).then(pl.col("permno").cast(pl.Utf8)).otherwise(pl.col("ticker")).alias("id")
    ).len()

    # Factor alignment check
    ff_df = pl.read_csv(data_dir / "ff.csv")
    ff_dates = set(ff_df["date"].str.to_date("%Y-%m-%d").to_list())
    panel_trading_dates = set(final_panel.select("date").unique()["date"].to_list())
    panel_dates_in_ff_range = {d for d in panel_trading_dates if datetime.date(2000, 1, 3) <= d <= datetime.date(2026, 7, 31)}
    missing_in_ff = panel_dates_in_ff_range - ff_dates
    logger.info("Dates in panel within FF range missing in ff.csv: %d", len(missing_in_ff))

    # Step 8: Log Validation Summary & Save Structured Metrics
    logger.info("=== Continuous Panel Validation Summary ===")
    logger.info("Unique Securities: %d (PERMNOs: %d, Tickers: %d)", n_total_stocks, n_unique_permnos, n_unique_tickers)
    logger.info("Total Security-Date Rows: %d (Range: %s to %s)", len(final_panel), min_date, max_date)
    logger.info("Pre-2000-06-30 Backfill: %d stocks (%d sessions)", pre_2000_stocks, pre_2000_rows)
    logger.info("History Length Distribution: Min=%s, Median=%.0f, Max=%s sessions",
                hist_lengths["len"].min(), hist_lengths["len"].median(), hist_lengths["len"].max())
    logger.info("Backfill Gained: %d stocks, %d sessions (Mean=%.1f, Median=%.0f, Max=%d)",
                stocks_with_backfill, len(us_backfill_only), avg_sessions_gained, median_sessions_gained, max_sessions_gained)
    logger.info("Features Ready (cumulative >= 332): %d / %d (%.2f%%)",
                ready_count, len(final_panel), (ready_count / len(final_panel)) * 100)

    import json
    metrics_path = processed_dir / "panel_validation_metrics.json"
    validation_metrics = {
        "n_total_stocks": int(n_total_stocks),
        "n_unique_permnos": int(n_unique_permnos),
        "n_unique_tickers": int(n_unique_tickers),
        "total_rows": int(len(final_panel)),
        "min_date": str(min_date),
        "max_date": str(max_date),
        "pre_2000_stocks": int(pre_2000_stocks),
        "pre_2000_rows": int(pre_2000_rows),
        "history_lengths": {
            "min": int(hist_lengths["len"].min()),
            "median": float(hist_lengths["len"].median()),
            "mean": float(hist_lengths["len"].mean()),
            "max": int(hist_lengths["len"].max()),
        },
        "backfill": {
            "stocks_with_backfill": int(stocks_with_backfill),
            "total_backfilled_sessions": int(len(us_backfill_only)),
            "avg_sessions_gained": float(avg_sessions_gained),
            "median_sessions_gained": float(median_sessions_gained),
            "max_sessions_gained": int(max_sessions_gained),
        },
        "features_ready_count": int(ready_count),
        "features_ready_pct": float((ready_count / len(final_panel)) * 100),
    }
    with open(metrics_path, "w", encoding="utf-8") as jf:
        json.dump(validation_metrics, jf, indent=2)
    logger.info("Saved validation metrics to %s", metrics_path)
    logger.info("Pipeline completed successfully in %.1f seconds.", time.time() - t0)

if __name__ == "__main__":
    main()
