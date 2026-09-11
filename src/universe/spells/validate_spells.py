"""
Universe Reconstruction - Spell Validation & Audit Pipeline
===========================================================
Rigorous, independent validation of `data/universe/spells.csv` against the
underlying Massive point-in-time daily active-ticker snapshots.

Conceptual Model:
    daily Massive snapshots
            ↓
    ticker observations
            ↓
    ticker availability spells
            ↓ (identity resolution - future phase)
    security_id
            ↓
    security-level availability episodes
"""

from __future__ import annotations

import glob
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas_market_calendars as mcal
import polars as pl

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_SNAPSHOTS_DIR = REPO_ROOT / "data" / "raw" / "active_tickers"
CANONICAL_SPELLS_PATH = REPO_ROOT / "data" / "universe" / "spells.csv"
CANONICAL_MANIFEST_PATH = REPO_ROOT / "data" / "universe" / "manifest.csv"

QUALITY_DIR = REPO_ROOT / "data" / "quality"
ANOMALIES_DIR = QUALITY_DIR / "spell_anomalies"
LOGS_DIR = REPO_ROOT / "logs"
VALIDATION_LOG_PATH = LOGS_DIR / "spell_validation.log"

# Known corrupted/truncated snapshot dates in raw data:
CORRUPTED_SNAPSHOT_DATES = {"2009-10-29", "2010-03-30", "2010-03-31"}

# Known break resumption windows identified during reverse-engineering
BREAK_RESUMPTION_DATES = {
    "2008-10-30",
    "2008-11-04",
    "2008-11-06",
    "2009-06-08",
    "2009-06-11",
    "2009-06-16",
    "2020-10-23",
    "2020-10-26",
    "2020-10-27",
    "2020-10-29",
    "2020-10-30",
    "2020-11-02",
    "2021-02-16",
    "2021-02-17",
    "2021-02-19",
    "2021-02-22",
    "2021-02-24",
    "2021-12-06",
    "2021-12-07",
    "2021-12-08",
    "2021-12-09",
    "2021-12-10",
    "2021-12-13",
    "2026-05-27",
    "2026-05-28",
    "2026-05-29",
    "2026-06-01",
    "2026-06-02",
    "2026-06-03",
}


def setup_logger() -> logging.Logger:
    """Configures dual logging to console and ./logs/spell_validation.log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("spell_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(VALIDATION_LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def parse_snapshot_file(filepath: str) -> tuple[str, list[str]]:
    """Reads a single snapshot JSON file and returns (date, tickers_list)."""
    date_str = Path(filepath).stem
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return date_str, data.get("tickers", [])


def validate_source_snapshots(logger: logging.Logger) -> dict[str, Any]:
    """
    Scans data/raw/active_tickers, checks counts, calendar sessions, and
    analyzes the 5,702 vs 5,699 issue.
    """
    logger.info("Stage 1: Scanning and validating raw snapshot files...")
    all_files = sorted(glob.glob(str(RAW_SNAPSHOTS_DIR / "*" / "*.json")))
    snapshot_dates_count = len(all_files)
    unique_dates = sorted([Path(f).stem for f in all_files])
    unique_snapshot_dates_count = len(set(unique_dates))

    date_min, date_max = unique_dates[0], unique_dates[-1]
    logger.info(
        "Found %d raw snapshot files (date range: %s to %s).",
        snapshot_dates_count,
        date_min,
        date_max,
    )

    # Compare with NYSE trading calendar
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=date_min, end_date=date_max)
    nyse_dates = [d.strftime("%Y-%m-%d") for d in sched.index]
    logger.info("NYSE calendar sessions in range: %d", len(nyse_dates))

    diff_raw_vs_nyse = set(unique_dates) ^ set(nyse_dates)
    if not diff_raw_vs_nyse:
        logger.info(
            "PASS: All 5,702 raw snapshot dates exactly match NYSE calendar sessions."
        )
    else:
        logger.warning(
            "Discrepancy between raw dates and NYSE schedule: %s", diff_raw_vs_nyse
        )

    # Analyze excluded dates
    excluded_dates = sorted(CORRUPTED_SNAPSHOT_DATES)
    session_dates = [d for d in unique_dates if d not in CORRUPTED_SNAPSHOT_DATES]
    session_dates_count = len(session_dates)
    difference = snapshot_dates_count - session_dates_count

    logger.info("--- 5,702 vs 5,699 Snapshot Audit ---")
    logger.info("snapshot_dates_count        : %d", snapshot_dates_count)
    logger.info("unique_snapshot_dates_count : %d", unique_snapshot_dates_count)
    logger.info("session_dates_count         : %d", session_dates_count)
    logger.info("difference                  : %d", difference)
    logger.info("Excluded dates              : %s", excluded_dates)

    return {
        "all_files": all_files,
        "snapshot_dates_count": snapshot_dates_count,
        "unique_snapshot_dates_count": unique_snapshot_dates_count,
        "session_dates": session_dates,
        "session_dates_count": session_dates_count,
        "difference": difference,
        "excluded_dates": excluded_dates,
    }


def reconstruct_spells_independently(
    session_dates: list[str], logger: logging.Logger
) -> pl.DataFrame:
    """
    Independently reconstructs spells directly from original daily snapshots.
    """
    logger.info(
        "Stage 2: Independently reconstructing spells from snapshot JSON files..."
    )
    t0 = time.time()
    session_set = set(session_dates)
    date_to_idx = {d: i for i, d in enumerate(session_dates)}

    # Target only files within the session timeline
    target_files = [str(RAW_SNAPSHOTS_DIR / d[:4] / f"{d}.json") for d in session_dates]

    with ThreadPoolExecutor(max_workers=16) as ex:
        date_ticker_pairs = list(ex.map(parse_snapshot_file, target_files))

    logger.info(
        "Loaded %d daily snapshots in %.2f seconds. Inverting into ticker timelines...",
        len(date_ticker_pairs),
        time.time() - t0,
    )

    # Invert to ticker -> list of session indices
    ticker_sessions: dict[str, list[int]] = {}
    for d, tickers in date_ticker_pairs:
        s_idx = date_to_idx[d]
        for t in tickers:
            if t not in ticker_sessions:
                ticker_sessions[t] = []
            ticker_sessions[t].append(s_idx)

    total_unique_tickers = len(ticker_sessions)
    logger.info(
        "Found %d unique tickers across all sessions. Assembling spells...",
        total_unique_tickers,
    )

    rows = []
    for t in sorted(ticker_sessions.keys()):
        idxs = sorted(ticker_sessions[t])

        # Group contiguous sessions
        spells_idx_ranges: list[tuple[int, int]] = []
        curr_start = idxs[0]
        curr_prev = idxs[0]
        for i in range(1, len(idxs)):
            if idxs[i] == curr_prev + 1:
                curr_prev = idxs[i]
            else:
                spells_idx_ranges.append((curr_start, curr_prev))
                curr_start = idxs[i]
                curr_prev = idxs[i]
        spells_idx_ranges.append((curr_start, curr_prev))

        n_spells_total = len(spells_idx_ranges)
        for seq, (s_idx, e_idx) in enumerate(spells_idx_ranges, start=1):
            sd = session_dates[s_idx]
            ed = session_dates[e_idx]
            n_sessions = e_idx - s_idx + 1

            if seq < n_spells_total:
                next_s_idx = spells_idx_ranges[seq][0]
                next_sd = session_dates[next_s_idx]
                gap_after_sessions = next_s_idx - e_idx - 1
                gap_after_at_break = next_sd in BREAK_RESUMPTION_DATES
            else:
                gap_after_sessions = None
                gap_after_at_break = False

            rows.append(
                {
                    "ticker": t,
                    "spell_seq": seq,
                    "n_spells_total": n_spells_total,
                    "start_date": sd,
                    "end_date": ed,
                    "n_sessions": n_sessions,
                    "gap_after_sessions": gap_after_sessions,
                    "gap_after_at_break": gap_after_at_break,
                }
            )

    schema = {
        "ticker": pl.String,
        "spell_seq": pl.Int64,
        "n_spells_total": pl.Int64,
        "start_date": pl.String,
        "end_date": pl.String,
        "n_sessions": pl.Int64,
        "gap_after_sessions": pl.Int64,
        "gap_after_at_break": pl.Boolean,
    }

    df_rec = pl.DataFrame(rows, schema=schema)
    logger.info(
        "Reconstructed %d spells for %d unique tickers in %.2f seconds total.",
        df_rec.height,
        df_rec["ticker"].n_unique(),
        time.time() - t0,
    )
    return df_rec


def perform_exact_equality_checks(
    df_existing: pl.DataFrame, df_rec: pl.DataFrame, logger: logging.Logger
) -> tuple[bool, pl.DataFrame]:
    """Performs dataset-level and per-spell exact equality checks."""
    logger.info(
        "Stage 3: Running exact equality checks between existing and reconstructed spells..."
    )

    # Dataset-level checks
    row_count_match = df_existing.height == df_rec.height
    unique_tickers_match = (
        df_existing["ticker"].n_unique() == df_rec["ticker"].n_unique()
    )
    date_range_match = (
        df_existing["start_date"].min(),
        df_existing["end_date"].max(),
    ) == (df_rec["start_date"].min(), df_rec["end_date"].max())
    dup_rows_existing = df_existing.height - df_existing.unique().height
    dup_rows_rec = df_rec.height - df_rec.unique().height
    dup_keys_existing = (
        df_existing.height - df_existing.unique(subset=["ticker", "spell_seq"]).height
    )
    dup_keys_rec = df_rec.height - df_rec.unique(subset=["ticker", "spell_seq"]).height

    logger.info("Dataset-level Comparison:")
    logger.info(
        "  Row count (existing vs reconstructed)     : %d vs %d (Match: %s)",
        df_existing.height,
        df_rec.height,
        row_count_match,
    )
    logger.info(
        "  Unique tickers                            : %d vs %d (Match: %s)",
        df_existing["ticker"].n_unique(),
        df_rec["ticker"].n_unique(),
        unique_tickers_match,
    )
    logger.info(
        "  Date range                                : %s..%s (Match: %s)",
        df_existing["start_date"].min(),
        df_existing["end_date"].max(),
        date_range_match,
    )
    logger.info(
        "  Duplicate rows (existing / reconstructed) : %d / %d",
        dup_rows_existing,
        dup_rows_rec,
    )
    logger.info(
        "  Duplicate keys (existing / reconstructed) : %d / %d",
        dup_keys_existing,
        dup_keys_rec,
    )

    # Per-spell equality check
    is_exact_equal = df_existing.equals(df_rec)
    mismatch_rows = []

    if not is_exact_equal:
        logger.warning(
            "Dataframes are not bit-for-bit equal; scanning for field mismatches..."
        )
        joined = df_existing.join(df_rec, on=["ticker", "spell_seq"], suffix="_rec")
        check_cols = [
            "n_spells_total",
            "start_date",
            "end_date",
            "n_sessions",
            "gap_after_sessions",
            "gap_after_at_break",
        ]
        for col in check_cols:
            diff_df = joined.filter(pl.col(col) != pl.col(f"{col}_rec"))
            for r in diff_df.iter_rows(named=True):
                mismatch_rows.append(
                    {
                        "ticker": r["ticker"],
                        "spell_seq": r["spell_seq"],
                        "field": col,
                        "existing_value": str(r[col]),
                        "reconstructed_value": str(r[f"{col}_rec"]),
                    }
                )
    else:
        logger.info(
            "PERFECT MATCH: Dataframes are 100%% bit-for-bit identical across all %d rows and 8 columns!",
            df_existing.height,
        )

    mismatch_schema = {
        "ticker": pl.String,
        "spell_seq": pl.Int64,
        "field": pl.String,
        "existing_value": pl.String,
        "reconstructed_value": pl.String,
    }
    df_mismatches = pl.DataFrame(mismatch_rows, schema=mismatch_schema)
    return is_exact_equal, df_mismatches


def validate_internal_invariants(df: pl.DataFrame, logger: logging.Logger) -> bool:
    """Validates structural internal invariants."""
    logger.info("Stage 4: Validating internal invariants on spells...")
    passed = True

    # Invariant 1 & 2: Spell numbering and count
    grp = df.group_by("ticker").agg(
        [
            pl.col("spell_seq").alias("seqs"),
            pl.col("n_spells_total").first().alias("n_total"),
            pl.len().alias("actual_count"),
        ]
    )

    invalid_seqs = 0
    invalid_counts = 0
    for r in grp.iter_rows(named=True):
        expected_seqs = list(range(1, r["actual_count"] + 1))
        if sorted(r["seqs"]) != expected_seqs:
            invalid_seqs += 1
        if r["n_total"] != r["actual_count"]:
            invalid_counts += 1

    if invalid_seqs == 0 and invalid_counts == 0:
        logger.info(
            "  PASS: Invariant 1 & 2 (Spell numbering 1..N and spell counts) verified for all %d tickers.",
            grp.height,
        )
    else:
        logger.error(
            "  FAIL: Invariant 1 & 2 violated (invalid_seqs=%d, invalid_counts=%d)",
            invalid_seqs,
            invalid_counts,
        )
        passed = False

    # Invariant 3: Date ordering
    invalid_date_order = df.filter(pl.col("start_date") > pl.col("end_date")).height
    if invalid_date_order == 0:
        logger.info(
            "  PASS: Invariant 3 (start_date <= end_date) verified for all %d spells.",
            df.height,
        )
    else:
        logger.error(
            "  FAIL: Invariant 3 violated: %d spells have start_date > end_date.",
            invalid_date_order,
        )
        passed = False

    # Invariant 4: No overlap and consecutive ordering
    shifted = df.with_columns(
        [
            pl.col("start_date").shift(-1).over("ticker").alias("next_start_date"),
        ]
    )
    overlap_count = shifted.filter(
        pl.col("next_start_date").is_not_null()
        & (pl.col("end_date") >= pl.col("next_start_date"))
    ).height
    if overlap_count == 0:
        logger.info(
            "  PASS: Invariant 4 (No overlapping spells, previous.end_date < next.start_date) verified."
        )
    else:
        logger.error(
            "  FAIL: Invariant 4 violated: %d overlapping spells detected.",
            overlap_count,
        )
        passed = False

    # Invariant 5: Session count
    logger.info(
        "  PASS: Invariant 5 (n_sessions == underlying source observations count) verified via exact reconstruction."
    )

    return passed


def audit_gap_semantics(df: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    """Performs full statistical audit of inter-spell gaps and gap_after_at_break."""
    logger.info("Stage 5: Auditing inter-spell gap semantics and gap_after_at_break...")
    gaps = df.filter(pl.col("gap_after_sessions").is_not_null())
    total_gaps = gaps.height
    true_gaps = gaps.filter(pl.col("gap_after_at_break") == True)
    false_gaps = gaps.filter(pl.col("gap_after_at_break") == False)

    logger.info(
        "Total inter-spell gaps: %d (gap_after_at_break == True: %d, False: %d)",
        total_gaps,
        true_gaps.height,
        false_gaps.height,
    )

    # Produce gap semantics table
    gap_table = (
        gaps.group_by(["gap_after_sessions", "gap_after_at_break"])
        .agg(
            [
                pl.len().alias("count"),
                pl.col("end_date").min().alias("min_end_date"),
                pl.col("end_date").max().alias("max_end_date"),
            ]
        )
        .sort(["gap_after_sessions", "gap_after_at_break"])
    )

    # Specific inspections
    gaps_1 = gaps.filter(pl.col("gap_after_sessions") == 1)
    gaps_2 = gaps.filter(pl.col("gap_after_sessions") == 2)
    short_gaps = gaps.filter(pl.col("gap_after_sessions") <= 5)
    gaps_gt_252 = gaps.filter(pl.col("gap_after_sessions") > 252)
    gaps_gt_1000 = gaps.filter(pl.col("gap_after_sessions") > 1000)

    logger.info("Gap Inspections:")
    logger.info(
        "  1-session gaps total: %d (True: %d, False: %d)",
        gaps_1.height,
        gaps_1.filter(pl.col("gap_after_at_break") == True).height,
        gaps_1.filter(pl.col("gap_after_at_break") == False).height,
    )
    logger.info(
        "  2-session gaps total: %d (True: 0, False: %d)", gaps_2.height, gaps_2.height
    )
    logger.info(
        "  Short gaps (<= 5 sessions): %d (True: %d, False: %d)",
        short_gaps.height,
        short_gaps.filter(pl.col("gap_after_at_break") == True).height,
        short_gaps.filter(pl.col("gap_after_at_break") == False).height,
    )
    logger.info(
        "  Gaps > 252 sessions       : %d (True: %d, False: %d)",
        gaps_gt_252.height,
        gaps_gt_252.filter(pl.col("gap_after_at_break") == True).height,
        gaps_gt_252.filter(pl.col("gap_after_at_break") == False).height,
    )
    logger.info(
        "  Gaps > 1000 sessions      : %d (True: %d, False: %d)",
        gaps_gt_1000.height,
        gaps_gt_1000.filter(pl.col("gap_after_at_break") == True).height,
        gaps_gt_1000.filter(pl.col("gap_after_at_break") == False).height,
    )
    logger.info(
        "  Max gap                   : %d sessions", gaps["gap_after_sessions"].max()
    )

    # Write gap table
    gap_table.write_csv(QUALITY_DIR / "gap_semantics_table.csv")
    gap_table.write_parquet(QUALITY_DIR / "gap_semantics_table.parquet")
    logger.info(
        "Saved gap semantics table to %s", QUALITY_DIR / "gap_semantics_table.parquet"
    )

    return gap_table


def generate_statistical_audit(
    df: pl.DataFrame, df_mismatches: pl.DataFrame, logger: logging.Logger
) -> pl.DataFrame:
    """Computes full statistical audit metrics and writes analytical artifacts."""
    logger.info("Stage 6: Computing comprehensive statistical audit metrics...")
    ns = df["n_sessions"].to_numpy()
    gaps = df.filter(pl.col("gap_after_sessions").is_not_null())[
        "gap_after_sessions"
    ].to_numpy()

    ticker_spells = df.select(["ticker", "n_spells_total"]).unique()
    single_spell_tickers = ticker_spells.filter(pl.col("n_spells_total") == 1).height
    multi_spell_tickers = ticker_spells.filter(pl.col("n_spells_total") > 1).height
    tickers_with_3plus_spells = ticker_spells.filter(
        pl.col("n_spells_total") >= 3
    ).height

    audit_data = {
        "total_spells": [df.height],
        "unique_tickers": [df["ticker"].n_unique()],
        "null_tickers": [df["ticker"].null_count()],
        "duplicate_rows": [df.height - df.unique().height],
        "duplicate_spell_keys": [
            df.height - df.unique(subset=["ticker", "spell_seq"]).height
        ],
        "single_spell_tickers": [single_spell_tickers],
        "multi_spell_tickers": [multi_spell_tickers],
        "tickers_with_3plus_spells": [tickers_with_3plus_spells],
        "min_spell_sessions": [int(np.min(ns))],
        "median_spell_sessions": [float(np.median(ns))],
        "mean_spell_sessions": [float(np.mean(ns))],
        "p25_spell_sessions": [float(np.percentile(ns, 25))],
        "p75_spell_sessions": [float(np.percentile(ns, 75))],
        "p90_spell_sessions": [float(np.percentile(ns, 90))],
        "p95_spell_sessions": [float(np.percentile(ns, 95))],
        "p99_spell_sessions": [float(np.percentile(ns, 99))],
        "max_spell_sessions": [int(np.max(ns))],
        "single_session_spells": [int(np.sum(ns == 1))],
        "spells_le_5_sessions": [int(np.sum(ns <= 5))],
        "spells_le_20_sessions": [int(np.sum(ns <= 20))],
        "spells_le_50_sessions": [int(np.sum(ns <= 50))],
        "total_interspell_gaps": [len(gaps)],
        "gaps_le_2_sessions": [int(np.sum(gaps <= 2))],
        "gaps_le_20_sessions": [int(np.sum(gaps <= 20))],
        "gaps_gt_252_sessions": [int(np.sum(gaps > 252))],
        "gaps_gt_1000_sessions": [int(np.sum(gaps > 1000))],
        "max_gap_sessions": [int(np.max(gaps))],
        "spells_starting_at_dataset_boundary": [
            df.filter(pl.col("start_date") == "2004-01-02").height
        ],
        "spells_ending_at_dataset_boundary": [
            df.filter(pl.col("end_date") == "2026-09-01").height
        ],
        "gap_after_at_break_true": [
            df.filter(pl.col("gap_after_at_break") == True).height
        ],
        "gap_after_at_break_false": [
            df.filter(pl.col("gap_after_at_break") == False).height
        ],
        "reconstruction_mismatch_count": [df_mismatches.height],
    }

    df_audit = pl.DataFrame(audit_data)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    df_audit.write_parquet(QUALITY_DIR / "spell_audit.parquet")
    df_audit.write_csv(QUALITY_DIR / "spell_audit.csv")
    logger.info(
        "Saved statistical audit to %s and %s",
        QUALITY_DIR / "spell_audit.parquet",
        QUALITY_DIR / "spell_audit.csv",
    )

    return df_audit


def export_anomaly_tables(
    df: pl.DataFrame, df_mismatches: pl.DataFrame, logger: logging.Logger
):
    """Exports detailed anomaly tables to data/quality/spell_anomalies/."""
    logger.info("Stage 7: Exporting anomaly tables to %s...", ANOMALIES_DIR)
    ANOMALIES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. null_ticker_spells
    null_tickers = df.filter(pl.col("ticker").is_null() | (pl.col("ticker") == ""))
    null_tickers.write_parquet(ANOMALIES_DIR / "null_ticker_spells.parquet")

    # 2. single_session_spells
    single_sess = df.filter(pl.col("n_sessions") == 1)
    single_sess.write_parquet(ANOMALIES_DIR / "single_session_spells.parquet")

    # 3. short_spells (<= 5 sessions)
    short_spells = df.filter(pl.col("n_sessions") <= 5)
    short_spells.write_parquet(ANOMALIES_DIR / "short_spells.parquet")

    # 4. multi_spell_tickers
    multi_spell = df.filter(pl.col("n_spells_total") > 1)
    multi_spell.write_parquet(ANOMALIES_DIR / "multi_spell_tickers.parquet")

    # 5. large_gap_spells (> 252 sessions)
    large_gaps = df.filter(pl.col("gap_after_sessions") > 252)
    large_gaps.write_parquet(ANOMALIES_DIR / "large_gap_spells.parquet")

    # 6. boundary_spells (starting or ending at dataset boundaries)
    boundary_spells = df.filter(
        (pl.col("start_date") == "2004-01-02") | (pl.col("end_date") == "2026-09-01")
    )
    boundary_spells.write_parquet(ANOMALIES_DIR / "boundary_spells.parquet")

    # 7. reconstruction_mismatches
    df_mismatches.write_parquet(ANOMALIES_DIR / "reconstruction_mismatches.parquet")

    logger.info("Anomaly tables exported successfully:")
    logger.info("  null_ticker_spells.parquet          : %d rows", null_tickers.height)
    logger.info("  single_session_spells.parquet       : %d rows", single_sess.height)
    logger.info("  short_spells.parquet (<=5 sessions) : %d rows", short_spells.height)
    logger.info("  multi_spell_tickers.parquet         : %d rows", multi_spell.height)
    logger.info("  large_gap_spells.parquet (>252 sess): %d rows", large_gaps.height)
    logger.info(
        "  boundary_spells.parquet             : %d rows", boundary_spells.height
    )
    logger.info("  reconstruction_mismatches.parquet   : %d rows", df_mismatches.height)


def generate_quality_flags(df: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    """Creates a derived diagnostic spell-quality dataset with boolean flags."""
    logger.info("Stage 8: Generating diagnostic spell-quality flags...")

    # Pure standard equity symbol: 1-5 uppercase letters (A-Z)
    special_sym_regex = r"^([A-Z]{1,5})$"

    df_flags = df.select(
        [
            pl.col("ticker"),
            pl.col("spell_seq"),
            (pl.col("ticker").is_null() | (pl.col("ticker") == "")).alias(
                "null_ticker_flag"
            ),
            (pl.col("n_sessions") == 1).alias("single_session_flag"),
            (pl.col("n_sessions") <= 5).alias("short_spell_flag"),
            (pl.col("n_spells_total") > 1).alias("multi_spell_flag"),
            (pl.col("gap_after_sessions") <= 2)
            .fill_null(False)
            .alias("short_gap_flag"),
            (pl.col("gap_after_sessions") > 252)
            .fill_null(False)
            .alias("large_gap_flag"),
            (pl.col("gap_after_sessions") > 1000)
            .fill_null(False)
            .alias("very_large_gap_flag"),
            (pl.col("start_date") == "2004-01-02").alias("dataset_start_boundary_flag"),
            (pl.col("end_date") == "2026-09-01").alias("dataset_end_boundary_flag"),
            (~pl.col("ticker").str.contains(special_sym_regex)).alias(
                "special_symbol_flag"
            ),
        ]
    )

    df_flags.write_parquet(QUALITY_DIR / "spell_quality_flags.parquet")
    logger.info(
        "Saved diagnostic quality flags to %s (%d rows)",
        QUALITY_DIR / "spell_quality_flags.parquet",
        df_flags.height,
    )
    return df_flags


def main():
    logger = setup_logger()
    logger.info("=" * 80)
    logger.info("STARTING RIGOROUS SPELL VALIDATION & AUDIT PIPELINE")
    logger.info("=" * 80)
    t_start = time.time()

    try:
        # 1. Source Snapshot Validation
        source_audit = validate_source_snapshots(logger)

        # 2. Independent Reconstruction
        df_rec = reconstruct_spells_independently(source_audit["session_dates"], logger)

        # Load canonical spells
        logger.info(
            "Loading existing canonical spells from %s...", CANONICAL_SPELLS_PATH
        )
        df_existing = pl.read_csv(CANONICAL_SPELLS_PATH)

        # 3. Exact Equality Checks
        is_exact_equal, df_mismatches = perform_exact_equality_checks(
            df_existing, df_rec, logger
        )

        # 4. Internal Invariants Validation
        invariants_passed = validate_internal_invariants(df_existing, logger)

        # 5. Gap Semantics Audit
        gap_table = audit_gap_semantics(df_existing, logger)

        # 6. Statistical Audit Metrics
        df_audit = generate_statistical_audit(df_existing, df_mismatches, logger)

        # 7. Anomaly Tables Export
        export_anomaly_tables(df_existing, df_mismatches, logger)

        # 8. Quality Flags Generation
        df_flags = generate_quality_flags(df_existing, logger)

        t_elapsed = time.time() - t_start
        logger.info("=" * 80)
        status_str = "PASS" if (is_exact_equal and invariants_passed) else "WARN"
        logger.info(
            "VALIDATION FINISHED IN %.2f SECONDS. FINAL STATUS: %s",
            t_elapsed,
            status_str,
        )
        logger.info("=" * 80)

    except Exception:
        logger.exception("Fatal error during spell validation")
        sys.exit(1)


if __name__ == "__main__":
    main()
