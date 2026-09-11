"""
Merge Backfill Checkpoints Tool.
================================
Scans, concatenates, deduplicates, and validates all checkpoint parquet files
generated during the Massive point-in-time backfill.

Usage:
    python3 scripts/merge_checkpoints.py
    python3 scripts/merge_checkpoints.py --checkpoints-dir backfill/checkpoints --output backfill/manifests/massive_manifest.parquet
    python3 scripts/merge_checkpoints.py --copy-to-v3 --csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINTS_DIR = REPO_ROOT / "data" / "backfill" / "checkpoints"
DEFAULT_OUTPUT_PARQUET = REPO_ROOT / "data" / "backfill" / "manifests" / "massive_manifest.parquet"
V3_MANIFEST_PARQUET = REPO_ROOT / "data" / "manifests" / "v3" / "massive_manifest.parquet"
SPELLS_CSV_PATH = REPO_ROOT / "data" / "universe" / "spells.csv"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("merge_checkpoints")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def merge_checkpoints(
    checkpoints_dir: Path,
    output_parquet: Path,
    spells_csv: Optional[Path] = SPELLS_CSV_PATH,
    copy_to_v3: bool = False,
    write_csv: bool = False,
    logger: Optional[logging.Logger] = None
) -> pl.DataFrame:
    if logger is None:
        logger = setup_logger()

    logger.info("=" * 80)
    logger.info("V3 CHECKPOINT MERGE & VALIDATION ENGINE")
    logger.info("=" * 80)
    logger.info("Scanning directory: %s", checkpoints_dir)

    if not checkpoints_dir.exists():
        logger.error("Checkpoints directory not found: %s", checkpoints_dir)
        sys.exit(1)

    chk_files = sorted(list(checkpoints_dir.glob("checkpoint_*.parquet")))
    if not chk_files:
        # Fallback to any parquet file in directory
        chk_files = sorted(list(checkpoints_dir.glob("*.parquet")))

    if not chk_files:
        logger.error("No checkpoint parquet files found in %s", checkpoints_dir)
        sys.exit(1)

    logger.info("Found %d checkpoint files to merge.", len(chk_files))

    t0 = time.time()
    frames: List[pl.DataFrame] = []
    corrupted_count = 0

    for idx, f in enumerate(chk_files, start=1):
        try:
            df_part = pl.read_parquet(f)
            frames.append(df_part)
        except Exception as exc:
            logger.warning("Could not read checkpoint %s (corrupted or partial): %s", f.name, exc)
            corrupted_count += 1

    if not frames:
        logger.error("All checkpoint files failed to load.")
        sys.exit(1)

    logger.info("Successfully loaded %d / %d files (%d corrupted/skipped).",
                len(frames), len(chk_files), corrupted_count)

    # 1. Concatenate all loaded chunks
    df_raw = pl.concat(frames)
    total_raw_rows = df_raw.height
    logger.info("Total rows before deduplication: %d", total_raw_rows)

    # 2. Deduplicate by spell_id (keeping last update)
    if "spell_id" in df_raw.columns:
        df_merged = df_raw.unique(subset=["spell_id"], keep="last")
    else:
        df_merged = df_raw.unique(subset=["ticker", "spell_seq"], keep="last")

    # 3. Deterministic primary key sorting
    sort_cols = [c for c in ["ticker", "spell_seq"] if c in df_merged.columns]
    if sort_cols:
        df_merged = df_merged.sort(sort_cols)
    elif "spell_id" in df_merged.columns:
        df_merged = df_merged.sort(["spell_id"])

    unique_spells = df_merged.height
    dups_removed = total_raw_rows - unique_spells
    logger.info("Total unique spells after deduplication: %d (removed %d duplicates)",
                unique_spells, dups_removed)

    # 4. Atomic Write to Output Parquet
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_parquet.with_suffix(".parquet.tmp")
    df_merged.write_parquet(tmp_output)
    tmp_output.replace(output_parquet)
    logger.info("Successfully wrote merged master manifest (%d rows) -> %s",
                df_merged.height, output_parquet)

    # Optional CSV Export
    if write_csv:
        csv_path = output_parquet.with_suffix(".csv")
        df_merged.write_csv(csv_path)
        logger.info("Exported CSV copy -> %s", csv_path)

    # Optional Copy to V3 Production Candidates / Manifests
    if copy_to_v3:
        V3_MANIFEST_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        tmp_v3 = V3_MANIFEST_PARQUET.with_suffix(".parquet.tmp")
        df_merged.write_parquet(tmp_v3)
        tmp_v3.replace(V3_MANIFEST_PARQUET)
        logger.info("Updated data/manifests/v3/massive_manifest.parquet (%d rows)", df_merged.height)

    elapsed = time.time() - t0

    # 5. Summary Statistics & Progress vs Universe
    logger.info("-" * 80)
    logger.info("MERGE SUMMARY & EVIDENCE BREAKDOWN:")
    logger.info("-" * 80)
    logger.info("Execution Time   : %.2f seconds", elapsed)
    logger.info("Checkpoint Files : %d read", len(chk_files))
    logger.info("Merged Spells    : %d", unique_spells)

    if spells_csv and spells_csv.exists():
        try:
            df_spells = pl.read_csv(spells_csv)
            univ_total = df_spells.height
            pct_complete = (unique_spells / univ_total) * 100.0
            rem_spells = max(0, univ_total - unique_spells)
            logger.info("Universe Progress: %d / %d (%.2f%% completed, %d remaining)",
                        unique_spells, univ_total, pct_complete, rem_spells)
        except Exception as e:
            logger.debug("Could not read spells.csv: %s", e)

    # Breakdown by lookup_status
    if "lookup_status" in df_merged.columns:
        by_status = df_merged.group_by("lookup_status").agg(pl.len().alias("count")).sort("count", descending=True)
        logger.info("Lookup Status Breakdown:")
        for r in by_status.iter_rows(named=True):
            logger.info("  - %-18s: %6d (%5.2f%%)", r["lookup_status"], r["count"], (r["count"] / unique_spells * 100))

    # Evidence coverage
    if "massive_figi" in df_merged.columns:
        with_figi = df_merged.filter(pl.col("massive_figi").is_not_null()).height
        logger.info("  - Spells with FIGI  : %6d (%5.2f%%)", with_figi, (with_figi / unique_spells * 100))
    if "massive_cik" in df_merged.columns:
        with_cik = df_merged.filter(pl.col("massive_cik").is_not_null()).height
        logger.info("  - Spells with CIK   : %6d (%5.2f%%)", with_cik, (with_cik / unique_spells * 100))
    if "drift_detected" in df_merged.columns:
        drifts = df_merged.filter(pl.col("drift_detected") == True).height
        logger.info("  - Within-Spell Drift: %6d", drifts)

    logger.info("=" * 80)
    return df_merged


def main():
    parser = argparse.ArgumentParser(description="Merge backfill checkpoints into master manifest")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINTS_DIR),
        help=f"Path to checkpoints directory (default: {DEFAULT_CHECKPOINTS_DIR})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PARQUET),
        help=f"Path to output parquet manifest (default: {DEFAULT_OUTPUT_PARQUET})"
    )
    parser.add_argument(
        "--spells-csv",
        type=str,
        default=str(SPELLS_CSV_PATH),
        help=f"Path to universe spells.csv for progress calculation (default: {SPELLS_CSV_PATH})"
    )
    parser.add_argument(
        "--copy-to-v3",
        action="store_true",
        help="Also update data/manifests/v3/massive_manifest.parquet"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also export a CSV version of the merged manifest"
    )
    args = parser.parse_args()

    merge_checkpoints(
        checkpoints_dir=Path(args.checkpoints_dir),
        output_parquet=Path(args.output),
        spells_csv=Path(args.spells_csv) if args.spells_csv else None,
        copy_to_v3=args.copy_to_v3,
        write_csv=args.csv
    )


if __name__ == "__main__":
    main()
