#!/usr/bin/env python3
"""
Merge annual Russell 1000 Parquet datasets into canonical unified panels.

Produces two consolidated Parquet files:
1. `russell1000_by_permno.parquet`: Ordered by security identifier (PERMNO, date)
   creating contiguous entity time series.
2. `russell1000_by_year.parquet`: Ordered chronologically (date, PERMNO, TICKER)
   creating cross-sectional snapshots progressing through time.

Handles schema harmonization between:
- WRDS/CRSP panels (2000-2024): 63 canonical variables with native PERMNO.
- Crawled continuation panels (2025-2026): Harmonized into the 63-variable CRSP schema
  with PERMNO and corporate identifiers mapped from the latest CRSP security master,
  and unavailable CRSP-specific variables set to null.

Usage:
    python tools/merge_annual_parquets.py
    python tools/merge_annual_parquets.py --compression zstd
    python tools/merge_annual_parquets.py --input-dir data/WRDS --output-dir data/WRDS
"""

from __future__ import annotations

import argparse
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
logger = logging.getLogger("merge_annual_parquets")

# Canonical 63 CRSP variables and target Polars dtypes
CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "PERMNO": pl.Int64,
    "date": pl.String,
    "NAMEENDT": pl.String,
    "SHRCD": pl.Int64,
    "EXCHCD": pl.Int64,
    "SICCD": pl.String,
    "NCUSIP": pl.String,
    "TICKER": pl.String,
    "COMNAM": pl.String,
    "SHRCLS": pl.String,
    "TSYMBOL": pl.String,
    "NAICS": pl.String,
    "PRIMEXCH": pl.String,
    "TRDSTAT": pl.String,
    "SECSTAT": pl.String,
    "PERMCO": pl.Int64,
    "ISSUNO": pl.Int64,
    "HEXCD": pl.Int64,
    "HSICCD": pl.Int64,
    "CUSIP": pl.String,
    "DCLRDT": pl.String,
    "DLAMT": pl.Float64,
    "DLPDT": pl.String,
    "DLSTCD": pl.Int64,
    "NEXTDT": pl.String,
    "PAYDT": pl.String,
    "RCRDDT": pl.String,
    "SHRFLG": pl.Int64,
    "HSICMG": pl.Int64,
    "HSICIG": pl.Int64,
    "DISTCD": pl.Int64,
    "DIVAMT": pl.Float64,
    "FACPR": pl.Float64,
    "FACSHR": pl.Float64,
    "ACPERM": pl.Int64,
    "ACCOMP": pl.Int64,
    "SHRENDDT": pl.String,
    "NWPERM": pl.Int64,
    "DLRETX": pl.String,
    "DLPRC": pl.Float64,
    "DLRET": pl.String,
    "TRTSCD": pl.Int64,
    "NMSIND": pl.Int64,
    "MMCNT": pl.Int64,
    "NSDINX": pl.Int64,
    "BIDLO": pl.Float64,
    "ASKHI": pl.Float64,
    "PRC": pl.Float64,
    "VOL": pl.Int64,
    "RET": pl.String,
    "BID": pl.Float64,
    "ASK": pl.Float64,
    "SHROUT": pl.Int64,
    "CFACPR": pl.Float64,
    "CFACSHR": pl.Float64,
    "OPENPRC": pl.Float64,
    "NUMTRD": pl.Float64,
    "RETX": pl.String,
    "vwretd": pl.Float64,
    "vwretx": pl.Float64,
    "ewretd": pl.Float64,
    "ewretx": pl.Float64,
    "sprtrn": pl.Float64,
}

MASTER_LOOKUP_COLS = [
    "date",
    "TICKER",
    "TSYMBOL",
    "PERMNO",
    "PERMCO",
    "CUSIP",
    "NCUSIP",
    "SHRCD",
    "EXCHCD",
    "SICCD",
    "NAICS",
    "PRIMEXCH",
    "SHRCLS",
]


def _format_size(size_bytes: int) -> str:
    """Format byte sizes into human-readable units."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:3.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def build_security_master(input_dir: Path, end_year: int = 2024) -> pl.DataFrame:
    """
    Build a comprehensive security master mapping from historical WRDS files.

    Extracts latest entity identifiers (PERMNO, PERMCO, CUSIP, etc.) keyed by
    both primary TICKER and exchange TSYMBOL.
    """
    logger.info("Building Security Master mapping from WRDS panels 2000-%d...", end_year)
    hist_frames: list[pl.DataFrame] = []

    for y in range(2000, end_year + 1):
        fpath = input_dir / f"{y}.parquet"
        if not fpath.exists():
            continue

        df = pl.read_parquet(fpath, columns=MASTER_LOOKUP_COLS).with_columns(
            [
                pl.col("PERMNO").cast(pl.Int64),
                pl.col("PERMCO").cast(pl.Int64),
                pl.col("SHRCD").cast(pl.Int64),
                pl.col("EXCHCD").cast(pl.Int64),
                pl.col("SICCD").cast(pl.String),
                pl.col("NAICS").cast(pl.String),
                pl.col("PRIMEXCH").cast(pl.String),
                pl.col("SHRCLS").cast(pl.String),
            ]
        )
        hist_frames.append(df)

    if not hist_frames:
        raise FileNotFoundError(f"No historical WRDS files found in {input_dir}")

    all_hist = pl.concat(hist_frames).drop_nulls(subset=["PERMNO"])
    info_cols = [
        "PERMNO",
        "PERMCO",
        "CUSIP",
        "NCUSIP",
        "SHRCD",
        "EXCHCD",
        "SICCD",
        "NAICS",
        "PRIMEXCH",
        "SHRCLS",
    ]

    by_ticker = (
        all_hist.drop_nulls(subset=["TICKER"])
        .sort("date")
        .group_by("TICKER")
        .last()
        .select(["TICKER"] + info_cols)
    )
    by_tsymbol = (
        all_hist.drop_nulls(subset=["TSYMBOL"])
        .sort("date")
        .group_by("TSYMBOL")
        .last()
        .select([pl.col("TSYMBOL").alias("TICKER")] + info_cols)
    )

    # Priority: primary TICKER overrides secondary TSYMBOL
    master_lookup = pl.concat([by_tsymbol, by_ticker]).group_by("TICKER").last()
    logger.info("Security master compiled: %d unique ticker mappings.", len(master_lookup))
    return master_lookup


def load_and_standardize_wrds(file_path: Path) -> pl.DataFrame:
    """Load a WRDS Parquet file (2000-2024) and cast to canonical schema."""
    df = pl.read_parquet(file_path)
    exprs = [pl.col(c).cast(t).alias(c) for c, t in CANONICAL_SCHEMA.items()]
    return df.select(exprs)


def load_and_harmonize_crawled(file_path: Path, master_lookup: pl.DataFrame) -> pl.DataFrame:
    """
    Load a crawled Parquet file (2025-2026) and harmonize into canonical CRSP schema.

    Maps:
    - Date -> date (YYYY-MM-DD string)
    - Ticker -> TICKER, TSYMBOL
    - Name -> COMNAM
    - Open/High/Low/Close -> OPENPRC, ASKHI, BIDLO, PRC
    - Volume -> VOL
    - CFACPR -> Close / Adj_Close
    - CFACSHR -> 1.0
    - PERMNO, PERMCO, CUSIP, etc. -> mapped from security master
    - Missing CRSP columns -> null
    """
    df_raw = pl.read_parquet(file_path)

    # Base field harmonization
    df_base = df_raw.with_columns(
        [
            pl.col("Date").dt.to_string("%Y-%m-%d").alias("date"),
            pl.col("Ticker").alias("TICKER"),
            pl.col("Ticker").alias("TSYMBOL"),
            pl.col("Name").alias("COMNAM"),
            pl.col("Open").cast(pl.Float64).alias("OPENPRC"),
            pl.col("High").cast(pl.Float64).alias("ASKHI"),
            pl.col("Low").cast(pl.Float64).alias("BIDLO"),
            pl.col("Close").cast(pl.Float64).alias("PRC"),
            pl.col("Volume").cast(pl.Int64).alias("VOL"),
            (pl.col("Close") / pl.col("Adj_Close")).cast(pl.Float64).alias("CFACPR"),
            pl.lit(1.0).alias("CFACSHR"),
        ]
    )

    # Join security master
    df_joined = df_base.join(master_lookup, on="TICKER", how="left")

    # Assemble all canonical columns with null placeholders where absent
    present_cols = set(df_joined.columns)
    exprs = []
    for col, dtype in CANONICAL_SCHEMA.items():
        if col in present_cols:
            exprs.append(pl.col(col).cast(dtype).alias(col))
        else:
            exprs.append(pl.lit(None, dtype=dtype).alias(col))

    return df_joined.select(exprs)


def merge_annual_parquets(
    input_dir: str | Path = "data/WRDS",
    output_dir: str | Path = "data/WRDS",
    compression: str = "snappy",
    start_year: int = 2000,
    end_year: int = 2026,
) -> tuple[Path, Path]:
    """
    Execute the merge process across all annual Parquet files.

    Returns the paths to the two output files.
    """
    t_start = time.perf_counter()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("RUSSELL 1000 ANNUAL PARQUET CONSOLIDATION")
    logger.info("Input directory:  %s", input_dir.resolve())
    logger.info("Output directory: %s", output_dir.resolve())
    logger.info("Year range:       %d to %d", start_year, end_year)
    logger.info("Compression:      %s", compression)
    logger.info("=" * 70)

    # 1. Build security master for crawled harmonization
    master_lookup = build_security_master(input_dir, end_year=min(end_year, 2024))

    # 2. Load and standardize all annual files
    annual_frames: list[pl.DataFrame] = []
    total_raw_rows = 0

    for year in range(start_year, end_year + 1):
        fpath = input_dir / f"{year}.parquet"
        if not fpath.exists():
            logger.warning("Year file %s does not exist, skipping.", fpath.name)
            continue

        raw_count = pl.scan_parquet(fpath).select(pl.len()).collect().item()
        total_raw_rows += raw_count

        if year <= 2024:
            df = load_and_standardize_wrds(fpath)
            logger.info("[%d] Loaded WRDS file (%d rows, 63 columns)", year, len(df))
        else:
            df = load_and_harmonize_crawled(fpath, master_lookup)
            n_permno = df.filter(pl.col("PERMNO").is_not_null()).shape[0]
            logger.info(
                "[%d] Harmonized crawled file (%d rows, %d with mapped PERMNO, %.1f%%)",
                year,
                len(df),
                n_permno,
                (n_permno / len(df) * 100.0) if len(df) > 0 else 0.0,
            )

        annual_frames.append(df)

    logger.info("Concatenating %d annual datasets...", len(annual_frames))
    combined_df = pl.concat(annual_frames)
    combined_rows = len(combined_df)
    logger.info("Combined dataset rows: %s", f"{combined_rows:,}")

    if combined_rows != total_raw_rows:
        raise ValueError(
            f"Row count mismatch! Raw total {total_raw_rows:,} vs combined {combined_rows:,}"
        )

    # 3. Generate File 1: Order by PERMNO
    logger.info("-" * 70)
    logger.info("Generating File 1: Sorted by PERMNO (PERMNO, TICKER, date)...")
    df_by_permno = combined_df.sort(
        by=[
            pl.col("PERMNO").is_null(),  # False (0) for valid PERMNO, True (1) for nulls
            pl.col("PERMNO"),
            pl.col("TICKER"),
            pl.col("date"),
        ]
    )

    out_permno = output_dir / "russell1000_by_permno.parquet"
    logger.info("Writing %s...", out_permno.name)
    df_by_permno.write_parquet(out_permno, compression=compression)
    size_permno = out_permno.stat().st_size
    logger.info(
        "✓ Wrote %s (%s, %s rows)",
        out_permno.name,
        _format_size(size_permno),
        f"{len(df_by_permno):,}",
    )

    # 4. Generate File 2: Order by Year / Date
    logger.info("-" * 70)
    logger.info("Generating File 2: Sorted by Year (date, PERMNO, TICKER)...")
    df_by_year = combined_df.sort(
        by=[
            pl.col("date"),
            pl.col("PERMNO").is_null(),
            pl.col("PERMNO"),
            pl.col("TICKER"),
        ]
    )

    out_year = output_dir / "russell1000_by_year.parquet"
    logger.info("Writing %s...", out_year.name)
    df_by_year.write_parquet(out_year, compression=compression)
    size_year = out_year.stat().st_size
    logger.info(
        "✓ Wrote %s (%s, %s rows)", out_year.name, _format_size(size_year), f"{len(df_by_year):,}"
    )

    # 5. Validation Assertions
    logger.info("-" * 70)
    logger.info("Validating output integrity...")
    assert len(df_by_permno) == total_raw_rows, "Permno row count validation failed"
    assert len(df_by_year) == total_raw_rows, "Year row count validation failed"
    assert df_by_permno.columns == list(CANONICAL_SCHEMA.keys()), (
        "Permno column order validation failed"
    )
    assert df_by_year.columns == list(CANONICAL_SCHEMA.keys()), (
        "Year column order validation failed"
    )

    min_date = df_by_year["date"].min()
    max_date = df_by_year["date"].max()
    unique_permnos = df_by_permno["PERMNO"].drop_nulls().n_unique()
    unique_tickers = df_by_permno["TICKER"].drop_nulls().n_unique()

    logger.info("Validation passed successfully:")
    logger.info("  Total rows:       %s", f"{total_raw_rows:,}")
    logger.info("  Date span:        %s to %s", min_date, max_date)
    logger.info("  Unique PERMNOs:   %d", unique_permnos)
    logger.info("  Unique TICKERs:   %d", unique_tickers)
    logger.info("  Total columns:    %d", len(CANONICAL_SCHEMA))
    logger.info("Total processing duration: %.2f seconds", time.perf_counter() - t_start)
    logger.info("=" * 70)

    return out_permno, out_year


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge annual Russell 1000 Parquet files into unified panels."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/WRDS"),
        help="Directory containing annual .parquet files (default: data/WRDS)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/Russell1000"),
        help="Directory to save merged .parquet files (default: data/Russell1000)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "zstd", "gzip", "lz4", "uncompressed"],
        help="Parquet compression codec (default: snappy)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2000,
        help="First annual dataset year (default: 2000)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        help="Last annual dataset year (default: 2026)",
    )

    args = parser.parse_args()

    try:
        merge_annual_parquets(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            compression=args.compression,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    except Exception:
        logger.exception("Consolidation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
