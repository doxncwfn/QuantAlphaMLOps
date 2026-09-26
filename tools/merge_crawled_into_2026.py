#!/usr/bin/env python3
"""
Merge crawled.parquet into 2026.parquet under data/Russell 1000/crawled.

Matches the exact schema and formatting of 2026.parquet:
- Date      : Datetime(time_unit='ns')
- Ticker    : String
- Name      : String
- Sector    : String
- Open      : Float32
- High      : Float32
- Low       : Float32
- Close     : Float32
- Adj_Close : Float32
- Volume    : Float64

Performs deduplication on ['Date', 'Ticker'] and sorts by ['Date', 'Ticker'].
"""

import logging
import shutil
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("merge_crawled_into_2026")

NAME_MAP = {
    "CCC": "CCC INTELLIGENT SOLUTIONS HOLDINGS",
    "FI": "FISERV INC",
    "LNW": "LIGHT WONDER INC",
}

SECTOR_MAP = {
    "CCC": "Information Technology",
    "FI": "Financials",
    "LNW": "Consumer Discretionary",
}


def merge(
    base_parquet: Path,
    crawled_parquet: Path,
    output_parquet: Path | None = None,
) -> pl.DataFrame:
    if output_parquet is None:
        output_parquet = base_parquet

    logger.info("Reading base parquet: %s", base_parquet)
    df_base = pl.read_parquet(base_parquet)
    initial_rows = len(df_base)

    logger.info("Reading crawled parquet: %s", crawled_parquet)
    df_crawled = pl.read_parquet(crawled_parquet)

    # Transform crawled columns to match df_base schema
    df_crawled_transformed = df_crawled.select(
        [
            pl.col("date")
            .str.to_datetime("%Y-%m-%d")
            .dt.cast_time_unit("ns")
            .alias("Date"),
            pl.col("TICKER").alias("Ticker"),
            pl.col("TICKER").replace(NAME_MAP).alias("Name"),
            pl.col("TICKER").replace(SECTOR_MAP).alias("Sector"),
            pl.col("OPENPRC").cast(pl.Float32).alias("Open"),
            pl.col("ASKHI").cast(pl.Float32).alias("High"),
            pl.col("BIDLO").cast(pl.Float32).alias("Low"),
            pl.col("PRC").cast(pl.Float32).alias("Close"),
            pl.col("PRC").cast(pl.Float32).alias("Adj_Close"),
            pl.col("VOL").cast(pl.Float64).alias("Volume"),
        ]
    )

    # Validate schema alignment
    assert df_base.columns == df_crawled_transformed.columns, "Column names mismatch!"
    for col in df_base.columns:
        assert df_base[col].dtype == df_crawled_transformed[col].dtype, (
            f"Type mismatch for {col}!"
        )

    # Backup base file if writing in-place
    backup_file = base_parquet.with_suffix(".parquet.bak")
    shutil.copy2(base_parquet, backup_file)
    logger.info("Created backup at %s", backup_file)

    try:
        # Concatenate and deduplicate on [Date, Ticker]
        combined = pl.concat([df_base, df_crawled_transformed])
        combined = combined.unique(subset=["Date", "Ticker"], keep="first").sort(
            ["Date", "Ticker"]
        )

        combined.write_parquet(output_parquet, compression="snappy")
        logger.info(
            "Successfully merged: %d -> %d rows (+%d rows) in %s",
            initial_rows,
            len(combined),
            len(combined) - initial_rows,
            output_parquet,
        )

        # Remove backup on success
        backup_file.unlink(missing_ok=True)
        return combined

    except Exception as e:
        logger.error("Error during merge: %s. Restoring from backup...", e)
        if backup_file.exists():
            shutil.copy2(backup_file, base_parquet)
        raise


def main():
    base_file = Path("data/Russell 1000/crawled/2026.parquet")
    crawled_file = Path("data/Russell 1000/crawled/crawled.parquet")

    df = merge(base_file, crawled_file)
    print("\n" + "=" * 60)
    print("MERGE COMPLETION REPORT")
    print("=" * 60)
    print(f"Total Rows in 2026.parquet : {len(df):,}")
    print(f"Total Unique Tickers       : {df['Ticker'].n_unique()}")
    print(f"Date Range                 : {df['Date'].min()} to {df['Date'].max()}")
    print("\nTicker Verification in 2026.parquet:")
    for ticker in ["CCC", "FI", "LNW"]:
        sub = df.filter(pl.col("Ticker") == ticker)
        print(
            f"  - {ticker:4}: {len(sub):3} rows | {sub['Date'].min().strftime('%Y-%m-%d')} to {sub['Date'].max().strftime('%Y-%m-%d')} | Sector: {sub['Sector'][0]}"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
