#!/usr/bin/env python3
"""
Convert crawled daily stock .txt files under data/Russell 1000/crawled
into a single consolidated Parquet file matching WRDS field formats.

Schema alignment with WRDS parquet:
- TICKER  : String   (Extracted from filename, e.g. CCC, FI, LNW)
- date    : String   (Converted from 'MMM DD, YYYY' to 'YYYY-MM-DD')
- OPENPRC : Float64  (Open price)
- ASKHI   : Float64  (High price)
- BIDLO   : Float64  (Low price)
- PRC     : Float64  (Close price)
- VOL     : Int64    (Trading volume)
- RET     : Float64  (Daily return from 'Change' percentage)
- RETX    : Float64  (Return without dividends, equals RET here)
"""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("crawled_to_parquet")


def parse_crawled_txt(txt_path: Path) -> list[dict[str, Any]]:
    """Parse a single crawled text file into records matching WRDS schema."""
    ticker = txt_path.stem.upper()
    records = []

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().strip()
        if not header_line:
            return records

        headers = [h.strip() for h in header_line.split("\t")]
        expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not expected_cols.issubset(set(headers)):
            logger.warning(
                "File %s does not contain expected columns: %s", txt_path.name, headers
            )
            return records

        col_idx = {h: i for i, h in enumerate(headers)}

        for line_no, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < len(headers):
                continue

            try:
                date_raw = parts[col_idx["Date"]].strip()
                date_formatted = (
                    datetime.strptime(date_raw, "%b %d, %Y")
                    .replace(tzinfo=UTC)
                    .strftime("%Y-%m-%d")
                )

                open_prc = float(parts[col_idx["Open"]].replace(",", "").strip())
                high_prc = float(parts[col_idx["High"]].replace(",", "").strip())
                low_prc = float(parts[col_idx["Low"]].replace(",", "").strip())
                close_prc = float(parts[col_idx["Close"]].replace(",", "").strip())

                vol_str = parts[col_idx["Volume"]].replace(",", "").strip()
                vol = int(vol_str) if vol_str and vol_str != "-" else 0

                # Return calculation from 'Change' column if present
                ret = 0.0
                if "Change" in col_idx:
                    chg_raw = parts[col_idx["Change"]].strip()
                    if chg_raw and chg_raw != "-":
                        ret = float(chg_raw.replace("%", "").strip()) / 100.0

                records.append(
                    {
                        "TICKER": ticker,
                        "date": date_formatted,
                        "OPENPRC": open_prc,
                        "ASKHI": high_prc,
                        "BIDLO": low_prc,
                        "PRC": close_prc,
                        "VOL": vol,
                        "RET": ret,
                        "RETX": ret,
                    }
                )
            except (ValueError, KeyError, IndexError) as e:
                logger.error(
                    "Error parsing %s line %d: %s (line: %r)",
                    txt_path.name,
                    line_no,
                    e,
                    line,
                )

    return records


def convert_crawled_txt_to_parquet(
    input_dir: Path,
    output_parquet: Path,
    compression: str = "snappy",
) -> pl.DataFrame:
    """Read all .txt files under input_dir and write out a single consolidated Parquet file."""
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    logger.info(
        "Found %d text files in %s: %s",
        len(txt_files),
        input_dir,
        [f.name for f in txt_files],
    )

    all_records = []
    for tf in txt_files:
        recs = parse_crawled_txt(tf)
        logger.info(
            "Parsed %s: %d rows (Ticker: %s)", tf.name, len(recs), tf.stem.upper()
        )
        all_records.extend(recs)

    # Build schema-enforced Polars DataFrame
    schema = {
        "TICKER": pl.String,
        "date": pl.String,
        "OPENPRC": pl.Float64,
        "ASKHI": pl.Float64,
        "BIDLO": pl.Float64,
        "PRC": pl.Float64,
        "VOL": pl.Int64,
        "RET": pl.Float64,
        "RETX": pl.Float64,
    }

    df = pl.DataFrame(all_records, schema=schema)
    # Sort chronologically by (TICKER, date) matching WRDS standard
    df = df.sort(["TICKER", "date"])

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_parquet, compression=compression)
    logger.info("Successfully wrote %d rows to %s", len(df), output_parquet)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate crawled .txt stock data into a single WRDS-formatted Parquet file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/Russell 1000/crawled"),
        help="Directory containing crawled .txt files (default: data/Russell 1000/crawled)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/Russell 1000/crawled/crawled.parquet"),
        help="Output Parquet destination (default: data/Russell 1000/crawled/crawled.parquet)",
    )
    parser.add_argument(
        "-c",
        "--compression",
        choices=["snappy", "zstd", "gzip", "lz4", "uncompressed"],
        default="snappy",
        help="Compression codec (default: snappy)",
    )
    args = parser.parse_args()

    df = convert_crawled_txt_to_parquet(
        input_dir=args.input_dir,
        output_parquet=args.output,
        compression=args.compression,
    )

    print("\n" + "=" * 60)
    print("CRAWLED DATA CONSOLIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Rows Saved : {len(df):,}")
    print(f"Tickers Included : {df['TICKER'].unique().to_list()}")
    print(f"Date Range       : {df['date'].min()} to {df['date'].max()}")
    print(
        f"Output File      : {args.output} ({args.output.stat().st_size / 1024:.1f} KB)"
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
