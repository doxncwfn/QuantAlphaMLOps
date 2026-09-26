"""
Extract line-separated tickers from Russell 1000 annual processed CSV files.

This script reads each [year].csv file in `data/Russell 1000/processed/`
and writes a corresponding [year].txt file containing line-separated tickers,
matching the format of reference files 2000.txt and 2001.txt.
"""

import argparse
import csv
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_tickers_from_csv(csv_path: Path) -> list[str]:
    """Read tickers from a year CSV file while preserving order."""
    tickers = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if "Ticker" not in (reader.fieldnames or []):
            raise ValueError(
                f"Missing 'Ticker' column in {csv_path}. Found headers: {reader.fieldnames}"
            )
        for row in reader:
            ticker = row.get("Ticker", "").strip()
            if ticker:
                tickers.append(ticker)
    return tickers


def process_year_csv(csv_path: Path, output_dir: Path | None = None) -> tuple[Path, int]:
    """
    Process a single [year].csv file and write [year].txt.

    Returns the output path and the number of tickers written.
    """
    if output_dir is None:
        output_dir = csv_path.parent

    output_path = output_dir / f"{csv_path.stem}.txt"
    tickers = extract_tickers_from_csv(csv_path)

    # Format matches reference files 2000.txt and 2001.txt: LF joined without trailing newline
    content = "\n".join(tickers)
    output_path.write_text(content, encoding="utf-8")

    return output_path, len(tickers)


def process_all_years(directory: Path) -> dict[str, int]:
    """Process all [year].csv files in the given directory."""
    year_pattern = re.compile(r"^\d{4}\.csv$")
    csv_files = sorted(f for f in directory.iterdir() if f.is_file() and year_pattern.match(f.name))

    if not csv_files:
        logger.warning("No [year].csv files found in %s", directory)
        return {}

    logger.info("Found %d annual CSV files in %s", len(csv_files), directory)
    results = {}
    for csv_file in csv_files:
        out_path, count = process_year_csv(csv_file)
        results[csv_file.name] = count
        logger.info("Generated %s (%d tickers)", out_path.name, count)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert Russell 1000 [year].csv files to [year].txt ticker lists."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/Russell 1000/processed"),
        help="Path to the directory containing [year].csv files (default: data/Russell 1000/processed)",
    )
    args = parser.parse_args()

    processed_dir = args.dir
    if not processed_dir.is_dir():
        logger.error("Directory not found: %s", processed_dir)
        raise SystemExit(1)

    results = process_all_years(processed_dir)
    logger.info("Completed processing %d files successfully.", len(results))


if __name__ == "__main__":
    main()
