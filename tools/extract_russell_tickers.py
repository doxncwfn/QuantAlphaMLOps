#!/usr/bin/env python3
"""
Extract line-separated tickers from Russell 1000 annual raw and processed files.

Supports:
1. Extracting tickers from annual processed CSV files in data/processed/.
2. Extracting tickers directly from raw source files (PDF, JSON, XLS) in data/raw/.
Writes standardized [year].txt files containing LF-joined ticker symbols.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

import pypdfium2

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


def extract_2003_pdf(pdf_path: Path) -> list[str]:
    """Extract tickers from 2003.pdf (each row: <Ticker> <Company Name>)."""
    pdf = pypdfium2.PdfDocument(str(pdf_path))
    tickers = []
    for page in pdf:
        text = page.get_textpage().get_text_range()
        for line in text.splitlines():
            line = line.strip()
            if (
                not line
                or "Russell 1000" in line
                or line.startswith("Page ")
                or line == "Ticker Name"
            ):
                continue
            parts = line.split()
            if parts and parts[0].strip():
                tickers.append(parts[0].strip())
    return tickers


def extract_2020_pdf(pdf_path: Path) -> list[str]:
    """Extract tickers from 2020.pdf (each row: <Company Name> <Ticker>)."""
    pdf = pypdfium2.PdfDocument(str(pdf_path))
    tickers = []
    num_pages = len(pdf) - 1  # Skip final legal disclaimer page
    for page_idx in range(num_pages):
        page = pdf[page_idx]
        text = page.get_textpage().get_text_range()
        for line in text.splitlines():
            line = line.strip()
            if not line or line == "Company Ticker":
                continue
            if re.search(
                r"(Membership list|Russell US Indexes|Russell 1000|ftserussell\.com|June \d+,\s*\d+)",
                line,
            ):
                continue
            parts = line.split()
            if parts and parts[-1].strip():
                tickers.append(parts[-1].strip())
    return tickers


def extract_2023_json(json_path: Path) -> list[str]:
    """Extract equity tickers from iShares DataTables JSON structure."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("aaData", [])
    tickers = []
    for row in rows:
        if len(row) > 3 and row[3] == "Equity":
            ticker = row[0].strip()
            if ticker and ticker != "-":
                tickers.append(ticker)
    return tickers


def process_year_csv(csv_path: Path, output_dir: Path | None = None) -> tuple[Path, int]:
    """Process a single [year].csv file and write [year].txt."""
    if output_dir is None:
        output_dir = csv_path.parent

    output_path = output_dir / f"{csv_path.stem}.txt"
    tickers = extract_tickers_from_csv(csv_path)
    output_path.write_text("\n".join(tickers), encoding="utf-8")
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
        description="Extract Russell 1000 constituent tickers to line-separated .txt files."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed directory (default: data/processed)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw source files (default: data/raw)",
    )
    parser.add_argument(
        "--extract-raw",
        action="store_true",
        help="Extract from raw PDFs/JSONs (2003, 2020, 2023) directly to processed dir",
    )
    args = parser.parse_args()

    if args.extract_raw:
        targets = [
            ("2003.pdf", extract_2003_pdf, "2003.txt"),
            ("2020.pdf", extract_2020_pdf, "2020.txt"),
            ("2023.json", extract_2023_json, "2023.txt"),
        ]
        for src_name, extractor, dest_name in targets:
            src_p = args.raw_dir / src_name
            dest_p = args.dir / dest_name
            if src_p.exists():
                tickers = extractor(src_p)
                dest_p.write_text("\n".join(tickers), encoding="utf-8")
                logger.info(
                    "Extracted %d tickers from %s -> %s", len(tickers), src_p.name, dest_p.name
                )
        return

    processed_dir = args.dir
    if not processed_dir.is_dir():
        logger.error("Directory not found: %s", processed_dir)
        raise SystemExit(1)

    results = process_all_years(processed_dir)
    logger.info("Completed processing %d files successfully.", len(results))


if __name__ == "__main__":
    main()
