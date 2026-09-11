"""Configuration settings for Security-Level Market Data Layer."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Input paths
SPELLS_CSV_PATH = DATA_DIR / "universe" / "spells.csv"
SECURITY_MASTER_PARQUET = DATA_DIR / "identity" / "security_master.parquet"
TICKER_HISTORY_PARQUET = DATA_DIR / "identity" / "ticker_history.parquet"
AVAILABILITY_EPISODES_PARQUET = DATA_DIR / "universe" / "availability_episodes.parquet"
EXPECTED_SECURITY_DATES_PARQUET = (
    DATA_DIR / "universe" / "expected_security_dates.parquet"
)
YFINANCE_CACHE_DIR = DATA_DIR / "quality" / "yfinance_cache"

# Output directories
MARKET_DIR = DATA_DIR / "market"
RAW_MARKET_DIR = DATA_DIR / "raw" / "market"
QUALITY_DIR = DATA_DIR / "quality"

# Deliverable paths
MARKET_DATA_QUEUE_PARQUET = MARKET_DIR / "market_data_queue.parquet"
MARKET_DATA_QUEUE_CSV = MARKET_DIR / "market_data_queue.csv"

SECURITY_DAILY_PRICES_PARQUET = MARKET_DIR / "security_daily_prices.parquet"

MARKET_COVERAGE_PARQUET = QUALITY_DIR / "market_coverage.parquet"
MARKET_COVERAGE_CSV = QUALITY_DIR / "market_coverage.csv"

MISSING_MARKET_DATES_PARQUET = QUALITY_DIR / "missing_market_dates.parquet"
MISSING_MARKET_DATES_CSV = QUALITY_DIR / "missing_market_dates.csv"

SPLIT_DETECTION_PARQUET = QUALITY_DIR / "split_detection.parquet"
SPLIT_DETECTION_CSV = QUALITY_DIR / "split_detection.csv"

PROVIDER_COMPARISON_PARQUET = QUALITY_DIR / "provider_comparison.parquet"
PROVIDER_COMPARISON_CSV = QUALITY_DIR / "provider_comparison.csv"

REPORT_MD_PATH = QUALITY_DIR / "market_coverage_report.md"
LOG_FILE_PATH = LOGS_DIR / "market_data.log"

# Rules and Thresholds
CALENDAR_BUFFER_DAYS: int = 30
CORRUPTED_DATES: list[str] = ["2009-10-29", "2010-03-30", "2010-03-31"]

# API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY") or "IbC9qw1ouX7vSkiyYpGVaDk9jCrk2t_K"
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
