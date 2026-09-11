"""Master pipeline runner for Security-Level Market Data Layer."""

from __future__ import annotations

import logging
import sys
import time

from src.market_data.acquisition.build_queue import build_market_data_queue
from src.market_data.config import LOG_FILE_PATH, LOGS_DIR
from src.market_data.merge.build_security_ohlcv import SecurityMarketDataConsolidator
from src.market_data.report import generate_market_coverage_report


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("market_data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(LOG_FILE_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def main():
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("STARTING SECURITY-LEVEL MARKET DATA LAYER PIPELINE")
    logger.info("=" * 80)
    t_start = time.time()

    try:
        # 1. Build Acquisition Queue
        logger.info("Step 1: Building prioritized market data acquisition queue...")
        build_market_data_queue()

        # 2. Consolidate Market Data
        logger.info("Step 2: Consolidating market data into security_daily_prices...")
        consolidator = SecurityMarketDataConsolidator()
        consolidator.consolidate()

        # 3. Generate Coverage Report
        logger.info("Step 3: Generating market data coverage diagnostic report...")
        generate_market_coverage_report()

        t_elapsed = time.time() - t_start
        logger.info("=" * 80)
        logger.info("MARKET DATA LAYER PIPELINE COMPLETED IN %.2f SECONDS.", t_elapsed)
        logger.info("=" * 80)

    except Exception:
        logger.exception("Fatal error in market data pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()
