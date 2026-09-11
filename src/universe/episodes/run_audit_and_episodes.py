"""Main orchestrator for Identity Resolution Audit and Availability Episodes Construction."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from src.universe.episodes.audit import run_identity_audit
from src.universe.episodes.builder import AvailabilityEpisodeBuilder
from src.universe.episodes.config import LOG_FILE_PATH, LOGS_DIR
from src.universe.episodes.report import generate_availability_report


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("availability_episodes")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
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
    logger.info("STARTING IDENTITY AUDIT & AVAILABILITY EPISODES PIPELINE")
    logger.info("=" * 80)
    t_start = time.time()

    try:
        # 1. Independent Identity Audit (Audits A - K)
        audit_results = run_identity_audit(logger)

        # 2. Availability Episodes Construction & Streaming
        builder = AvailabilityEpisodeBuilder(logger)
        builder.load_data()
        builder.build_episodes()
        builder.build_manual_review_queue()
        builder.stream_expected_security_dates()
        builder.save_artifacts()

        # 3. Comprehensive Diagnostic Markdown Report
        generate_availability_report()

        t_elapsed = time.time() - t_start
        logger.info("=" * 80)
        logger.info("IDENTITY AUDIT & AVAILABILITY EPISODES COMPLETED IN %.2f SECONDS.", t_elapsed)
        logger.info("=" * 80)

    except Exception as exc:
        logger.exception("Fatal error during audit and episode construction: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
