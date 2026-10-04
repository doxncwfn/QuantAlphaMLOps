"""Unified CLI entry point for Russell 1000 forensic research modules."""

from __future__ import annotations

import argparse
import sys
import time

from analysis.forensics.adjustments import run_adjustments
from analysis.forensics.common import ensure_directories, get_logger
from analysis.forensics.delistings import run_delistings
from analysis.forensics.eligibility import run_eligibility
from analysis.forensics.factors import run_factors
from analysis.forensics.features import run_features
from analysis.forensics.inventory import run_inventory
from analysis.forensics.missingness import run_missingness

logger = get_logger("forensics.cli")

TASKS = {
    "inventory": ("Data Inventory & Schema Checks", run_inventory),
    "adjustments": ("Price Adjustments & Split Detection", run_adjustments),
    "factors": ("Factor Forensics & Date Alignment", run_factors),
    "delistings": ("Delistings & Terminal Returns", run_delistings),
    "features": ("Feature Feasibility & Transition", run_features),
    "missingness": ("Missingness Mechanisms & Run Lengths", run_missingness),
    "eligibility": ("Target Definition & Eligibility Reconcile", run_eligibility),
}


def main() -> None:
    """Parse CLI arguments and dispatch forensic research modules."""
    parser = argparse.ArgumentParser(
        description="Russell 1000 Forensic Research & Exploratory Audit Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=["all", *TASKS.keys()],
        default="all",
        help="Specific forensic task to execute (default: all)",
    )

    args = parser.parse_args()
    ensure_directories()

    t0 = time.time()
    logger.info("=== Starting Forensic Research Suite (Task: %s) ===", args.task)

    if args.task == "all":
        for task_name, (description, func) in TASKS.items():
            logger.info("--> Executing [%s]: %s", task_name, description)
            try:
                func()
            except Exception:
                logger.exception("Error executing task %s", task_name)
                sys.exit(1)
    else:
        description, func = TASKS[args.task]
        logger.info("--> Executing [%s]: %s", args.task, description)
        func()

    elapsed = time.time() - t0
    logger.info("=== Forensic Research Suite Completed in %.2f seconds ===", elapsed)


if __name__ == "__main__":
    main()
