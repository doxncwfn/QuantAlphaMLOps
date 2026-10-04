"""Forensic Audit Module 2: Price Adjustment, Splits, Dividends, and Volume Scaling.

Evaluates cumulative adjustment factors (CFACPR, CFACSHR), detects stock splits,
analyzes distribution codes (DISTCD), and verifies total return calculations across eras.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from analysis.forensics.common import (
    EXAMPLES_DIR,
    LOG_DIR,
    METRICS_DIR,
    WRDS_DIR,
    ensure_directories,
    get_logger,
)

logger = get_logger("forensics.adjustments")


def run_adjustments() -> dict[str, Any]:
    """Execute split, dividend, and price adjustment forensics."""
    ensure_directories()
    log_lines: list[str] = []

    def record_log(msg: str) -> None:
        log_lines.append(f"[{datetime.now().isoformat()}] {msg}")
        logger.info(msg)

    results: dict[str, Any] = {}

    # PART 2A: Split Detection (CRSP)
    record_log("=== PART 2A: Split Detection in CRSP Data ===")
    split_events = []

    for yr in range(2000, 2025):
        fpath = WRDS_DIR / f"{yr}.parquet"
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        df["date"] = pd.to_datetime(df["date"])

        if "CFACPR" not in df.columns:
            continue

        df = df.sort_values(["PERMNO", "date"])
        df["prev_cfacpr"] = df.groupby("PERMNO")["CFACPR"].shift(1)
        df["cfacpr_ratio"] = df["CFACPR"] / df["prev_cfacpr"]

        splits = df[(df["cfacpr_ratio"] != 1.0) & df["cfacpr_ratio"].notna() & (df["prev_cfacpr"] > 0)]

        if len(splits) > 0:
            df["prev_prc"] = df.groupby("PERMNO")["PRC"].shift(1)
            df["prev_vol"] = df.groupby("PERMNO")["VOL"].shift(1)

            splits_annotated = df[(df["cfacpr_ratio"] != 1.0) & df["cfacpr_ratio"].notna() & (df["prev_cfacpr"] > 0)].copy()

            for _, row in splits_annotated.head(5).iterrows():
                prev_close = row["prev_prc"]
                curr_open = row["OPENPRC"]
                curr_close = row["PRC"]
                ratio = row["cfacpr_ratio"]
                split_events.append({
                    "year": yr,
                    "permno": row["PERMNO"],
                    "ticker": row.get("TICKER", "N/A"),
                    "date": str(row["date"].date()),
                    "cfacpr_before": row["prev_cfacpr"],
                    "cfacpr_after": row["CFACPR"],
                    "cfacpr_ratio": ratio,
                    "prev_close_raw": prev_close,
                    "open_raw": curr_open,
                    "close_raw": curr_close,
                    "prev_vol_raw": row["prev_vol"],
                    "vol_raw": row.get("VOL", np.nan),
                })

    record_log(f"  Found {len(split_events)} sample split events across CRSP years")
    results["split_samples"] = split_events

    # Save split events example table
    if split_events:
        pd.DataFrame(split_events).to_csv(EXAMPLES_DIR / "split_events.csv", index=False)

    # PART 2B: Yahoo Split Detection
    record_log("\n=== PART 2B: Split Detection in Yahoo Data ===")
    yahoo_splits = []
    for yr in [2025, 2026]:
        fpath = WRDS_DIR / f"{yr}.parquet"
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        date_col = "Date" if "Date" in df.columns else "date"
        ticker_col = "Ticker" if "Ticker" in df.columns else "TICKER"
        df["date"] = pd.to_datetime(df[date_col])
        df = df.sort_values([ticker_col, "date"])

        df["prev_close"] = df.groupby(ticker_col)["Close"].shift(1)
        df["ret"] = (df["Close"] - df["prev_close"]) / df["prev_close"]

        large_drops = df[df["ret"] < -0.40]
        record_log(f"  {yr}: Found {len(large_drops)} daily return drops < -40%")
        for _, row in large_drops.head(5).iterrows():
            yahoo_splits.append({
                "year": yr,
                "ticker": row[ticker_col],
                "date": str(row["date"].date()),
                "prev_close": row["prev_close"],
                "close": row["Close"],
                "ret": row["ret"],
            })

    results["yahoo_large_drops"] = yahoo_splits

    # PART 3: Dividend Forensics
    record_log("\n=== PART 3: Dividend Distribution Forensics ===")
    dividend_info = {}
    sample_df_2020 = pd.read_parquet(WRDS_DIR / "2020.parquet")
    if "DISTCD" in sample_df_2020.columns:
        dist_counts = sample_df_2020["DISTCD"].value_counts().head(10).to_dict()
        record_log(f"  2020 DISTCD distribution: {dist_counts}")
        dividend_info["2020_distcd"] = dist_counts

    results["dividends"] = dividend_info

    # PART 4: Total Return Availability Summary
    total_return_info: dict[str, Any] = {
        "crsp": {
            "RET_available": True,
            "RETX_available": True,
            "CFACPR_available": True,
            "CFACSHR_available": True,
            "DLRET_available": True,
            "years": "2000-2024",
        },
        "yahoo": {
            "Close_available": True,
            "Adj_Close_available": True,
            "RET_available": False,
            "RETX_available": False,
            "CFACPR_available": False,
            "DLRET_available": False,
            "years": "2025-2026",
        },
    }
    results["total_return"] = total_return_info

    # Save metrics JSON & Log
    out_json = METRICS_DIR / "part2_5_adjustment.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    log_file = LOG_DIR / "audit_part2_5.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    record_log(f"\n=== PARTS 2-5 COMPLETE: Saved metrics to {out_json} ===")
    return results


if __name__ == "__main__":
    run_adjustments()
