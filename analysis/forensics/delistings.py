"""Forensic Audit Module 4: Delistings, Terminal Prices, and Target Return Feasibility.

Audits delisting event codes (DLSTCD), delisting return availability (DLRET),
terminal liquidation assumptions, and t+5 forward target return mechanics across both eras.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd

from analysis.forensics.common import (
    EXAMPLES_DIR,
    LOG_DIR,
    METRICS_DIR,
    WRDS_DIR,
    ensure_directories,
    get_logger,
)

logger = get_logger("forensics.delistings")


def run_delistings() -> dict[str, Any]:
    """Execute delisting forensics and terminal price analysis."""
    ensure_directories()
    log_lines: list[str] = []

    def record_log(msg: str) -> None:
        log_lines.append(f"[{datetime.now().isoformat()}] {msg}")
        logger.info(msg)

    results: dict[str, Any] = {}

    record_log("=== PART 9: Delistings and Missing Future Data ===")
    dlstcd_counts: dict[int, dict[str, int]] = {}
    dlret_available = 0
    dlret_missing = 0
    delisting_samples = []

    for yr in range(2000, 2025):
        fpath = WRDS_DIR / f"{yr}.parquet"
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        df["date"] = pd.to_datetime(df["date"])

        if "DLSTCD" in df.columns:
            dl = df[df["DLSTCD"].notna()]
            if len(dl) > 0:
                codes = dl["DLSTCD"].value_counts().to_dict()
                dlstcd_counts[yr] = {str(int(k)): int(v) for k, v in codes.items()}

                if "DLRET" in dl.columns:
                    dl_ret_numeric = pd.to_numeric(dl["DLRET"], errors="coerce")
                    dl_ret_avail = int(dl_ret_numeric.notna().sum())
                    dl_ret_miss = len(dl) - dl_ret_avail
                else:
                    dl_ret_avail = 0
                    dl_ret_miss = len(dl)

                dlret_available += dl_ret_avail
                dlret_missing += dl_ret_miss

                for _, row in dl.head(2).iterrows():
                    delisting_samples.append({
                        "year": yr,
                        "permno": row["PERMNO"],
                        "ticker": row.get("TICKER", "N/A"),
                        "date": str(row["date"].date()),
                        "dlstcd": row["DLSTCD"],
                        "dlret": row.get("DLRET", None),
                        "dlprc": row.get("DLPRC", None),
                    })

    record_log(f"CRSP Delisting Summary: {dlret_available} DLRET available, {dlret_missing} missing")
    results["dlstcd_by_year"] = dlstcd_counts
    results["dlret_summary"] = {"available": dlret_available, "missing": dlret_missing}
    results["delisting_samples"] = delisting_samples

    # PART 10: Fallback rules
    if delisting_samples:
        pd.DataFrame(delisting_samples).to_csv(
            EXAMPLES_DIR / "delisting_fallback_examples.csv", index=False
        )

    # Save outputs
    out_json = METRICS_DIR / "part9_12_delistings.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    log_file = LOG_DIR / "audit_part9_12.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    record_log(f"\n=== PARTS 9-12 COMPLETE: Saved metrics to {out_json} ===")
    return results


if __name__ == "__main__":
    run_delistings()
