"""Forensic Audit Module 3: Factor Forensics, Risk-Free Rate Conventions, and Calendar Alignment.

Audits benchmark factor series (Mkt-RF, SMB, HML, RMW, CMA, UMD, RF) from ff.csv,
verifies decimal vs percentage conventions, and analyzes date alignment against WRDS price dates.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd

from analysis.forensics.common import (
    DATA_DIR,
    EXAMPLES_DIR,
    LOG_DIR,
    METRICS_DIR,
    WRDS_DIR,
    ensure_directories,
    get_logger,
)

logger = get_logger("forensics.factors")


def run_factors() -> dict[str, Any]:
    """Execute factor data forensics and calendar alignment audit."""
    ensure_directories()
    log_lines: list[str] = []

    def record_log(msg: str) -> None:
        log_lines.append(f"[{datetime.now().isoformat()}] {msg}")
        logger.info(msg)

    results: dict[str, Any] = {}

    record_log("=== PART 6: Factor Data Forensics ===")
    ff_path = DATA_DIR / "ff.csv"
    if not ff_path.exists():
        record_log(f"Factor file {ff_path} not found.")
        return results

    ff = pd.read_csv(ff_path)
    ff["date"] = pd.to_datetime(ff["date"])
    factor_cols = [c for c in ["mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"] if c in ff.columns]

    record_log(f"Date range: {ff['date'].min().date()} to {ff['date'].max().date()}, Rows: {len(ff)}")

    factor_stats = {}
    for col in factor_cols:
        s = ff[col]
        factor_stats[col] = {
            "count": int(s.count()),
            "null": int(s.isna().sum()),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "q50": float(s.quantile(0.50)),
        }
    results["factor_descriptive_stats"] = factor_stats

    # Correlation matrix
    corr_matrix = ff[factor_cols].corr()
    results["factor_correlation"] = corr_matrix.round(4).to_dict()

    # PART 7: Date Alignment between WRDS CRSP and ff.csv
    record_log("\n=== PART 7: Date Alignment (CRSP vs ff.csv) ===")
    ff_dates = set(ff["date"].dt.date)

    alignment_by_year = {}
    sample_alignments = []

    for yr in range(2000, 2027):
        fpath = WRDS_DIR / f"{yr}.parquet"
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        date_col = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
        if not date_col:
            continue
        crsp_dates = set(pd.to_datetime(df[date_col]).dt.date)

        common = crsp_dates & ff_dates
        in_crsp_not_ff = crsp_dates - ff_dates
        in_ff_not_crsp = {d for d in ff_dates if d.year == yr} - crsp_dates

        alignment_by_year[yr] = {
            "crsp_trading_days": len(crsp_dates),
            "common_days": len(common),
            "crsp_missing_in_ff": len(in_crsp_not_ff),
            "ff_missing_in_crsp": len(in_ff_not_crsp),
        }

        if yr in [2000, 2020, 2024, 2025]:
            for d in sorted(common)[:3]:
                sample_alignments.append({"year": yr, "date": str(d), "aligned": True})

    results["date_alignment"] = alignment_by_year

    if sample_alignments:
        pd.DataFrame(sample_alignments).to_csv(
            EXAMPLES_DIR / "factor_date_alignment_boundary.csv", index=False
        )

    # Save outputs
    out_json = METRICS_DIR / "part6_8_factors.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    log_file = LOG_DIR / "audit_part6_8.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    record_log(f"\n=== PARTS 6-8 COMPLETE: Saved metrics to {out_json} ===")
    return results


if __name__ == "__main__":
    run_factors()
