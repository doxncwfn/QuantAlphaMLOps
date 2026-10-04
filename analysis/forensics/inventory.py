"""Forensic Audit Module 1: Data Inventory & Quality Checks.

Inventories all datasets, detects schema variations, duplicates, impossible OHLC bounds,
zero/negative prices, and missing observations across WRDS and crawled datasets.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from analysis.forensics.common import (
    DATA_DIR,
    LOG_DIR,
    METRICS_DIR,
    WRDS_DIR,
    ensure_directories,
    get_logger,
)

logger = get_logger("forensics.inventory")


def run_inventory() -> dict[str, Any]:
    """Execute inventory and quality checks across all raw and processed partitions."""
    ensure_directories()
    log_lines: list[str] = []

    def record_log(msg: str) -> None:
        log_lines.append(f"[{datetime.now().isoformat()}] {msg}")
        logger.info(msg)

    results: dict[str, Any] = {}

    # 1A. WRDS yearly files
    record_log("=== PART 1A: WRDS Yearly File Inventory ===")
    wrds_inventory = []

    for yr in range(2000, 2027):
        fpath = WRDS_DIR / f"{yr}.parquet"
        if not fpath.exists():
            record_log(f"  WRDS/{yr}.parquet: NOT FOUND")
            continue
        df = pd.read_parquet(fpath)
        cols = list(df.columns)
        n_rows = len(df)

        date_col = "date" if "date" in cols else ("Date" if "Date" in cols else None)
        if date_col:
            dates = pd.to_datetime(df[date_col])
            date_min = str(dates.min().date())
            date_max = str(dates.max().date())
        else:
            date_min = date_max = "N/A"

        ticker_col = "TICKER" if "TICKER" in cols else ("Ticker" if "Ticker" in cols else None)
        permno_col = "PERMNO" if "PERMNO" in cols else None

        n_tickers = df[ticker_col].nunique() if ticker_col else 0
        n_permnos = df[permno_col].nunique() if permno_col else 0

        info = {
            "year": yr,
            "file": f"{yr}.parquet",
            "rows": n_rows,
            "cols": len(cols),
            "date_range": f"{date_min} to {date_max}",
            "n_tickers": n_tickers,
            "n_permnos": n_permnos,
            "schema_type": "CRSP" if "PERMNO" in cols else "Yahoo",
            "columns": cols,
        }
        wrds_inventory.append(info)
        record_log(
            f"  {yr}: {n_rows:,} rows, {len(cols)} cols, {info['schema_type']}, "
            f"{n_tickers} tickers, {n_permnos} permnos, [{date_min} .. {date_max}]"
        )

    results["wrds_inventory"] = wrds_inventory

    # 1B. Schema consistency
    record_log("\n=== PART 1B: Schema Consistency Across Years ===")
    crsp_schemas: dict[int, list[str]] = {}
    yahoo_schemas: dict[int, list[str]] = {}

    for item in wrds_inventory:
        if item["schema_type"] == "CRSP":
            crsp_schemas[item["year"]] = item["columns"]
        else:
            yahoo_schemas[item["year"]] = item["columns"]

    all_crsp_cols = set()
    for cols in crsp_schemas.values():
        all_crsp_cols.update(cols)

    common_crsp = set.intersection(*(set(c) for c in crsp_schemas.values())) if crsp_schemas else set()
    record_log(f"  CRSP years ({len(crsp_schemas)}): total unique cols={len(all_crsp_cols)}, common to all={len(common_crsp)}")

    for yr, cols in crsp_schemas.items():
        diff = all_crsp_cols - set(cols)
        if diff:
            record_log(f"    {yr} missing: {sorted(diff)}")

    results["schema_consistency"] = {
        "crsp_common_cols": sorted(common_crsp),
        "crsp_total_cols": sorted(all_crsp_cols),
        "yahoo_cols": dict(yahoo_schemas),
    }

    # 1C. Duplicates check
    record_log("\n=== PART 1C: Duplicate Checks ===")
    dup_results = []
    for yr in range(2000, 2027):
        fpath = WRDS_DIR / f"{yr}.parquet"
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        cols = df.columns
        d_info: dict[str, Any] = {"year": yr, "total_rows": len(df)}

        if "PERMNO" in cols and "date" in cols:
            dups = df.duplicated(subset=["PERMNO", "date"]).sum()
            d_info["key"] = "PERMNO+date"
            d_info["duplicates"] = int(dups)
        elif "Ticker" in cols and "Date" in cols:
            dups = df.duplicated(subset=["Ticker", "Date"]).sum()
            d_info["key"] = "Ticker+Date"
            d_info["duplicates"] = int(dups)
        elif "TICKER" in cols and "date" in cols:
            dups = df.duplicated(subset=["TICKER", "date"]).sum()
            d_info["key"] = "TICKER+date"
            d_info["duplicates"] = int(dups)
        else:
            d_info["key"] = "unknown"
            d_info["duplicates"] = -1

        full_dups = df.duplicated().sum()
        d_info["full_row_duplicates"] = int(full_dups)
        dup_results.append(d_info)
        if d_info["duplicates"] > 0:
            record_log(f"  {yr}: {d_info['duplicates']} key duplicates ({d_info['key']}), {full_dups} full dups")
        else:
            record_log(f"  {yr}: CLEAN (0 duplicates)")

    results["duplicates"] = dup_results

    # 1D. Price quality and impossible OHLC
    record_log("\n=== PART 1D: Price & Volume Quality / Impossible OHLC ===")
    quality_results = []
    for yr in range(2000, 2027):
        fpath = WRDS_DIR / f"{yr}.parquet"
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        q: dict[str, Any] = {"year": yr}

        if "PERMNO" in df.columns:
            for col in ["PRC", "OPENPRC", "ASKHI", "BIDLO"]:
                if col in df.columns:
                    s = df[col]
                    q[f"{col}_null"] = int(s.isna().sum())
                    q[f"{col}_zero"] = int((s == 0).sum())
                    q[f"{col}_negative"] = int((s < 0).sum())

            vol = df["VOL"] if "VOL" in df.columns else None
            if vol is not None:
                q["vol_null"] = int(vol.isna().sum())
                q["vol_zero"] = int((vol == 0).sum())
                q["vol_negative"] = int((vol < 0).sum())

            mask_valid = (
                (df["OPENPRC"].notna())
                & (df["PRC"].notna())
                & (df["ASKHI"].notna())
                & (df["BIDLO"].notna())
                & (df["OPENPRC"].abs() > 0)
                & (df["PRC"].abs() > 0)
                & (df["ASKHI"] > 0)
                & (df["BIDLO"] > 0)
            )
            if mask_valid.sum() > 0:
                v = df[mask_valid]
                o = v["OPENPRC"].abs()
                c = v["PRC"].abs()
                h = v["ASKHI"]
                lo = v["BIDLO"]
                h_lt_max_oc = (h < np.maximum(o, c) - 1e-6).sum()
                l_gt_min_oc = (lo > np.minimum(o, c) + 1e-6).sum()
                h_lt_l = (h < lo).sum()
                q["ohlc_valid_rows"] = int(mask_valid.sum())
                q["impossible_h_lt_max_oc"] = int(h_lt_max_oc)
                q["impossible_l_gt_min_oc"] = int(l_gt_min_oc)
                q["impossible_h_lt_l"] = int(h_lt_l)

            record_log(
                f"  {yr} (CRSP): prc_null={q.get('PRC_null', 0)}, prc_neg={q.get('PRC_negative', 0)}, "
                f"H<max(O,C)={q.get('impossible_h_lt_max_oc','N/A')}, H<L={q.get('impossible_h_lt_l','N/A')}"
            )
        else:
            for col in ["Open", "High", "Low", "Close", "Adj_Close"]:
                if col in df.columns:
                    s = df[col]
                    q[f"{col}_null"] = int(s.isna().sum())
                    q[f"{col}_zero"] = int((s == 0).sum())
                    q[f"{col}_negative"] = int((s < 0).sum())

            vol = df["Volume"] if "Volume" in df.columns else None
            if vol is not None:
                q["vol_null"] = int(vol.isna().sum())
                q["vol_zero"] = int((vol == 0).sum())

            mask_valid = (df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)
            mask_valid = mask_valid & df[["Open", "High", "Low", "Close"]].notna().all(axis=1)
            if mask_valid.sum() > 0:
                v = df[mask_valid]
                h_lt_max_oc = (v["High"] < np.maximum(v["Open"], v["Close"]) - 1e-6).sum()
                l_gt_min_oc = (v["Low"] > np.minimum(v["Open"], v["Close"]) + 1e-6).sum()
                h_lt_l = (v["High"] < v["Low"]).sum()
                q["ohlc_valid_rows"] = int(mask_valid.sum())
                q["impossible_h_lt_max_oc"] = int(h_lt_max_oc)
                q["impossible_l_gt_min_oc"] = int(l_gt_min_oc)
                q["impossible_h_lt_l"] = int(h_lt_l)

            record_log(
                f"  {yr} (Yahoo): Open_null={q.get('Open_null','N/A')}, Close_null={q.get('Close_null','N/A')}, "
                f"H<max(O,C)={q.get('impossible_h_lt_max_oc','N/A')}, H<L={q.get('impossible_h_lt_l','N/A')}"
            )

        quality_results.append(q)

    results["quality"] = quality_results

    # 1E. Factor data inventory
    record_log("\n=== PART 1E: Factor Data (ff.csv) Inventory ===")
    ff_path = DATA_DIR / "ff.csv"
    if ff_path.exists():
        ff = pd.read_csv(ff_path)
        ff["date"] = pd.to_datetime(ff["date"])
        ff["dow"] = ff["date"].dt.dayofweek
        weekend = ff[ff["dow"] >= 5]
        factor_inv = {
            "path": "data/ff.csv",
            "rows": len(ff),
            "columns": list(ff.columns),
            "date_range": f"{ff['date'].min().date()} to {ff['date'].max().date()}",
            "n_trading_days": len(ff),
            "duplicated_dates": int(ff["date"].duplicated().sum()),
            "weekend_obs": len(weekend),
            "any_nulls": {col: int(ff[col].isna().sum()) for col in ff.columns if col != "date"},
        }
        record_log(f"  ff.csv: {len(ff)} rows, [{ff['date'].min().date()} .. {ff['date'].max().date()}], weekend_obs={len(weekend)}")
        results["factor_inventory"] = factor_inv

    # 1F. Russell 1000 constituent inventory
    record_log("\n=== PART 1F: Russell 1000 Universe Files ===")
    r1k_path = DATA_DIR / "processed" / "russell1000_all_years.csv"
    if r1k_path.exists():
        r1k_csv = pd.read_csv(r1k_path)
        results["universe"] = {
            "rows": len(r1k_csv),
            "columns": list(r1k_csv.columns),
            "year_range": f"{r1k_csv['Year'].min()}-{r1k_csv['Year'].max()}",
            "tickers_per_year": r1k_csv.groupby("Year")["Ticker"].nunique().to_dict(),
        }
        record_log(f"  russell1000_all_years.csv: {len(r1k_csv)} rows, years={results['universe']['year_range']}")

    # Save outputs
    out_json = METRICS_DIR / "part1_inventory.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    log_file = LOG_DIR / "audit_part1.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    record_log(f"\n=== PART 1 COMPLETE: Saved metrics to {out_json} ===")
    return results


if __name__ == "__main__":
    run_inventory()
