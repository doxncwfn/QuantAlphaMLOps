"""Exploratory Data Analysis (EDA) for universe coverage, missingness, liquidity, and returns using Polars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch


def summarize_duckdb_warehouse(
    warehouse_path: Path, start_date: str = "2000-07-01", end_date: str = "2024-12-31"
) -> pl.DataFrame:
    """Summarize raw DuckDB silver.v_prices table coverage as a Polars DataFrame."""
    import duckdb

    with duckdb.connect(str(warehouse_path), read_only=True) as con:
        return con.sql(
            f"""
            SELECT
                COUNT(*)::BIGINT AS total_rows,
                COUNT(DISTINCT security_id)::BIGINT AS unique_securities,
                COUNT(DISTINCT date)::BIGINT AS trading_days,
                MIN(date)::VARCHAR AS min_date,
                MAX(date)::VARCHAR AS max_date,
                AVG(CASE WHEN close IS NULL THEN 1.0 ELSE 0.0 END) AS close_null_rate,
                AVG(CASE WHEN ret_total IS NULL THEN 1.0 ELSE 0.0 END) AS ret_null_rate,
                AVG(CASE WHEN volume IS NULL OR volume <= 0 THEN 1.0 ELSE 0.0 END) AS zero_vol_rate
            FROM silver.v_prices
            WHERE date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
            """
        ).pl()


def summarize_cached_universe(
    cache_path: Path, out_dir: Path | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute EDA metrics on the cached top-1000 34-feature tensor dataset using Polars."""
    pt: dict[str, Any] = torch.load(cache_path, map_location="cpu", weights_only=False)
    cal = pd.to_datetime(pt["calendar"])
    slots = pt["slots"]
    X = pt["X_dense"]
    Y = pt["Y_dense"]
    features = list(pt["FEATURES"])
    dec_idx = pt["decision_indices"]

    active_counts = (slots >= 0).sum(axis=1)
    refresh_churn = []
    prev_set: set[int] = set()
    for t in dec_idx:
        cur = {int(s) for s in slots[t] if s >= 0}
        if prev_set and cur != prev_set:
            added = len(cur - prev_set)
            refresh_churn.append(
                {
                    "date": str(cal[t].date()),
                    "n_stocks": len(cur),
                    "added": added,
                    "turnover_pct": 100.0 * added / max(len(cur), 1),
                }
            )
        prev_set = cur
    churn_df = pl.DataFrame(refresh_churn) if refresh_churn else pl.DataFrame()

    X_dec = X[dec_idx]
    missing_rates = np.isnan(X_dec).mean(axis=(0, 1))
    feat_rows = []
    for fi, fname in enumerate(features):
        vals = X_dec[:, :, fi]
        finite = vals[np.isfinite(vals)]
        feat_rows.append(
            {
                "feature": fname,
                "missing_pct": float(missing_rates[fi] * 100.0),
                "mean": float(np.mean(finite)) if len(finite) else np.nan,
                "std": float(np.std(finite)) if len(finite) else np.nan,
                "p01": float(np.percentile(finite, 1)) if len(finite) else np.nan,
                "p50": float(np.percentile(finite, 50)) if len(finite) else np.nan,
                "p99": float(np.percentile(finite, 99)) if len(finite) else np.nan,
            }
        )
    feat_eda = pl.DataFrame(feat_rows)

    y_dec = Y[dec_idx]
    yearly_rows = []
    years = pd.DatetimeIndex(cal[dec_idx]).year
    for yr in sorted(np.unique(years)):
        mask_yr = years == yr
        y_sub = y_dec[mask_yr]
        finite_y = y_sub[np.isfinite(y_sub)]
        yearly_rows.append(
            {
                "year": int(yr),
                "days": int(mask_yr.sum()),
                "avg_active_stocks": float(active_counts[dec_idx][mask_yr].mean()),
                "mean_1d_ret_bps": float(np.mean(finite_y) * 1e4)
                if len(finite_y)
                else np.nan,
                "cs_disp_pct": float(np.nanmean(np.nanstd(y_sub, axis=1)) * 100.0),
            }
        )
    yearly_eda = pl.DataFrame(yearly_rows)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        feat_eda.write_csv(out_dir / "eda_feature_summary.csv")
        yearly_eda.write_csv(out_dir / "eda_yearly_coverage.csv")
        if not churn_df.is_empty():
            churn_df.write_csv(out_dir / "eda_universe_refresh_churn.csv")
    return feat_eda, yearly_eda
