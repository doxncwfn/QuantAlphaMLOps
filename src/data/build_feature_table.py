"""Orchestrator for building the v1.3 feature table (silver.features_daily and Parquet)."""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd

from src.data.build_universe import assign_in_universe
from src.data.compute_features import (
    CONTEXT_FEATURE_COLS,
    STOCK_FEATURE_COLS,
    compute_all_features,
)
from src.data.compute_target import compute_target
from src.data.normalize_schema import load_warehouse_data, seed_everything

OUTPUT_COLUMNS: list[str] = (
    ["permno", "date"]
    + STOCK_FEATURE_COLS
    + CONTEXT_FEATURE_COLS
    + ["target_5d", "y_star", "in_universe", "features_ready", "era"]
)


def build_feature_table(
    warehouse_path: str | Path = "data/warehouse/russell1000.duckdb",
    parquet_dir: str | Path = "data/features",
    write_duckdb: bool = True,
    write_parquet: bool = True,
    start_date: str = "2000-06-30",
    end_date: str = "2026-07-31",
    filter_shrcd: bool = False,
    table_name: str = "silver.features_daily",
) -> pd.DataFrame:
    """Build the complete v1.3 feature table from warehouse inputs.

    Steps:
    1. Seed all RNGs (`seed_everything(42)`).
    2. Load and normalize CRSP (with 360-session pre-2000 warmup) + Yahoo (2025-2026) + FF factors + Russell lists.
    3. Assign point-in-time `in_universe` flag.
    4. Compute Layer 1 (6), Layer 2 (10), Layer 4 (3), and Layer 5 (11) raw features.
    5. Compute `target_5d` (with delisting policy) and `y_star` cross-sectional rank.
    6. Compute `features_ready` flag (`lifetime_session >= 332` and core 60d/272d span features valid).
    7. Filter to `[start_date, end_date]`, verify schema/bounds, and persist to DuckDB and Parquet.
    """
    t0 = time.time()
    seed_everything(42)

    print(f"[1/6] Loading and normalizing warehouse data from {warehouse_path}...")
    stock_df, factors_df, russell_df = load_warehouse_data(
        warehouse_path=warehouse_path, include_yahoo=True
    )
    print(
        f"      Loaded {len(stock_df):,} normalized stock rows across {stock_df['permno'].nunique():,} securities."
    )

    print(
        f"[2/6] Assigning Point-in-Time Russell 1000 universe membership (filter_shrcd={filter_shrcd})..."
    )
    stock_df["in_universe"] = assign_in_universe(
        stock_df, russell_df, filter_shrcd=filter_shrcd, convention="june_30"
    )

    print("[3/6] Computing Layers 1, 2, 4, and 5 features...")
    feat_df = compute_all_features(stock_df, factors_df)

    print("[4/6] Computing forward 5-session target_5d and cross-sectional y_star...")
    feat_df = compute_target(feat_df)

    print("[5/6] Computing features_ready flag and filtering to evaluation window...")
    # features_ready requires 332+ lifetime trading sessions (60-session model window + 272-session
    # max lookback span for resid_mom_20 / resid_vol_20) and valid long-horizon features.
    # Transient single-day NaNs from 20-day split-contamination masks (Risk R8) or null OPENPRC (Risk R5)
    # are handled by cross-sectional normal_scores rank imputation at training time.
    feat_df["features_ready"] = (
        (feat_df["lifetime_session"] >= 332)
        & feat_df["r_cc"].notna()
        & feat_df["ret_60d"].notna()
        & feat_df["vol_60d"].notna()
        & feat_df["resid_ret"].notna()
        & feat_df["resid_mom_20"].notna()
        & feat_df["resid_vol_20"].notna()
        & feat_df["ctx_mktrf"].notna()
    )

    # Filter to target evaluation window [start_date, end_date]
    mask_window = (feat_df["date"] >= pd.Timestamp(start_date)) & (
        feat_df["date"] <= pd.Timestamp(end_date)
    )
    final_df = feat_df.loc[mask_window, OUTPUT_COLUMNS].copy()
    final_df = final_df.sort_values(["permno", "date"]).reset_index(drop=True)

    # Verify schema and integrity
    assert list(final_df.columns) == OUTPUT_COLUMNS, (
        "Column mismatch in final feature table"
    )
    assert not final_df.duplicated(subset=["permno", "date"]).any(), (
        "Duplicate (permno, date) found!"
    )

    univ_rows = final_df.loc[final_df["in_universe"]]
    crsp_univ = univ_rows.loc[univ_rows["era"] == "CRSP"]
    ready_pct_crsp = crsp_univ["features_ready"].mean() * 100.0
    ready_pct_all = univ_rows["features_ready"].mean() * 100.0
    print(
        f"      Final table: {len(final_df):,} rows ({len(univ_rows):,} in-universe rows). "
        f"features_ready: {ready_pct_crsp:.2f}% (CRSP 2000-2024), {ready_pct_all:.2f}% (2000-2026)."
    )

    if write_parquet:
        out_dir = Path(parquet_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[6/6] Writing partitioned Parquet files to {out_dir}...")
        years = final_df["date"].dt.year.unique()
        for yr in sorted(years):
            yr_df = final_df.loc[final_df["date"].dt.year == yr]
            yr_df.to_parquet(out_dir / f"features_{int(yr)}.parquet", index=False)

    if write_duckdb:
        if "." in table_name:
            schema_part, tbl_part = table_name.split(".", 1)
            tbl_clean = tbl_part.strip('"')
            qualified_table = f'{schema_part}."{tbl_clean}"'
        else:
            schema_part = "silver"
            tbl_clean = table_name.strip('"')
            qualified_table = f'"{tbl_clean}"'
        print(f"      Writing {qualified_table} to {warehouse_path}...")
        con = duckdb.connect(str(warehouse_path), read_only=False)
        try:
            con.register("final_features_view", final_df)
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_part}")
            con.execute(
                f"CREATE OR REPLACE TABLE {qualified_table} AS SELECT * FROM final_features_view"
            )
            con.unregister("final_features_view")
        finally:
            con.close()

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s.")
    return final_df


if __name__ == "__main__":
    build_feature_table()
