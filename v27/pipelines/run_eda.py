"""Pipeline 1 CLI: Exploratory Data Analysis on cached dataset and optional DuckDB warehouse."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from v27.config import PathConfig
from v27.eda.universe_eda import summarize_cached_universe, summarize_duckdb_warehouse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v27 universe and feature EDA.")
    parser.add_argument("--qh-dir", type=Path, default=PathConfig().qh_v7_dir)
    parser.add_argument("--output-dir", type=Path, default=PathConfig().output_dir)
    parser.add_argument(
        "--from-duckdb",
        action="store_true",
        help="Also query silver.v_prices coverage from DuckDB.",
    )
    args = parser.parse_args()

    out_tab = args.output_dir / "tables"
    cache_path = args.qh_dir / PathConfig.cache_filename
    feat_df, yearly_df = summarize_cached_universe(cache_path, out_dir=out_tab)
    print(
        f"[EDA] Cached universe summary written to {out_tab}: {feat_df.height} features, {yearly_df.height} years."
    )
    with pl.Config(tbl_rows=30):
        print(yearly_df)

    if args.from_duckdb:
        wh_df = summarize_duckdb_warehouse(PathConfig().warehouse_path)
        wh_df.write_csv(out_tab / "eda_duckdb_warehouse.csv")
        print("\n[EDA] DuckDB Warehouse Coverage:")
        print(wh_df)


if __name__ == "__main__":
    main()
