"""Sources: where the daily price panel comes from.

Contract: fn(warehouse: Path, start: date, end: date, **kwargs) -> pl.DataFrame with
columns security_id (Int64), date (Date), open, high, low, close, volume, adj_close, ret_total.
One row per security per trading day it has a record; prices are raw (not split-adjusted),
adj_close is split-adjusted, ret_total includes delisting returns.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import polars as pl

from qam.registry import register

PANEL_COLUMNS = ["security_id", "date", "open", "high", "low", "close", "volume", "adj_close", "ret_total"]


@register("source", "warehouse_duckdb")
def warehouse_duckdb(warehouse: Path, start: dt.date, end: dt.date, view: str = "silver.v_prices") -> pl.DataFrame:
    """Read silver.v_prices from the shared warehouse file (read-only)."""
    cols = ", ".join(PANEL_COLUMNS)
    with duckdb.connect(str(warehouse), read_only=True) as con:
        frame = con.execute(
            f"SELECT {cols} FROM {view} WHERE date BETWEEN ? AND ? ORDER BY security_id, date",
            [start, end],
        ).pl()
    return frame.with_columns(pl.col("security_id").cast(pl.Int64), pl.col("date").cast(pl.Date))
