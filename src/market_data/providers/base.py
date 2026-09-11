"""Abstract Base Provider for Market Data Acquisition."""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


class BaseMarketDataProvider(ABC):
    name: str = "BASE"

    @abstractmethod
    def fetch_daily_bars(
        self, ticker: str, start_date: str, end_date: str
    ) -> pl.DataFrame | None:
        """Fetches daily OHLCV bars.

        Returns standard schema:
            date: pl.Utf8 ("YYYY-MM-DD")
            open: pl.Float64
            high: pl.Float64
            low: pl.Float64
            close: pl.Float64
            adj_close: pl.Float64
            volume: pl.Float64
        """
