"""Alpaca Market Data v2 Provider."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import polars as pl
from src.market_data.config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from src.market_data.providers.base import BaseMarketDataProvider

logger = logging.getLogger(__name__)


class AlpacaProvider(BaseMarketDataProvider):
    name: str = "ALPACA"

    def __init__(self, api_key: Optional[str] = ALPACA_API_KEY, secret_key: Optional[str] = ALPACA_SECRET_KEY):
        self.api_key = api_key
        self.secret_key = secret_key
        self.client = None
        if self.api_key and self.secret_key:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
            except Exception as e:
                logger.debug("Alpaca client init error: %s", e)

    def fetch_daily_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pl.DataFrame]:
        if not self.client:
            return None

        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            req = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=TimeFrame.Day,
                start=datetime.strptime(start_date, "%Y-%m-%d"),
                end=datetime.strptime(end_date, "%Y-%m-%d"),
                feed=DataFeed.IEX
            )
            bars = self.client.get_stock_bars(req)
            if bars.df.empty:
                return None

            df = bars.df.reset_index()
            df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
            records = []
            for _, r in df.iterrows():
                records.append({
                    "date": r["date"],
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "adj_close": float(r["close"]),
                    "volume": float(r["volume"])
                })
            return pl.DataFrame(records)
        except Exception as exc:
            logger.debug("Alpaca fetch failed for %s: %s", ticker, exc)
            return None
