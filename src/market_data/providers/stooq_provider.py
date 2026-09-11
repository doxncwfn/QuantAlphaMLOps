"""Stooq Market Data Provider."""

from __future__ import annotations

import io
import logging
from typing import Optional

import pandas as pd
import polars as pl
import requests

from src.market_data.providers.base import BaseMarketDataProvider

logger = logging.getLogger(__name__)


class StooqProvider(BaseMarketDataProvider):
    name: str = "STOOQ"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    def fetch_daily_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pl.DataFrame]:
        symbol = f"{ticker.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"

        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200 or "Date,Open,High,Low,Close" not in resp.text:
                return None

            pdf = pd.read_csv(io.StringIO(resp.text))
            pdf["date"] = pd.to_datetime(pdf["Date"]).dt.strftime("%Y-%m-%d")
            pdf = pdf[(pdf["date"] >= start_date) & (pdf["date"] <= end_date)]
            if pdf.empty:
                return None

            pdf = pdf.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
            })
            pdf["adj_close"] = pdf["close"]

            df = pl.from_pandas(pdf[["date", "open", "high", "low", "close", "adj_close", "volume"]])
            return df.sort("date")
        except Exception as exc:
            logger.debug("Stooq fetch failed for %s: %s", ticker, exc)
            return None
