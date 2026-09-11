"""TwelveData Market Data Provider."""

from __future__ import annotations

import logging

import polars as pl
import requests

from src.market_data.config import TWELVEDATA_API_KEY
from src.market_data.providers.base import BaseMarketDataProvider

logger = logging.getLogger(__name__)


class TwelveDataProvider(BaseMarketDataProvider):
    name: str = "TWELVEDATA"

    def __init__(self, api_key: str | None = TWELVEDATA_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()

    def fetch_daily_bars(
        self, ticker: str, start_date: str, end_date: str
    ) -> pl.DataFrame | None:
        if not self.api_key:
            return None

        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": ticker,
            "interval": "1day",
            "start_date": start_date,
            "end_date": end_date,
            "apikey": self.api_key,
            "outputsize": 5000,
        }

        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            values = data.get("values", [])
            if not values:
                return None

            records = []
            for v in values:
                records.append(
                    {
                        "date": v["datetime"],
                        "open": float(v["open"]),
                        "high": float(v["high"]),
                        "low": float(v["low"]),
                        "close": float(v["close"]),
                        "adj_close": float(v["close"]),
                        "volume": float(v.get("volume", 0.0)),
                    }
                )
            df = pl.DataFrame(records)
            return df.sort("date")
        except (
            requests.RequestException,
            json.JSONDecodeError,
            ValueError,
            KeyError,
            OSError,
        ) as exc:
            logger.debug("TwelveData fetch failed for %s: %s", ticker, exc)
            return None
