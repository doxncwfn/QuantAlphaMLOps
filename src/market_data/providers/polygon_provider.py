"""Polygon / Massive Aggregates Provider."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import polars as pl
import requests

from src.market_data.config import MASSIVE_API_KEY
from src.market_data.providers.base import BaseMarketDataProvider

logger = logging.getLogger(__name__)


class PolygonProvider(BaseMarketDataProvider):
    name: str = "POLYGON"

    def __init__(self, api_key: str = MASSIVE_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()

    def fetch_daily_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pl.DataFrame]:
        if not self.api_key:
            return None

        tk_clean = ticker.strip().upper()
        url = f"https://api.massive.com/v2/aggs/ticker/{tk_clean}/range/1/day/{start_date}/{end_date}"
        params = {"apiKey": self.api_key, "adjusted": "true", "limit": 50000}

        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 403:
                # Subscription limited
                return None
            if resp.status_code != 200:
                return None

            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None

            records = []
            for bar in results:
                # bar: {v, vw, o, c, h, l, t, n}
                ts = bar.get("t", 0) / 1000.0
                dt_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                records.append({
                    "date": dt_str,
                    "open": float(bar.get("o", 0.0)),
                    "high": float(bar.get("h", 0.0)),
                    "low": float(bar.get("l", 0.0)),
                    "close": float(bar.get("c", 0.0)),
                    "adj_close": float(bar.get("c", 0.0)),
                    "volume": float(bar.get("v", 0.0))
                })

            return pl.DataFrame(records)

        except Exception as exc:
            logger.debug("Polygon query for %s failed: %s", tk_clean, exc)
            return None
