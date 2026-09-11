"""Multi-source batch market data acquisition."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.market_data.config import RAW_MARKET_DIR
from src.market_data.providers.alpaca_provider import AlpacaProvider
from src.market_data.providers.base import BaseMarketDataProvider
from src.market_data.providers.polygon_provider import PolygonProvider
from src.market_data.providers.stooq_provider import StooqProvider
from src.market_data.providers.twelvedata_provider import TwelveDataProvider
from src.market_data.providers.yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)


class BatchAcquisitionEngine:
    def __init__(self, raw_dir: Path = RAW_MARKET_DIR):
        self.raw_dir = raw_dir
        self.providers: dict[str, BaseMarketDataProvider] = {
            "YFINANCE": YFinanceProvider(),
            "POLYGON": PolygonProvider(),
            "ALPACA": AlpacaProvider(),
            "TWELVEDATA": TwelveDataProvider(),
            "STOOQ": StooqProvider(),
        }

    def fetch_for_queue_item(
        self,
        security_id: str,
        ticker: str,
        start_date: str,
        end_date: str,
        provider_preference: list[str] | None = None,
    ) -> tuple[str, pl.DataFrame] | None:
        """Fetches daily bars for a security across preferred providers."""
        if provider_preference is None:
            provider_preference = [
                "YFINANCE",
                "POLYGON",
                "ALPACA",
                "STOOQ",
                "TWELVEDATA",
            ]

        # Check existing raw files
        for p_name in provider_preference:
            raw_path = self.raw_dir / p_name.lower() / security_id / "data.parquet"
            if raw_path.exists():
                try:
                    df = pl.read_parquet(raw_path)
                    if df.height > 0:
                        return p_name, df
                except (
                    OSError,
                    pl.exceptions.PolarsError,
                    RuntimeError,
                    ValueError,
                ) as err:
                    logger.debug("Failed reading cache %s: %s", raw_path, err)

        # Query provider sequence
        for p_name in provider_preference:
            provider = self.providers.get(p_name)
            if not provider:
                continue

            try:
                df = provider.fetch_daily_bars(ticker, start_date, end_date)
                if df is not None and df.height > 0:
                    # Save raw response
                    save_dir = self.raw_dir / p_name.lower() / security_id
                    save_dir.mkdir(parents=True, exist_ok=True)
                    raw_path = save_dir / "data.parquet"
                    df.write_parquet(raw_path)
                    return p_name, df
            except (
                requests.RequestException,
                OSError,
                ValueError,
                RuntimeError,
            ) as exc:
                logger.debug(
                    "Provider %s failed for %s (%s): %s",
                    p_name,
                    security_id,
                    ticker,
                    exc,
                )

        return None
