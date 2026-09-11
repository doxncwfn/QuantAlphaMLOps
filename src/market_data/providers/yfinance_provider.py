"""Yahoo Finance Market Data Provider."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
import yfinance as yf

from src.market_data.config import YFINANCE_CACHE_DIR
from src.market_data.providers.base import BaseMarketDataProvider

logger = logging.getLogger(__name__)


class YFinanceProvider(BaseMarketDataProvider):
    name: str = "YFINANCE"

    def __init__(self, cache_dir: Path = YFINANCE_CACHE_DIR):
        self.cache_dir = cache_dir

    def fetch_daily_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pl.DataFrame]:
        tk_clean = ticker.strip().upper()
        
        # 1. Check existing local yfinance_cache
        cache_file = self.cache_dir / f"{tk_clean}.parquet"
        if cache_file.exists():
            try:
                df = pl.read_parquet(cache_file)
                if df.height > 0:
                    col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
                    df = df.rename(col_map)
                    if "date" in df.columns and df["date"].dtype in (pl.Datetime, pl.Date):
                        df = df.with_columns(pl.col("date").dt.strftime("%Y-%m-%d"))

                    filtered = df.filter(
                        (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
                    )
                    if filtered.height > 0:
                        return filtered.select([
                            pl.col("date").cast(pl.Utf8),
                            pl.col("open").cast(pl.Float64),
                            pl.col("high").cast(pl.Float64),
                            pl.col("low").cast(pl.Float64),
                            pl.col("close").cast(pl.Float64),
                            pl.col("adj_close").cast(pl.Float64) if "adj_close" in filtered.columns else pl.col("close").cast(pl.Float64).alias("adj_close"),
                            pl.col("volume").cast(pl.Float64)
                        ])
            except Exception as e:
                logger.warning("Error reading cached yfinance parquet for %s: %s", tk_clean, e)

        # 2. Query yfinance API
        yf_symbol = tk_clean.replace(".", "-")
        try:
            pdf = yf.download(
                yf_symbol,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False
            )
            if pdf is None or pdf.empty:
                return None

            if isinstance(pdf.columns, pd.MultiIndex):
                pdf.columns = pdf.columns.get_level_values(0)

            pdf = pdf.reset_index()
            pdf = pdf.dropna(how="all")
            if pdf.empty:
                return None

            date_col = "Date" if "Date" in pdf.columns else pdf.columns[0]
            pdf["date"] = pd.to_datetime(pdf[date_col]).dt.strftime("%Y-%m-%d")

            cols_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume"
            }
            pdf = pdf.rename(columns=cols_map)
            
            if "adj_close" not in pdf.columns:
                pdf["adj_close"] = pdf["close"]

            df = pl.from_pandas(pdf[["date", "open", "high", "low", "close", "adj_close", "volume"]])
            return df

        except Exception as exc:
            logger.warning("Error fetching yfinance bars for %s: %s", tk_clean, exc)
            return None
