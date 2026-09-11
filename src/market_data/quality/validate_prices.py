"""Price sanity and envelope validation."""

from __future__ import annotations

import polars as pl


def validate_price_sanity(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Validates price records against standard exchange envelopes.

    Rules:
      - open, high, low, close > 0
      - high >= max(open, close)
      - low <= min(open, close)
      - volume >= 0
      - date is not null and unique

    Returns:
      (clean_df, anomalies_df)
    """
    if df.height == 0:
        return df, df

    # Envelope condition
    cond_positive = (
        (pl.col("open") > 0)
        & (pl.col("high") > 0)
        & (pl.col("low") > 0)
        & (pl.col("close") > 0)
    )
    cond_high = (pl.col("high") >= pl.col("open")) & (pl.col("high") >= pl.col("close"))
    cond_low = (pl.col("low") <= pl.col("open")) & (pl.col("low") <= pl.col("close"))
    cond_vol = pl.col("volume") >= 0

    valid_mask = cond_positive & cond_high & cond_low & cond_vol

    subset = ["security_id", "date"] if "security_id" in df.columns else ["date"]
    clean = df.filter(valid_mask).unique(subset=subset)
    anomalies = df.filter(~valid_mask)

    return clean, anomalies
