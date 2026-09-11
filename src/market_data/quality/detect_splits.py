"""Corporate actions and split jump detection."""

from __future__ import annotations

import polars as pl


def detect_price_splits(df: pl.DataFrame, threshold_low: float = 0.55, threshold_high: float = 1.85) -> pl.DataFrame:
    """Detects likely stock splits and reverse splits based on unadjusted close price jumps.

    Flags ratio jumps:
      - 2:1, 3:1 forward splits (close drops to ~0.50, ~0.33)
      - 1:2, 1:10 reverse splits (close jumps to ~2.0, ~10.0)
    """
    if df.height < 2 or "close" not in df.columns:
        return pl.DataFrame(schema={
            "security_id": pl.Utf8,
            "date": pl.Utf8,
            "prev_date": pl.Utf8,
            "close": pl.Float64,
            "prev_close": pl.Float64,
            "jump_ratio": pl.Float64,
            "split_type": pl.Utf8
        })

    if "security_id" in df.columns:
        sorted_df = df.sort(["security_id", "date"])
        lagged = sorted_df.with_columns([
            pl.col("date").shift(1).over("security_id").alias("prev_date"),
            pl.col("close").shift(1).over("security_id").alias("prev_close"),
        ]).filter(pl.col("prev_close").is_not_null() & (pl.col("prev_close") > 0))
    else:
        sorted_df = df.sort("date")
        lagged = sorted_df.with_columns([
            pl.col("date").shift(1).alias("prev_date"),
            pl.col("close").shift(1).alias("prev_close"),
        ]).filter(pl.col("prev_close").is_not_null() & (pl.col("prev_close") > 0))

    with_ratio = lagged.with_columns(
        (pl.col("close") / pl.col("prev_close")).alias("jump_ratio")
    )

    splits = with_ratio.filter(
        (pl.col("jump_ratio") <= threshold_low) | (pl.col("jump_ratio") >= threshold_high)
    ).with_columns(
        pl.when(pl.col("jump_ratio") <= 0.6)
        .then(pl.lit("FORWARD_SPLIT_SUSPECT"))
        .otherwise(pl.lit("REVERSE_SPLIT_SUSPECT"))
        .alias("split_type")
    )

    cols = ["security_id", "date", "prev_date", "close", "prev_close", "jump_ratio", "split_type"]
    return splits.select([c for c in cols if c in splits.columns])
