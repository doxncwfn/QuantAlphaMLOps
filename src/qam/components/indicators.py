"""Economic / market-level indicators (context/05: indicator tokens).

Contract: fn(panel, calendar, **kwargs) -> pl.DataFrame[date, value], one row per trading day.
Set fn.kind = "ts" for a time series (embedded by the model over the lookback window) or
"static" for a value used as its own token (the value as of t - 1).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from qam.registry import register


@register("indicator", "equal_weight_return")
def equal_weight_return(panel: pl.DataFrame, calendar: np.ndarray) -> pl.DataFrame:
    """Equal-weighted mean daily log return across the panel. Example indicator."""
    return (
        panel.group_by("date")
        .agg(pl.col("ret_total").log1p().mean().alias("value"))
        .sort("date")
    )


equal_weight_return.kind = "ts"
