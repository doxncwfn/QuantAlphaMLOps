"""Labels: the realised outcome a ranking is judged against. Used only for training and evaluation.

Contract: fn(panel, calendar, start, end, **kwargs) -> pl.DataFrame[date, security_id, label]
for decision dates in [start, end]. label is NaN when it cannot be computed
(context/04: stocks with no value at t + h are left out of the objective).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from qam.data import dense, security_ids
from qam.registry import register


@register("label", "forward_return")
def forward_return(
    panel: pl.DataFrame,
    calendar: np.ndarray,
    start: dt.date,
    end: dt.date,
    h: int = 5,
    offset: int = 0,
) -> pl.DataFrame:
    """Compounded ret_total over the h trading days [t + offset, t + offset + h).

    The model at decision date t sees data up to t - 1, so offset = 0 means the position is
    opened at the close of t - 1, the timing of the legacy target (features up to close D,
    target close D -> close D + 5). offset = 1 opens it at the close of t. Rows exist for stocks that traded on t - 1. The label is NaN
    if the stock has no record on the last day of the horizon (e.g. it was delisted; the
    delisting return itself is in ret_total) or any record inside the horizon has no return.
    """
    ids = security_ids(panel)
    ret = dense(panel, "ret_total", calendar, ids)
    close = dense(panel, "close", calendar, ids)
    present = dense(panel.with_columns(pl.lit(1.0).alias("_row")), "_row", calendar, ids) == 1.0

    bad = present & np.isnan(ret)
    log_ret = np.where(np.isnan(ret), 0.0, np.log1p(np.where(np.isnan(ret), 0.0, ret)))
    zero = np.zeros((1, len(ids)))
    cum_log = np.vstack([zero, np.cumsum(log_ret, axis=0)])
    cum_bad = np.vstack([zero, np.cumsum(bad, axis=0)])

    first = max(int(np.searchsorted(calendar, np.datetime64(start, "D"))), 1)
    last = int(np.searchsorted(calendar, np.datetime64(end, "D"), side="right")) - 1

    frames = []
    for i in range(first, last + 1):
        alive = ~np.isnan(close[i - 1])
        if not alive.any():
            continue
        lo, hi = i + offset, i + offset + h
        if hi <= len(calendar):
            value = np.expm1(cum_log[hi] - cum_log[lo])
            ok = present[hi - 1] & ~np.isnan(ret[hi - 1]) & ((cum_bad[hi] - cum_bad[lo]) == 0)
            value = np.where(ok, value, np.nan)
        else:
            value = np.full(len(ids), np.nan)
        frames.append(
            pl.DataFrame(
                {
                    "date": np.repeat(calendar[i], alive.sum()).astype("datetime64[D]"),
                    "security_id": ids[alive],
                    "label": value[alive],
                }
            )
        )
    if not frames:
        return pl.DataFrame(schema={"date": pl.Date, "security_id": pl.Int64, "label": pl.Float64})
    out = pl.concat(frames)
    return out.with_columns(pl.col("date").cast(pl.Date), pl.col("security_id").cast(pl.Int64)).fill_nan(None)
