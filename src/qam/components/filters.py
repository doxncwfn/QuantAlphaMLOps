"""Filters: which k stocks the model sees, refreshed every `ttl` trading days.

Contract: fn(panel: pl.DataFrame, calendar: np.ndarray[datetime64[D]], start: date, end: date, **params)
-> pl.DataFrame[refresh_date, valid_until, slot, security_id, is_pad] with exactly k rows per refresh_date.
valid_until is exclusive (the next refresh date). Only data strictly before refresh_date may be used.
"""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import polars as pl

from qam.data import dense, security_ids
from qam.registry import register

_AGG = {"sum": np.nansum, "mean": np.nanmean, "median": np.nanmedian}


@register("filter", "topk_dollar_volume")
def topk_dollar_volume(
    panel: pl.DataFrame,
    calendar: np.ndarray,
    start: dt.date,
    end: dt.date,
    k: int,
    buffer_ratio: float = 0.1,
    x: int = 20,
    y: int = 30,
    ttl: int = 21,
    agg: str = "sum",
    on_delist: str = "keep",
) -> pl.DataFrame:
    """Top k by dollar volume with a retention buffer (context/03_filter.md).

    At each refresh date t_j:
      1. starting universe: stocks with data on the last trading day before t_j
      2. eligible: data on at least x of the last y trading days before t_j
      3. liquidity: agg of close * volume over the last x trading days before t_j
      4. T_j = top k + b, b = floor(k * buffer_ratio)
      5. keep incumbents of S_{j-1} that are still in T_j, fill the rest of the k slots by rank
      6. fewer than k eligible: the remaining slots are padding (is_pad = True)
    """
    if y < x:
        raise ValueError("filter: y must be >= x")
    if agg not in _AGG:
        raise ValueError(f"filter: agg must be one of {sorted(_AGG)}")
    if on_delist != "keep":
        # Open question 3 in context/08: refill a slot when its stock stops trading inside the TTL.
        raise NotImplementedError("filter: only on_delist='keep' is implemented")

    ids = security_ids(panel)
    close = dense(panel, "close", calendar, ids)
    volume = dense(panel, "volume", calendar, ids)
    has = ~np.isnan(close)
    dollar = close * volume

    first = int(np.searchsorted(calendar, np.datetime64(start, "D")))
    last = int(np.searchsorted(calendar, np.datetime64(end, "D"), side="right")) - 1
    if first < y:
        raise ValueError(f"filter needs {y} trading days before {start}; increase dataset.warmup_days")
    refresh = list(range(first, last + 1, ttl))
    b = int(k * buffer_ratio)
    aggregate = _AGG[agg]

    rows: list[tuple] = []
    previous: set[int] = set()
    for j, i in enumerate(refresh):
        eligible = has[i - 1] & (has[i - y : i].sum(axis=0) >= x)
        candidates = ids[eligible]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            score = aggregate(dollar[i - x : i][:, eligible], axis=0)
        score = np.where(np.isnan(score), -np.inf, score)
        order = np.lexsort((candidates, -score))  # highest liquidity first, ties by security_id
        top = candidates[order][: k + b]
        kept = [s for s in top if s in previous]
        kept_set = set(kept)
        fill = [s for s in top if s not in kept_set][: k - len(kept)]
        chosen = set(kept) | set(fill)
        selected = [int(s) for s in top if s in chosen]  # slot order = liquidity rank

        refresh_date = calendar[i].astype(dt.date)
        if j + 1 < len(refresh):
            valid_until = calendar[refresh[j + 1]].astype(dt.date)
        else:
            valid_until = calendar[last].astype(dt.date) + dt.timedelta(days=1)
        for slot in range(k):
            if slot < len(selected):
                rows.append((refresh_date, valid_until, slot, selected[slot], False))
            else:
                rows.append((refresh_date, valid_until, slot, None, True))
        previous = set(selected)

    return pl.DataFrame(
        rows,
        schema={
            "refresh_date": pl.Date,
            "valid_until": pl.Date,
            "slot": pl.Int32,
            "security_id": pl.Int64,
            "is_pad": pl.Boolean,
        },
        orient="row",
    )
