from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from qam import registry
from qam.data import trading_calendar

topk = registry.get("filter", "topk_dollar_volume")
forward_return = registry.get("label", "forward_return")


def toy_panel(volumes: dict[int, list[float]], start="2020-01-01") -> pl.DataFrame:
    """One row per stock and day; close = 1 so dollar volume = volume. NaN volume = no record."""
    days = len(next(iter(volumes.values())))
    dates = [dt.date.fromisoformat(start) + dt.timedelta(days=i) for i in range(days)]
    rows = [
        (sid, dates[d], 1.0, v, 1.0, 0.0)
        for sid, vols in volumes.items()
        for d, v in enumerate(vols)
        if not np.isnan(v)
    ]
    return pl.DataFrame(
        rows,
        schema={"security_id": pl.Int64, "date": pl.Date, "close": pl.Float64, "volume": pl.Float64, "adj_close": pl.Float64, "ret_total": pl.Float64},
        orient="row",
    )


def run(panel, start_day, end_day, **params):
    cal = trading_calendar(panel)
    out = topk(panel, cal, cal[start_day].astype(dt.date), cal[end_day].astype(dt.date), **params)
    return {r: g.sort("slot") for (r,), g in out.group_by("refresh_date", maintain_order=True)}


def test_buffer_keeps_incumbent_and_blocks_newcomer():
    # k=2, b=1 -> top 3 is the tolerance band. Day 0-2: A, B lead. Day 3-5: C overtakes B; B is 3rd.
    vols = {1: [9, 9, 9, 9, 9, 9, 9], 2: [8, 8, 8, 5, 5, 5, 5], 3: [1, 1, 1, 7, 7, 7, 7], 4: [2, 2, 2, 2, 2, 2, 2]}
    periods = run(toy_panel(vols), 3, 6, k=2, buffer_ratio=0.5, x=3, y=3, ttl=3)
    first, second = list(periods.values())
    assert first["security_id"].to_list() == [1, 2]  # uses days 0-2 only
    assert second["security_id"].to_list() == [1, 2]  # B (rank 3 <= k + b) is kept over C (rank 2)


def test_incumbent_leaves_when_outside_band():
    vols = {1: [9] * 7, 2: [8, 8, 8, 1, 1, 1, 1], 3: [1, 1, 1, 7, 7, 7, 7], 4: [2, 2, 2, 6, 6, 6, 6]}
    periods = list(run(toy_panel(vols), 3, 6, k=2, buffer_ratio=0.5, x=3, y=3, ttl=3).values())
    assert periods[1]["security_id"].to_list() == [1, 3]  # B fell to rank 4 > k + b


def test_padding_when_too_few_eligible():
    nan = np.nan
    vols = {1: [5, 5, 5, 5], 2: [4, 4, 4, 4], 3: [nan, nan, 3, 3]}  # stock 3 has 1 of the last 3 days
    (period,) = run(toy_panel(vols), 3, 3, k=4, buffer_ratio=0.0, x=2, y=3, ttl=5).values()
    assert period["is_pad"].to_list() == [False, False, True, True]
    assert period["slot"].to_list() == [0, 1, 2, 3]


def test_filter_is_point_in_time():
    vols = {1: [9, 9, 9, 1], 2: [1, 1, 1, 50], 3: [5, 5, 5, 5]}
    before = run(toy_panel(vols), 3, 3, k=1, buffer_ratio=0.0, x=3, y=3, ttl=5)
    vols[2][3] = 1e9  # a huge volume ON the refresh date must not matter
    after = run(toy_panel(vols), 3, 3, k=1, buffer_ratio=0.0, x=3, y=3, ttl=5)
    assert list(before.values())[0]["security_id"].to_list() == list(after.values())[0]["security_id"].to_list() == [1]


def test_valid_until_is_next_refresh(panel):
    cal = trading_calendar(panel)
    out = topk(panel, cal, cal[100].astype(dt.date), cal[140].astype(dt.date), k=5, x=5, y=10, ttl=10)
    refresh = out["refresh_date"].unique().sort().to_list()
    until = out.group_by("refresh_date").agg(pl.col("valid_until").first()).sort("refresh_date")["valid_until"].to_list()
    assert until[:-1] == refresh[1:]
    assert out.group_by("refresh_date").len()["len"].unique().to_list() == [5]


def test_refill_is_not_implemented_yet(panel):
    cal = trading_calendar(panel)
    with pytest.raises(NotImplementedError):
        topk(panel, cal, cal[100].astype(dt.date), cal[110].astype(dt.date), k=5, on_delist="refill")


# ---------------------------------------------------------------- labels


def returns_panel(rets: list[float | None], sid=1) -> pl.DataFrame:
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(len(rets))]
    price = np.cumprod([1.0] + [1 + (r or 0) for r in rets[1:]])
    return pl.DataFrame(
        {"security_id": [sid] * len(rets), "date": dates, "close": price, "volume": [1.0] * len(rets), "adj_close": price, "ret_total": rets},
        schema_overrides={"ret_total": pl.Float64},
    )


def test_forward_return_compounds_h_days():
    p = returns_panel([None, 0.01, 0.02, -0.03, 0.04, 0.05, 0.06])
    cal = trading_calendar(p)
    out = forward_return(p, cal, cal[2].astype(dt.date), cal[2].astype(dt.date), h=3)
    assert out["label"][0] == pytest.approx(1.02 * 0.97 * 1.04 - 1)


def test_forward_return_offset():
    p = returns_panel([None, 0.01, 0.02, -0.03, 0.04, 0.05, 0.06])
    cal = trading_calendar(p)
    out = forward_return(p, cal, cal[2].astype(dt.date), cal[2].astype(dt.date), h=2, offset=1)
    assert out["label"][0] == pytest.approx(0.97 * 1.04 - 1)


def test_label_is_null_after_delisting(panel):
    from tests.conftest import DELIST_DAY, DELISTED

    cal = trading_calendar(panel)
    out = forward_return(panel, cal, cal[DELIST_DAY - 3].astype(dt.date), cal[DELIST_DAY].astype(dt.date), h=5)
    sid = 10_000 + DELISTED
    labels = out.filter(pl.col("security_id") == sid).sort("date")["label"].to_list()
    assert all(v is None for v in labels)  # horizon passes the last record
    assert out.filter(pl.col("security_id") != sid)["label"].null_count() == 0
