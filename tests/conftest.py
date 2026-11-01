"""Synthetic data only (never real CRSP data in tests)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

N_STOCKS = 30
DELISTED = 29  # last record on day DELIST_DAY, with a -50% delisting return
DELIST_DAY = 300
NEW_LISTING = 28  # first record on day 200
GAPPY = 27  # no records on days 250..259


def make_panel(seed: int = 0, start: str = "2019-01-01", days: int = 520) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days)
    rows = []
    for s in range(N_STOCKS):
        sid = 10_000 + s
        rets = rng.normal(0.0003, 0.02, size=days)
        price = 50.0 * np.exp(np.cumsum(np.log1p(rets)))
        volume = (s + 1) * 1e5 * rng.uniform(0.8, 1.2, size=days)
        for d in range(days):
            if s == DELISTED and d > DELIST_DAY:
                break
            if s == NEW_LISTING and d < 200:
                continue
            if s == GAPPY and 250 <= d < 260:
                continue
            ret = rets[d] if d > 0 else None  # missing return is NULL, as in the warehouse
            if s == DELISTED and d == DELIST_DAY:
                ret = -0.5
            rows.append((sid, dates[d].date(), price[d], price[d], price[d], price[d], volume[d], price[d], ret))
    return pl.DataFrame(
        rows,
        schema={
            "security_id": pl.Int64,
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "adj_close": pl.Float64,
            "ret_total": pl.Float64,
        },
        orient="row",
    )


@pytest.fixture(scope="session")
def panel() -> pl.DataFrame:
    return make_panel()


@pytest.fixture(scope="session")
def warehouse(tmp_path_factory, panel) -> Path:
    path = tmp_path_factory.mktemp("wh") / "synthetic.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE SCHEMA silver")
        con.register("p", panel.to_arrow())
        con.execute("CREATE TABLE silver.v_prices AS SELECT * FROM p")
    return path


def experiment_dict(warehouse: Path, root: Path) -> dict:
    return {
        "name": "synthetic",
        "seed": 7,
        "paths": {"warehouse": str(warehouse), "data_root": str(root / "data"), "runs_root": str(root / "runs")},
        "data_spec": {
            "dataset_name": "synthetic",
            "source": {"func": "warehouse_duckdb"},
            "period": {"start": "2019-04-01", "end": "2020-09-30"},
            "warmup_days": 120,
        },
        "feature_spec": [
            {"func": "log_ret"},
            {"func": "vol", "kwargs": {"n": 10}},
            {"func": "dist_sma", "kwargs": {"n": 10}},
        ],
        "filter_spec": {"func": "topk_dollar_volume", "kwargs": {"k": 12, "buffer_ratio": 0.25, "x": 5, "y": 10, "ttl": 10}},
        "indicator_spec": [{"func": "equal_weight_return"}],
        "label_spec": {"func": "forward_return", "kwargs": {"h": 5}},
        "window_spec": {
            "lookback": 20,
            "max_missing": 2,
            "fill": "zero",
            "transforms": [{"func": "cs_zscore"}, {"func": "clip", "kwargs": {"lo": -3, "hi": 3}}],
        },
        "model_spec": {"func": "hybrid_lstm_tfm", "kwargs": {"hidden_dim": 16, "lstm_layers": 1, "tfm_layers": 1, "heads": 2, "ff_dim": 16, "dropout": 0.0}},
        "loss_spec": {"func": "neg_pearson_rank"},
        "split_spec": {"func": "purged_walk_forward", "kwargs": {"n_folds": 2, "purge": 5, "embargo": 5, "val_frac": 0.2}},
        "train_spec": {"max_epochs": 1, "batch_dates": 16, "min_valid_slots": 5, "accelerator": "cpu"},
        "evaluation_spec": {
            "metrics": [{"func": "mean_ic", "kwargs": {"min_stocks": 5}}, {"func": "decile_spread", "kwargs": {"n": 3, "min_stocks": 5}}]
        },
    }


@pytest.fixture()
def config_file(tmp_path, warehouse) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment_dict(warehouse, tmp_path)))
    return path


def date_of(i: int, start: str = "2019-01-01") -> dt.date:
    return pd.bdate_range(start, periods=i + 1)[-1].date()


def edit_spec(path: Path, changes: dict[str, Any], name: str) -> Path:
    """Write a copy of the spec at `path` with dotted keys changed, next to it as <name>.yaml.
    Tests only: real runs take every parameter from a spec file."""
    data = yaml.safe_load(path.read_text())
    for dotted, value in changes.items():
        *parents, leaf = dotted.split(".")
        node = data
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    out = path.with_name(f"{name}.yaml")
    out.write_text(yaml.safe_dump(data))
    return out
