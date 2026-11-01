"""Dataset build, load_window and a full end-to-end run on the synthetic warehouse."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from qam import pipeline
from qam.config import load_config
from qam.data import DatasetStore, DatasetView, resolve_tables, table_hashes
from tests.conftest import edit_spec


@pytest.fixture()
def built(config_file):
    cfg = load_config(config_file, env=None)
    pipeline.build_dataset(cfg)
    return cfg


def view_of(cfg) -> DatasetView:
    return DatasetView.load(DatasetStore.for_config(cfg), cfg, table_hashes(cfg))


def versions(cfg) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for v in DatasetStore.for_config(cfg).versions():
        out.setdefault(v["table_name"], []).append(v["config_hash"])
    return out


def test_tables_and_manifest(built):
    store = DatasetStore.for_config(built)
    hashes = table_hashes(built)
    assert versions(built) == {t: [h] for t, h in hashes.items()}
    universe = store.read("universe", hashes["universe"])
    assert (universe.group_by("refresh_date").len()["len"] == built.filter_spec.kwargs["k"]).all()
    assert store.read("market", hashes["market"]).columns == ["date", "ts__equal_weight_return"]


def test_only_changed_table_is_rebuilt_and_old_version_is_kept(config_file, built):
    old = table_hashes(built)
    changed = load_config(edit_spec(config_file, {"filter_spec.kwargs.k": 10}, "k10"), env=None)
    new = pipeline.build_dataset(changed)
    assert new["universe"] != old["universe"]
    assert {t: new[t] for t in ("features", "market", "labels")} == {t: old[t] for t in ("features", "market", "labels")}
    after = versions(built)
    assert sorted(after["universe"]) == sorted([old["universe"], new["universe"]])
    assert all(len(after[t]) == 1 for t in ("features", "market", "labels"))
    # The first universe is untouched and still readable.
    store = DatasetStore.for_config(built)
    assert store.read("universe", old["universe"])["slot"].max() == built.filter_spec.kwargs["k"] - 1
    assert store.read("universe", new["universe"])["slot"].max() == 9


def test_experiments_sharing_a_dataset_name_do_not_clobber_each_other(config_file, built):
    other = load_config(edit_spec(config_file, {"feature_spec": [{"func": "log_ret"}]}, "one_feature"), env=None)
    pipeline.build_dataset(other)
    pipeline.build_dataset(built)  # switching back needs no rebuild
    assert len(versions(built)["features"]) == 2
    assert view_of(built).features.shape[2] == 3 and view_of(other).features.shape[2] == 1


def test_without_warehouse_the_stored_versions_are_used(config_file, built, tmp_path):
    gone = str(tmp_path / "missing.duckdb")
    cfg = load_config(edit_spec(config_file, {"paths.warehouse": gone}, "no_warehouse"), env=None)
    store = DatasetStore.for_config(cfg)
    assert resolve_tables(cfg, store) == table_hashes(built)
    assert pipeline.build_dataset(cfg) == table_hashes(built)
    changed = load_config(edit_spec(config_file, {"paths.warehouse": gone, "label_spec.kwargs.h": 3}, "no_warehouse_h3"), env=None)
    with pytest.raises(FileNotFoundError, match="labels"):
        resolve_tables(changed, store)
    with pytest.raises(FileNotFoundError):
        pipeline.build_dataset(cfg, force=True)


def test_gc_drops_only_unreferenced_versions(config_file, built):
    old = table_hashes(built)
    other = load_config(edit_spec(config_file, {"filter_spec.kwargs.k": 10}, "k10"), env=None)
    pipeline.build_dataset(other)
    unused = pipeline.gc(built)  # dry run: the k=10 universe is used by no run and not current
    assert unused == [("universe", table_hashes(other)["universe"])]
    assert len(versions(built)["universe"]) == 2
    pipeline.gc(built, apply=True)
    assert versions(built) == {t: [h] for t, h in old.items()}


def test_load_window_shapes_and_mask(built):
    view = view_of(built)
    i = int(view.decision_indices()[50])
    w = view.load_window(i)
    k, L, F = built.filter_spec.kwargs["k"], built.window_spec.lookback, len(built.feature_spec)
    assert w["x"].shape == (k, L, F) and w["mask"].shape == (k,)
    assert w["x_mkt_ts"].shape == (L, 1)
    assert np.isfinite(w["x"]).all()
    assert np.all(w["x"][~w["mask"]] == 0)
    real = w["x"][w["mask"]]
    assert np.abs(real).max() <= 3.0  # clipped
    np.testing.assert_allclose(real.mean(axis=0), 0, atol=0.05)  # z-scored across the real slots only (clip may shift it slightly)


def test_load_window_is_point_in_time(built):
    view = view_of(built)
    i = int(view.decision_indices()[60])
    before = view.load_window(i)["x"].copy()
    view.features[i:] = 1e6  # tamper with day t and later
    np.testing.assert_array_equal(view.load_window(i)["x"], before)


def test_missing_cell_is_left_out_of_the_zscore_and_set_to_zero(built):
    view = view_of(built)
    L = built.window_spec.lookback
    i = int(view.decision_indices()[60])
    cols = view.slots[i]
    raw = view.features[i - L : i, np.maximum(cols, 0), :]  # [L, k, F]
    complete = (cols >= 0) & ~np.isnan(raw).any(axis=(0, 2))
    s = int(np.flatnonzero(complete)[0])
    view.features[i - 1, cols[s], 2] = np.nan  # dist_sma: level feature, raw 0 would be extreme
    w = view.load_window(i)
    assert w["mask"][s] and w["x"][s, -1, 2] == 0.0
    others = w["mask"].copy()
    others[s] = False
    values = raw[-1, others, 2].astype(np.float64)
    present = ~np.isnan(values)
    expected = np.clip((values[present] - values[present].mean()) / (values[present].std(ddof=1) + 1e-8), -3, 3)
    np.testing.assert_allclose(w["x"][others, -1, 2][present], expected, rtol=1e-4, atol=1e-5)
    assert np.all(w["x"][others, -1, 2][~present] == 0.0)


def test_gappy_stock_is_masked_when_window_has_too_many_missing_days(built):
    from tests.conftest import GAPPY

    view = view_of(built)
    sid = 10_000 + GAPPY
    if sid not in view.ids:
        pytest.skip("gappy stock never selected")
    col = int(np.searchsorted(view.ids, sid))
    gap_end = int(np.searchsorted(view.calendar, np.datetime64(pd.bdate_range("2019-01-01", periods=261)[-1].date())))
    hits = [i for i in view.decision_indices() if i >= gap_end - 2 and col in view.slots[i]]
    for i in hits[:3]:
        slot = int(np.flatnonzero(view.slots[i] == col)[0])
        assert not view.load_window(i)["mask"][slot]


def test_end_to_end_run(config_file):
    cfg = load_config(config_file, env=None)
    ctx = pipeline.run(cfg)
    run_dir = ctx.run_dir
    for name in ("config.resolved.yaml", "meta.json", "metrics.json", "run.log", "predictions.parquet", "metrics_per_date.parquet"):
        assert (run_dir / name).exists(), name
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["status"] == "finished" and meta["backtest"] == "skipped"
    assert set(meta["dataset_tables"]) == {"features", "universe", "market", "labels"}
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert {"test/mean_ic", "test/decile_spread"} <= set(metrics)
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    assert set(preds["fold"]) == {"fold_1", "fold_2"}
    assert preds.groupby("date").size().max() <= cfg.filter_spec.kwargs["k"]

    # Re-running reuses the dataset (no rebuild) and creates a new run directory.
    before = DatasetStore.for_config(cfg).versions()
    second = pipeline.run(cfg, stages=("build",))
    assert second.run_dir != run_dir
    assert DatasetStore.for_config(cfg).versions() == before

    # Building other labels under the same dataset name does not change what this run is
    # evaluated against: evaluate reads the table versions recorded in its meta.json.
    other = load_config(edit_spec(config_file, {"label_spec.kwargs.h": 3}, "h3"), env=None)
    pipeline.build_dataset(other)
    pipeline.run(other, stages=("evaluate",), run_dir=run_dir)
    again = json.loads((run_dir / "metrics.json").read_text())
    assert again["test/mean_ic"] == metrics["test/mean_ic"]
    assert json.loads((run_dir / "meta.json").read_text())["dataset_tables"] == meta["dataset_tables"]


def test_placeholder_backtest_does_not_need_the_warehouse(config_file, tmp_path):
    from qam.tracking import RunContext

    cfg = load_config(edit_spec(config_file, {"paths.warehouse": str(tmp_path / "missing.duckdb")}, "no_warehouse"), env=None)
    with RunContext(cfg) as ctx:
        pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "security_id": [10_001], "score": [0.1]}).to_parquet(ctx.run_dir / "predictions.parquet")
        pipeline.backtest(cfg, ctx)
    assert json.loads((ctx.run_dir / "meta.json").read_text())["backtest"] == "skipped"


# ---------------------------------------------------------------- component code in the hash

MODULE = """
\"\"\"Module docstring.\"\"\"
import polars as pl
from qam.registry import register

SCALE = 2.0


def _helper():
    return pl.col("x") * SCALE


@register("feature", "a")
def a(n: int = 1):
    \"\"\"Docstring of a.\"\"\"
    return _helper().alias("a")


@register("feature", "b")
def b():
    return pl.col("x").alias("b")
"""


def test_component_code_is_part_of_the_fingerprint():
    from qam.data import component_source

    base = component_source(MODULE, "a")
    assert "def b" not in base and "_helper" in base and "SCALE = 2.0" in base
    # Comments, docstrings and formatting do not count.
    cosmetic = MODULE.replace('"""Docstring of a."""', '"""Reworded."""').replace("SCALE = 2.0", "SCALE = 2.0  # comment")
    assert component_source(cosmetic, "a") == base
    # Adding or editing another component does not count.
    assert component_source(MODULE + "\n@register('feature', 'c')\ndef c():\n    return 1\n", "a") == base
    assert component_source(MODULE.replace('pl.col("x").alias("b")', 'pl.col("y").alias("b")'), "a") == base
    # Its own code, a helper or a constant does.
    assert component_source(MODULE.replace('alias("a")', 'alias("a2")'), "a") != base
    assert component_source(MODULE.replace('pl.col("x") * SCALE', 'pl.col("x") + SCALE'), "a") != base
    assert component_source(MODULE.replace("SCALE = 2.0", "SCALE = 3.0"), "a") != base


def test_table_hash_includes_component_code(config_file, monkeypatch):
    from qam import data

    cfg = load_config(config_file, env=None)
    before = table_hashes(cfg)
    real = data.code_fingerprint
    monkeypatch.setattr(data, "code_fingerprint", lambda obj: real(obj) + ("x" if obj.__name__ == "forward_return" else ""))
    after = table_hashes(cfg)
    assert after["labels"] != before["labels"]
    assert {t: after[t] for t in ("features", "universe", "market")} == {t: before[t] for t in ("features", "universe", "market")}
