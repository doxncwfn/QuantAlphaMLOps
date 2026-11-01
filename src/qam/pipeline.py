"""The fixed run template: build_dataset -> train -> evaluate -> backtest.

Stages read their behaviour from the config and the registry; nothing here should need
editing to try a new feature, filter, label, model, loss, metric or backtest.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from qam import registry
from qam.config import RunConfig
from qam.data import DatasetStore, resolve_tables, table_configs, table_hashes, trading_calendar
from qam.tracking import RunContext

log = logging.getLogger("qam.pipeline")

STAGES = ("build", "train", "evaluate", "backtest")
# Calendar days of prices loaded after period.end so labels of the last decision dates exist.
LABEL_BUFFER_DAYS = 120


# ---------------------------------------------------------------- build


def build_dataset(cfg: RunConfig, force: bool = False) -> dict[str, str]:
    """Build every table version this config needs that is not stored yet (all, with force).
    Returns table -> version hash. Without the warehouse, only already-built versions work."""
    store = DatasetStore.for_config(cfg)
    if not Path(cfg.paths.warehouse).exists():
        if force:
            raise FileNotFoundError(f"warehouse {cfg.paths.warehouse} not found; cannot rebuild")
        return resolve_tables(cfg, store)
    hashes = table_hashes(cfg)
    specs = table_configs(cfg)
    stale = [t for t in hashes if force or not store.has(t, hashes[t])]
    if not stale:
        log.info("dataset %s is up to date (%s)", cfg.data_spec.dataset_name, store.path)
        return hashes

    data = cfg.data_spec
    load_start = data.period.start - dt.timedelta(days=data.warmup_days)
    load_end = data.period.end + dt.timedelta(days=LABEL_BUFFER_DAYS)
    log.info("loading panel %s .. %s from %s", load_start, load_end, cfg.paths.warehouse)
    source = registry.get("source", data.source.func)
    panel = source(cfg.paths.warehouse, load_start, load_end, **data.source.kwargs).sort(["security_id", "date"])
    calendar = trading_calendar(panel)
    start, end = data.period.start, data.period.end
    log.info("panel: %d rows, %d securities, %d trading days", panel.height, panel["security_id"].n_unique(), len(calendar))

    builders = {
        "features": lambda: _build_features(cfg, panel),
        "universe": lambda: _call("filter", cfg.filter_spec, panel, calendar, start, end),
        "market": lambda: _build_market(cfg, panel, calendar),
        "labels": lambda: _call("label", cfg.label_spec, panel, calendar, start, end),
    }
    date_cols = {"features": "date", "universe": "refresh_date", "market": "date", "labels": "date"}
    for table in stale:
        log.info("building table %s (hash %s)", table, hashes[table])
        frame = builders[table]()
        store.write(table, frame, hashes[table], specs[table], date_cols[table])
        log.info("table %s: %d rows", table, frame.height)
    return hashes


def _call(kind: str, component, *args):
    return registry.get(kind, component.func)(*args, **component.kwargs)


def _build_features(cfg: RunConfig, panel: pl.DataFrame) -> pl.DataFrame:
    exprs = [registry.build("feature", c) for c in cfg.feature_spec]
    names = [e.meta.output_name() for e in exprs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"duplicate feature names: {sorted(duplicates)}")
    out = panel.with_columns([e.over("security_id") for e in exprs])
    return (
        out.select(["date", "security_id", *names])
        .filter(pl.col("date") <= cfg.data_spec.period.end)
        .with_columns([pl.col(n).cast(pl.Float32).fill_nan(None) for n in names])
    )


def _build_market(cfg: RunConfig, panel: pl.DataFrame, calendar: np.ndarray) -> pl.DataFrame:
    market = pl.DataFrame({"date": calendar}).with_columns(pl.col("date").cast(pl.Date))
    for c in cfg.indicator_spec:
        fn = registry.get("indicator", c.func)
        kind = getattr(fn, "kind", "ts")
        if kind not in ("ts", "static"):
            raise ValueError(f"indicator {c.func}: kind must be 'ts' or 'static'")
        column = f"{kind}__{c.kwargs.get('alias', c.func)}"
        params = {k: v for k, v in c.kwargs.items() if k != "alias"}
        values = fn(panel, calendar, **params).select(pl.col("date").cast(pl.Date), pl.col("value").alias(column))
        market = market.join(values, on="date", how="left")
    return market.filter(pl.col("date") <= cfg.data_spec.period.end)


# ---------------------------------------------------------------- evaluate


def evaluate(cfg: RunConfig, run: RunContext, tables: dict[str, str]) -> dict[str, float]:
    preds = pd.read_parquet(run.run_dir / "predictions.parquet")
    labels = DatasetStore.for_config(cfg).read("labels", tables["labels"]).to_pandas()
    preds["date"] = pd.to_datetime(preds["date"]).astype("datetime64[ns]")
    labels["date"] = pd.to_datetime(labels["date"]).astype("datetime64[ns]")
    df = preds.merge(labels, on=["date", "security_id"], how="left").dropna(subset=["label"])
    log.info("evaluating %d labelled predictions over %d dates", len(df), df["date"].nunique())

    results: dict[str, float] = {}
    per_date: dict[str, pd.Series] = {}
    for c in cfg.evaluation_spec.metrics:
        fn = registry.get("metric", c.func)
        key = c.kwargs.get("alias", c.func)
        params = {k: v for k, v in c.kwargs.items() if k != "alias"}
        res = fn(df[["date", "security_id", "score", "label"]], **params)
        results[key] = res.value
        if res.per_date is not None:
            per_date[key] = res.per_date
        for fold, part in df.groupby("fold"):
            run.log_metrics({key: fn(part[["date", "security_id", "score", "label"]], **params).value}, prefix=f"{fold}/test")
    run.log_metrics(results, prefix="test")
    if per_date:
        pd.DataFrame(per_date).rename_axis("date").to_parquet(run.run_dir / "metrics_per_date.parquet")
    for key, value in results.items():
        log.info("test %s = %.5f", key, value)
    return results


# ---------------------------------------------------------------- backtest


def backtest(cfg: RunConfig, run: RunContext) -> None:
    engine = registry.build("backtest", cfg.backtest_spec)
    scores = pd.read_parquet(run.run_dir / "predictions.parquet")[["date", "security_id", "score"]]
    try:
        result = engine.run(scores, lambda: _daily_returns(cfg, scores))
    except NotImplementedError as exc:
        log.warning("backtest skipped: %s", exc)
        run.update_meta(backtest="skipped")
        return
    result.nav.to_frame("nav").to_parquet(run.run_dir / "backtest_nav.parquet")
    run.log_metrics(result.stats, prefix="backtest")
    run.update_meta(backtest="done")


def _daily_returns(cfg: RunConfig, scores: pd.DataFrame) -> pd.DataFrame:
    """Daily total returns of the scored stocks, through the configured source component."""
    source = cfg.data_spec.source
    lo = pd.Timestamp(scores["date"].min()).date()
    hi = (pd.Timestamp(scores["date"].max()) + pd.Timedelta(days=LABEL_BUFFER_DAYS)).date()
    panel = registry.get("source", source.func)(cfg.paths.warehouse, lo, hi, **source.kwargs)
    ids = scores["security_id"].unique().tolist()
    return panel.filter(pl.col("security_id").is_in(ids)).select("date", "security_id", "ret_total").to_pandas()


# ---------------------------------------------------------------- gc


def gc(cfg: RunConfig, apply: bool = False) -> list[tuple[str, str]]:
    """Table versions of this dataset that no run under runs_root and not the current config
    use. Dropped only with apply=True. Runs kept elsewhere (e.g. on Modal) are not seen."""
    store = DatasetStore.for_config(cfg)
    keep = {(t, h) for t, h in table_hashes(cfg).items()} if Path(cfg.paths.warehouse).exists() else set()
    for meta_path in Path(cfg.paths.runs_root).glob("*/meta.json"):
        tables = json.loads(meta_path.read_text()).get("dataset_tables") or {}
        keep |= set(tables.items())
    unused = [(v["table_name"], v["config_hash"]) for v in store.versions() if (v["table_name"], v["config_hash"]) not in keep]
    if apply:
        for table, config_hash in unused:
            store.drop(table, config_hash)
            log.info("dropped %s version %s", table, config_hash)
    return unused


# ---------------------------------------------------------------- run


def run(cfg: RunConfig, stages: tuple[str, ...] = STAGES, run_dir: Path | None = None, force_build: bool = False) -> RunContext:
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; expected {STAGES}")
    from qam.train import train  # torch / lightning are only imported when needed

    context = RunContext(cfg, run_dir)
    with context as ctx:
        # The table versions are fixed once per run; later stages on the same run directory
        # read exactly these, whatever has been built since.
        if "build" in stages:
            tables = build_dataset(cfg, force=force_build)
        else:
            tables = ctx.meta().get("dataset_tables") or resolve_tables(cfg, DatasetStore.for_config(cfg))
        ctx.update_meta(dataset_tables=tables)
        if "train" in stages:
            train(cfg, ctx, tables)
        if "evaluate" in stages:
            evaluate(cfg, ctx, tables)
        if "backtest" in stages:
            backtest(cfg, ctx)
    return context
