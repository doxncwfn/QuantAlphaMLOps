"""Dataset storage and the single read path for training and inference.

One DuckDB file per dataset (data_root/datasets/<name>.duckdb) holding versions of four
tables. Each table depends only on its own part of the config (plus the source, period,
warehouse and the code of the components that build it); that part is hashed and every
hash is stored as its own immutable table <table>__<hash>:

    features__<h>  (date, security_id, f1..fF)                           <- feature_spec
    universe__<h>  (refresh_date, valid_until, slot, security_id, is_pad) <- filter_spec
    market__<h>    (date, ts__*, static__*)  also the trading calendar    <- indicator_spec
    labels__<h>    (date, security_id, label)                            <- label_spec
    _manifest      (table_name, config_hash, n_rows, min_date, max_date, built_at, config_json)

Versions are never overwritten, so experiments that share a dataset name do not clobber
each other, and a run (meta.json -> dataset_tables) can always read exactly what it used.
DatasetView.load_window(t) composes the four tables into what the model consumes at date t.
"""

from __future__ import annotations

import ast
import datetime as dt
import inspect
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
import torch

from qam import registry
from qam.config import RunConfig, stable_hash

TABLES = ("features", "universe", "market", "labels")
# Bump when framework build code (this file, pipeline.py) changes table contents. Component
# code is hashed automatically (code_fingerprint), so editing a component needs no bump.
BUILD_VERSION = 1
# Seconds to wait for another process holding the DuckDB write lock.
LOCK_TIMEOUT_S = 600

log = logging.getLogger("qam.data")


# ---------------------------------------------------------------- dense helpers


def security_ids(panel: pl.DataFrame) -> np.ndarray:
    return np.sort(panel["security_id"].unique().to_numpy()).astype(np.int64)


def trading_calendar(panel: pl.DataFrame) -> np.ndarray:
    return np.sort(panel["date"].unique().to_numpy()).astype("datetime64[D]")


def dense(panel: pl.DataFrame, column: str, calendar: np.ndarray, ids: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Pivot one panel column to [len(calendar), len(ids)], NaN where a stock has no value."""
    out = np.full((len(calendar), len(ids)), np.nan, dtype=dtype)
    if panel.height == 0 or len(calendar) == 0 or len(ids) == 0:
        return out
    dates = panel["date"].to_numpy().astype("datetime64[D]")
    secs = panel["security_id"].to_numpy()
    values = panel[column].cast(pl.Float64).fill_null(np.nan).to_numpy()
    rows = np.minimum(np.searchsorted(calendar, dates), len(calendar) - 1)
    cols = np.minimum(np.searchsorted(ids, secs), len(ids) - 1)
    ok = (calendar[rows] == dates) & (ids[cols] == secs)  # drop rows outside the calendar or id list
    out[rows[ok], cols[ok]] = values[ok]
    return out


# ---------------------------------------------------------------- table specs


def code_fingerprint(obj: Any) -> str:
    """Hash of a component's code: its own definition plus the shared code of its module
    (helpers, constants, imports). Comments, docstrings and formatting are ignored, and the
    other registered components of the module are left out, so adding a component does not
    invalidate tables built with the existing ones."""
    module = sys.modules.get(getattr(obj, "__module__", ""))
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):  # defined interactively: fall back to the qualified name
        return stable_hash(f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', repr(obj))}")
    return stable_hash(component_source(source, obj.__name__))


def component_source(module_source: str, name: str) -> str:
    """Normalised source of component `name` plus every module-level statement that is
    not another registered component."""
    body = ast.parse(module_source).body
    if body and _is_docstring(body[0]):
        body = body[1:]
    kept = [node for node in body if not _is_component(node) or node.name == name]
    for node in kept:
        _strip_docstrings(node)
    return "\n".join(ast.unparse(node) for node in kept)


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _is_component(node: ast.stmt) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return False
    for dec in node.decorator_list:
        func = dec.func if isinstance(dec, ast.Call) else dec
        if (isinstance(func, ast.Name) and func.id == "register") or (isinstance(func, ast.Attribute) and func.attr == "register"):
            return True
    return False


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body and _is_docstring(node.body[0]):
            node.body = node.body[1:] or [ast.Pass()]


def table_configs(cfg: RunConfig) -> dict[str, dict[str, Any]]:
    """The part of the config each table depends on; its hash names the table version."""
    data = cfg.data_spec
    warehouse = Path(cfg.paths.warehouse)
    stat = warehouse.stat() if warehouse.exists() else None
    code = lambda kind, c: code_fingerprint(registry.get(kind, c.func))  # noqa: E731
    common = {
        "build_version": BUILD_VERSION,
        "source": data.source.model_dump(),
        "source_code": code("source", data.source),
        "period": data.period.model_dump(mode="json"),
        "warmup_days": data.warmup_days,
        "warehouse": {"name": warehouse.name, "size": stat.st_size if stat else None, "mtime": stat.st_mtime if stat else None},
    }
    return {
        "features": {**common, "features": [c.model_dump() for c in cfg.feature_spec], "code": [code("feature", c) for c in cfg.feature_spec]},
        "universe": {**common, "filter": cfg.filter_spec.model_dump(), "code": code("filter", cfg.filter_spec)},
        "market": {**common, "indicators": [c.model_dump() for c in cfg.indicator_spec], "code": [code("indicator", c) for c in cfg.indicator_spec]},
        "labels": {**common, "label": cfg.label_spec.model_dump(), "code": code("label", cfg.label_spec)},
    }


def table_hashes(cfg: RunConfig) -> dict[str, str]:
    return {name: stable_hash(spec) for name, spec in table_configs(cfg).items()}


def _without_warehouse(spec: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in spec.items() if k != "warehouse"}, sort_keys=True, default=str)


def resolve_tables(cfg: RunConfig, store: "DatasetStore") -> dict[str, str]:
    """The table versions this config reads.

    With the warehouse present this is table_hashes(cfg). Without it (Modal, an inference
    machine) the warehouse fingerprint cannot be computed, so the newest stored version
    built from the same config and component code is used instead.
    """
    if Path(cfg.paths.warehouse).exists():
        return table_hashes(cfg)
    resolved = {}
    for table, spec in table_configs(cfg).items():
        want = _without_warehouse(spec)
        matches = [v for v in store.versions(table) if _without_warehouse(json.loads(v["config_json"])) == want]
        if not matches:
            raise FileNotFoundError(
                f"warehouse {cfg.paths.warehouse} not found and {store.path} has no {table} table built from this config"
            )
        resolved[table] = max(matches, key=lambda v: v["built_at"])["config_hash"]
    log.warning("warehouse not found; using stored table versions %s", resolved)
    return resolved


# ---------------------------------------------------------------- store

_HASH = re.compile(r"^[0-9a-f]{6,64}$")


class DatasetStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def for_config(cls, cfg: RunConfig) -> "DatasetStore":
        return cls(cfg.paths.data_root / "datasets" / f"{cfg.data_spec.dataset_name}.duckdb")

    @staticmethod
    def physical(table: str, config_hash: str) -> str:
        if table not in TABLES or not _HASH.match(str(config_hash)):
            raise ValueError(f"invalid table version {table!r} / {config_hash!r}")
        return f"{table}__{config_hash}"

    def _connect(self, read_only: bool) -> duckdb.DuckDBPyConnection:
        """Open the file, waiting while another process holds the write lock (one writer or
        many readers at a time, e.g. two Modal jobs on one volume)."""
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            try:
                return duckdb.connect(str(self.path), read_only=read_only)
            except duckdb.IOException as exc:
                if "lock" not in str(exc).lower() or time.monotonic() > deadline:
                    raise
                log.info("waiting for the lock on %s", self.path)
                time.sleep(2.0)

    def versions(self, table: str | None = None) -> list[dict[str, Any]]:
        """Manifest rows, oldest first; optionally only one table's."""
        if not self.path.exists():
            return []
        with self._connect(read_only=True) as con:
            exists = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = '_manifest'").fetchone()[0]
            if not exists:
                return []
            rows = con.execute("SELECT * FROM _manifest ORDER BY built_at").pl().to_dicts()
        return [row for row in rows if table is None or row["table_name"] == table]

    def has(self, table: str, config_hash: str) -> bool:
        return any(v["config_hash"] == config_hash for v in self.versions(table))

    def write(self, table: str, frame: pl.DataFrame, config_hash: str, config: dict[str, Any], date_col: str) -> None:
        """Store one table version and its manifest row in a single transaction."""
        name = self.physical(table, config_hash)
        dates = frame[date_col] if frame.height else None
        record = {
            "table_name": table,
            "config_hash": config_hash,
            "n_rows": frame.height,
            "min_date": dates.min() if dates is not None else None,
            "max_date": dates.max() if dates is not None else None,
            "built_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            "config_json": json.dumps(config, sort_keys=True, default=str),
        }
        with self._connect(read_only=False) as con:
            # No primary key: older DuckDB rejects delete + re-insert of one key in a transaction;
            # the DELETE below keeps (table_name, config_hash) unique.
            con.execute(
                "CREATE TABLE IF NOT EXISTS _manifest (table_name VARCHAR, config_hash VARCHAR, n_rows BIGINT, min_date DATE,"
                " max_date DATE, built_at TIMESTAMP, config_json VARCHAR)"
            )
            con.register("frame_view", frame.to_arrow())
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM frame_view")
                con.execute("DELETE FROM _manifest WHERE table_name = ? AND config_hash = ?", [table, config_hash])
                con.execute("INSERT INTO _manifest VALUES (?, ?, ?, ?, ?, ?, ?)", list(record.values()))
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.unregister("frame_view")

    def read(self, table: str, config_hash: str, where: str | None = None) -> pl.DataFrame:
        name = self.physical(table, config_hash)
        with self._connect(read_only=True) as con:
            sql = f"SELECT * FROM {name}" + (f" WHERE {where}" if where else "")
            return con.execute(sql).pl()

    def drop(self, table: str, config_hash: str) -> None:
        """Delete one version. DuckDB reuses the space but does not shrink the file."""
        name = self.physical(table, config_hash)
        with self._connect(read_only=False) as con:
            con.execute("BEGIN TRANSACTION")
            con.execute(f"DROP TABLE IF EXISTS {name}")
            con.execute("DELETE FROM _manifest WHERE table_name = ? AND config_hash = ?", [table, config_hash])
            con.execute("COMMIT")


# ---------------------------------------------------------------- read path


def apply_transforms(x: np.ndarray, mask: np.ndarray, transforms, feature_names: list[str]) -> np.ndarray:
    """Run window_spec.transforms on one window. A transform may restrict itself to some
    features with kwargs.features: [...]; the framework handles that key."""
    for t in transforms:
        kwargs = dict(t.kwargs)
        only = kwargs.pop("features", None)
        fn = registry.get("transform", t.func)
        if only is None:
            x = fn(x, mask, **kwargs)
        else:
            idx = [feature_names.index(name) for name in only]
            x[:, :, idx] = fn(x[:, :, idx], mask, **kwargs)
    return x


@dataclass
class DatasetView:
    """In-memory, dense view of a built dataset. One instance serves training and inference."""

    calendar: np.ndarray  # [D] datetime64[D] trading days
    ids: np.ndarray  # [N] security ids that ever appear in the universe
    feature_names: list[str]
    features: np.ndarray  # [D, N, F] float32, NaN where missing
    slots: np.ndarray  # [D, k] column index into ids, -1 for padding / no universe
    covered: np.ndarray  # [D] bool, a universe period covers this date
    labels: np.ndarray | None  # [D, N] float32
    mkt_ts: np.ndarray  # [D, m'] float32
    mkt_static: np.ndarray  # [D, m - m'] float32
    mkt_ts_names: list[str]
    mkt_static_names: list[str]
    cfg: RunConfig

    @classmethod
    def load(cls, store: DatasetStore, cfg: RunConfig, tables: dict[str, str], with_labels: bool = True) -> "DatasetView":
        """tables: table -> version hash, e.g. resolve_tables(cfg) or meta.json dataset_tables."""
        market = store.read("market", tables["market"]).sort("date")
        calendar = market["date"].to_numpy().astype("datetime64[D]")
        ts_names = [c for c in market.columns if c.startswith("ts__")]
        static_names = [c for c in market.columns if c.startswith("static__")]

        universe = store.read("universe", tables["universe"]).sort(["refresh_date", "slot"])
        ids = np.sort(universe.filter(~pl.col("is_pad"))["security_id"].unique().to_numpy()).astype(np.int64)
        k = int(universe["slot"].max()) + 1

        id_list = ",".join(map(str, ids.tolist())) or "NULL"
        feats = store.read("features", tables["features"], where=f"security_id IN ({id_list})")
        names = [c for c in feats.columns if c not in ("date", "security_id")]
        features = np.stack([dense(feats, n, calendar, ids, np.float32) for n in names], axis=-1) if names else np.zeros((len(calendar), len(ids), 0), np.float32)

        slots = np.full((len(calendar), k), -1, dtype=np.int64)
        covered = np.zeros(len(calendar), dtype=bool)
        for (refresh, until), period in universe.group_by(["refresh_date", "valid_until"], maintain_order=True):
            lo = np.searchsorted(calendar, np.datetime64(refresh, "D"))
            hi = np.searchsorted(calendar, np.datetime64(until, "D"))
            col = np.full(k, -1, dtype=np.int64)
            real = period.filter(~pl.col("is_pad"))
            col[real["slot"].to_numpy()] = np.searchsorted(ids, real["security_id"].to_numpy())
            slots[lo:hi] = col
            covered[lo:hi] = True

        labels = None
        if with_labels:
            lab = store.read("labels", tables["labels"], where=f"security_id IN ({id_list})")
            labels = dense(lab, "label", calendar, ids, np.float32)

        def market_matrix(cols: list[str]) -> np.ndarray:
            if not cols:
                return np.zeros((len(calendar), 0), np.float32)
            return market.select(cols).fill_null(np.nan).to_numpy().astype(np.float32)

        return cls(
            calendar=calendar,
            ids=ids,
            feature_names=names,
            features=features,
            slots=slots,
            covered=covered,
            labels=labels,
            mkt_ts=market_matrix(ts_names),
            mkt_static=market_matrix(static_names),
            mkt_ts_names=ts_names,
            mkt_static_names=static_names,
            cfg=cfg,
        )

    @property
    def k(self) -> int:
        return self.slots.shape[1]

    def decision_indices(self) -> np.ndarray:
        """Calendar positions that can be decision dates: inside the period, covered by the
        universe and with a full lookback window before them."""
        L = self.cfg.window_spec.lookback
        start = np.datetime64(self.cfg.data_spec.period.start, "D")
        end = np.datetime64(self.cfg.data_spec.period.end, "D")
        idx = np.arange(len(self.calendar))
        ok = self.covered & (idx >= L) & (self.calendar >= start) & (self.calendar <= end)
        return idx[ok]

    def index_of(self, t: dt.date | np.datetime64) -> int:
        t = np.datetime64(t, "D")
        i = int(np.searchsorted(self.calendar, t))
        if i >= len(self.calendar) or self.calendar[i] != t:
            raise KeyError(f"{t} is not a trading day in this dataset")
        return i

    def load_window(self, t: dt.date | np.datetime64 | int) -> dict[str, np.ndarray]:
        """Everything the model needs at decision date t, using data strictly before t."""
        i = t if isinstance(t, (int, np.integer)) else self.index_of(t)
        window = self.cfg.window_spec
        L = window.lookback
        if i < L:
            raise ValueError(f"decision index {i} has fewer than {L} trading days of history")
        cols = self.slots[i]
        real = cols >= 0
        k, F = self.k, self.features.shape[2]

        x = np.full((k, L, F), np.nan, dtype=np.float32)
        x[real] = self.features[i - L : i, cols[real], :].transpose(1, 0, 2)
        gaps = np.isnan(x).any(axis=2)  # [k, L] day missing if any feature is missing
        mask = real & (gaps.sum(axis=1) <= window.max_missing)
        if window.fill == "ffill":
            x = _ffill_time(x)
        # Missing cells stay NaN through the transforms, which ignore them, and become 0 only
        # afterwards: after a z-score 0 is the cross-sectional mean, not an extreme raw value.
        x = apply_transforms(x, mask, window.transforms, self.feature_names)
        x = np.nan_to_num(x, nan=0.0)
        x[~mask] = 0.0

        out = {
            "x": x.astype(np.float32),
            "mask": mask,
            "ids": np.where(real, self.ids[np.maximum(cols, 0)], -1),
            "x_mkt_ts": np.nan_to_num(self.mkt_ts[i - L : i]).astype(np.float32),
            "x_mkt_static": np.nan_to_num(self.mkt_static[i - 1]).astype(np.float32),
            "date_index": np.int64(i),
        }
        if self.labels is not None:
            y = np.full(k, np.nan, dtype=np.float32)
            y[real] = self.labels[i, cols[real]]
            out["label_mask"] = mask & ~np.isnan(y)
            out["label"] = np.nan_to_num(y, nan=0.0)
        return out


def _ffill_time(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along the lookback axis of [k, L, F]."""
    k, L, F = x.shape
    valid = ~np.isnan(x)
    pos = np.where(valid, np.arange(L)[None, :, None], 0)
    np.maximum.accumulate(pos, axis=1, out=pos)
    filled = np.take_along_axis(x, pos, axis=1)
    return filled


class WindowDataset(torch.utils.data.Dataset):
    """One item = one decision date: the full cross-section of k slots."""

    def __init__(self, view: DatasetView, indices: np.ndarray):
        self.view = view
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, n: int) -> dict[str, torch.Tensor]:
        item = self.view.load_window(int(self.indices[n]))
        return {key: torch.as_tensor(value) for key, value in item.items()}
