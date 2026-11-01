"""Run configuration: pydantic schema and YAML loader.

One experiment = one self-contained spec file (configs/experiments/<name>.yaml) with one
section per pipeline stage: data_spec, feature_spec, filter_spec, indicator_spec, label_spec,
window_spec, model_spec, loss_spec, split_spec, train_spec, evaluation_spec, backtest_spec,
tracking_spec. A pluggable part is written as {func: <registered name>, kwargs: {...}}.

Parameters come only from the spec file. Two exceptions:
  - configs/env/<env>.yaml supplies `paths` (and nothing else);
  - time parameters given on the command line (CLI_TIME_KEYS: --start, --end) win over the
    spec, with a warning when they differ from it.
Relative paths under `paths:` are resolved against the project root (the nearest ancestor
of the spec file that contains pyproject.toml).
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from qam import registry

log = logging.getLogger("qam.config")

# Command-line time parameters -> where they live in the spec. Only these may be set on the CLI.
CLI_TIME_KEYS: dict[str, tuple[str, ...]] = {
    "start": ("data_spec", "period", "start"),
    "end": ("data_spec", "period", "end"),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Component(_Strict):
    """A registered component: func = its registered name, kwargs = keyword arguments."""

    func: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class PathsConfig(_Strict):
    warehouse: Path
    data_root: Path
    runs_root: Path


class PeriodConfig(_Strict):
    start: dt.date
    end: dt.date

    @model_validator(mode="after")
    def _ordered(self) -> "PeriodConfig":
        if self.end <= self.start:
            raise ValueError("period.end must be after period.start")
        return self


class DataSpec(_Strict):
    dataset_name: str = Field(description="names the dataset file data_root/datasets/<dataset_name>.duckdb")
    source: Component
    period: PeriodConfig
    warmup_days: int = Field(400, ge=0, description="calendar days loaded before period.start for feature warm-up")


class WindowSpec(_Strict):
    lookback: int = Field(60, gt=0)
    max_missing: int = Field(2, ge=0, description="a slot with more incomplete days in the window is masked")
    fill: Literal["zero", "ffill"] = "zero"
    transforms: list[Component] = Field(default_factory=list)


class TrainSpec(_Strict):
    max_epochs: int = Field(15, gt=0)
    batch_dates: int = Field(32, gt=0)
    lr: float = Field(1e-4, gt=0)
    weight_decay: float = Field(1e-5, ge=0)
    lr_patience: int = 3
    lr_factor: float = 0.5
    monitor: Literal["val_ic", "val_loss"] = "val_ic"
    early_stopping_patience: int | None = None
    min_valid_slots: int = Field(20, ge=2, description="decision dates with fewer labelled slots are skipped")
    accelerator: str = "auto"
    num_workers: int = 0
    deterministic: bool = True


class EvaluationSpec(_Strict):
    metrics: list[Component] = Field(default_factory=list)


class WandbConfig(_Strict):
    enabled: bool = False
    project: str = "HCMUT-Specialized-Project"
    entity: str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"


class TrackingSpec(_Strict):
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class RunConfig(_Strict):
    name: str
    seed: int = 42
    paths: PathsConfig
    data_spec: DataSpec
    feature_spec: list[Component]
    filter_spec: Component
    indicator_spec: list[Component] = Field(default_factory=list)
    label_spec: Component
    window_spec: WindowSpec = Field(default_factory=WindowSpec)
    model_spec: Component
    loss_spec: Component
    split_spec: Component
    train_spec: TrainSpec = Field(default_factory=TrainSpec)
    evaluation_spec: EvaluationSpec = Field(default_factory=EvaluationSpec)
    backtest_spec: Component = Field(default_factory=lambda: Component(func="placeholder"))
    tracking_spec: TrackingSpec = Field(default_factory=TrackingSpec)

    def to_plain(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def hash(self) -> str:
        return stable_hash(self.to_plain())


def stable_hash(obj: Any, length: int = 12) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


# ---------------------------------------------------------------- loading


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read(path: Path) -> dict:
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def _apply_time(data: dict, flag: str, value: dt.date) -> None:
    """Put a CLI time value into the spec; warn when the spec says something else."""
    *parents, leaf = CLI_TIME_KEYS[flag]
    node = data
    for key in parents:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"--{flag}: {key!r} in the spec is not a mapping")
    in_spec = node.get(leaf)
    if in_spec is not None and str(in_spec) != value.isoformat():
        log.warning("--%s %s overrides %s = %s from the spec", flag, value, ".".join(CLI_TIME_KEYS[flag]), in_spec)
    node[leaf] = value


def find_project_root(start: Path) -> Path:
    """Nearest ancestor holding pyproject.toml; the config's own directory if there is none."""
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


def load_config(path: str | Path, env: str | None = "local", times: dict[str, dt.date | None] | None = None) -> RunConfig:
    """Load a spec. times: CLI time parameters (keys of CLI_TIME_KEYS); None values are ignored."""
    path = Path(path).resolve()
    root = find_project_root(path.parent)
    data = _read(path)
    if env:
        env_file = root / "configs" / "env" / f"{env}.yaml"
        if not env_file.exists():
            raise FileNotFoundError(f"environment overlay not found: {env_file}")
        overlay = _read(env_file)
        if set(overlay) - {"paths"}:
            raise ValueError(f"{env_file}: an environment file may only set `paths`; put parameters in the spec")
        data = _deep_merge(data, overlay)
    for flag, value in (times or {}).items():
        if flag not in CLI_TIME_KEYS:
            raise ValueError(f"{flag!r} cannot be set from the command line; only {sorted(CLI_TIME_KEYS)}")
        if value is not None:
            _apply_time(data, flag, value)

    cfg = RunConfig.model_validate(data)
    for field in ("warehouse", "data_root", "runs_root"):
        value = getattr(cfg.paths, field)
        if not value.is_absolute():
            setattr(cfg.paths, field, (root / value).resolve())
    check_components(cfg)
    return cfg


def check_components(cfg: RunConfig) -> None:
    """Fail before any work starts if a func has no registered implementation."""
    wanted: list[tuple[str, Component]] = [
        ("source", cfg.data_spec.source),
        ("filter", cfg.filter_spec),
        ("label", cfg.label_spec),
        ("model", cfg.model_spec),
        ("loss", cfg.loss_spec),
        ("splitter", cfg.split_spec),
        ("backtest", cfg.backtest_spec),
    ]
    wanted += [("feature", c) for c in cfg.feature_spec]
    wanted += [("indicator", c) for c in cfg.indicator_spec]
    wanted += [("transform", c) for c in cfg.window_spec.transforms]
    wanted += [("metric", c) for c in cfg.evaluation_spec.metrics]
    errors = []
    for kind, component in wanted:
        try:
            registry.get(kind, component.func)
        except KeyError as exc:
            errors.append(str(exc.args[0]))
    if errors:
        raise ValueError("unknown components:\n  " + "\n  ".join(errors))
