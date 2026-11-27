"""YAML-backed typed configuration loader and shared constants for v27."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(os.environ.get("V27_CONFIG", "config/v27.yaml"))


def _load_yaml(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


_RAW_CFG: dict[str, Any] = _load_yaml() if DEFAULT_CONFIG_PATH.exists() else {}

EXPERIMENT_SPECS: dict[str, dict[str, Any]] = dict(_RAW_CFG.get("experiments", {}))
EXPERIMENTS_V7: tuple[str, ...] = tuple(EXPERIMENT_SPECS.keys())
STRATEGIES_V7: tuple[str, ...] = tuple(_RAW_CFG.get("strategies", ()))
CANDIDATES_24: tuple[str, ...] = tuple(
    f"{exp}__{strat}" for exp in EXPERIMENTS_V7 for strat in STRATEGIES_V7
)
CANDIDATES_1D_16: tuple[str, ...] = tuple(
    f"{exp}__{strat}" for exp in ("raw_t1", "risk_t1") for strat in STRATEGIES_V7
)
FEATURE_CLUSTERS: dict[str, list[str]] = dict(_RAW_CFG.get("feature_clusters", {}))
SECTOR12_NAMES: tuple[str, ...] = tuple(_RAW_CFG.get("sector12_names", ()))


@dataclass(frozen=True)
class PathConfig:
    warehouse_path: Path = Path(
        _RAW_CFG.get("paths", {}).get(
            "warehouse_path", "data/warehouse/russell1000.duckdb"
        )
    )
    qh_v7_dir: Path = Path(_RAW_CFG.get("paths", {}).get("qh_v7_dir", "outputs/QH_v7"))
    output_dir: Path = Path(_RAW_CFG.get("paths", {}).get("output_dir", "outputs/v27"))
    cache_filename: str = str(
        _RAW_CFG.get("paths", {}).get("cache_filename", "dataset_top1000_f34_cache.pt")
    )
    ff6_cache_path: Path = Path(
        _RAW_CFG.get("paths", {}).get(
            "ff6_cache_path", "outputs/QH_v7/v7_ensemble/ff6_daily_cache.parquet"
        )
    )

    @property
    def cache_file(self) -> Path:
        return self.qh_v7_dir / self.cache_filename

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def weights_dir(self) -> Path:
        return self.output_dir / "weights"

    @property
    def alpha_dir(self) -> Path:
        return self.output_dir / "alpha"


@dataclass(frozen=True)
class DataConfig:
    start_date: str = str(_RAW_CFG.get("data", {}).get("start_date", "2000-07-01"))
    test_start: str = str(_RAW_CFG.get("data", {}).get("test_start", "2010-01-01"))
    test_end: str = str(_RAW_CFG.get("data", {}).get("test_end", "2024-12-31"))
    n_folds: int = int(_RAW_CFG.get("data", {}).get("n_folds", 14))
    top_k: int = int(_RAW_CFG.get("data", {}).get("top_k", 1000))
    ttl_days: int = int(_RAW_CFG.get("data", {}).get("ttl_days", 60))
    buffer_ratio: float = float(_RAW_CFG.get("data", {}).get("buffer_ratio", 0.10))
    x_win: int = int(_RAW_CFG.get("data", {}).get("x_win", 20))
    y_win: int = int(_RAW_CFG.get("data", {}).get("y_win", 30))
    lookback: int = int(_RAW_CFG.get("data", {}).get("lookback", 60))
    max_missing: int = int(_RAW_CFG.get("data", {}).get("max_missing", 2))
    feature_norm: str = str(_RAW_CFG.get("data", {}).get("feature_norm", "rankgauss"))
    feature_clip: float = float(_RAW_CFG.get("data", {}).get("feature_clip", 3.0))
    vol_floor: float = float(_RAW_CFG.get("data", {}).get("vol_floor", 0.005))
    mkt_windows: tuple[int, ...] = tuple(
        _RAW_CFG.get("data", {}).get("mkt_windows", (1, 5, 20, 60))
    )


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = int(_RAW_CFG.get("model", {}).get("hidden_dim", 64))
    lstm_layers: int = int(_RAW_CFG.get("model", {}).get("lstm_layers", 2))
    tfm_layers: int = int(_RAW_CFG.get("model", {}).get("tfm_layers", 2))
    heads: int = int(_RAW_CFG.get("model", {}).get("heads", 4))
    dropout: float = float(_RAW_CFG.get("model", {}).get("dropout", 0.2))
    gate_temp: float = float(_RAW_CFG.get("model", {}).get("gate_temp", 5.0))


@dataclass(frozen=True)
class TrainConfig:
    seed: int = int(_RAW_CFG.get("train", {}).get("seed", 42))
    batch_dates: int = int(_RAW_CFG.get("train", {}).get("batch_dates", 16))
    max_epochs: int = int(_RAW_CFG.get("train", {}).get("max_epochs", 20))
    patience: int = int(_RAW_CFG.get("train", {}).get("patience", 4))
    lr: float = float(_RAW_CFG.get("train", {}).get("lr", 3e-4))
    weight_decay: float = float(_RAW_CFG.get("train", {}).get("weight_decay", 1e-4))
    train_subsample: int = int(_RAW_CFG.get("train", {}).get("train_subsample", 1000))
    val_days: int = int(_RAW_CFG.get("train", {}).get("val_days", 400))
    embargo: int = int(_RAW_CFG.get("train", {}).get("embargo", 5))
    time_decay_halflife: int = int(
        _RAW_CFG.get("train", {}).get("time_decay_halflife", 1500)
    )
    min_stocks: int = int(_RAW_CFG.get("train", {}).get("min_stocks", 20))
    soft_tau: float = float(_RAW_CFG.get("train", {}).get("soft_tau", 0.1))
    tail_q: float = float(_RAW_CFG.get("train", {}).get("tail_q", 0.05))
    tail_ic_lambda: float = float(_RAW_CFG.get("train", {}).get("tail_ic_lambda", 0.50))
    hit_q: float = float(_RAW_CFG.get("train", {}).get("hit_q", 0.10))


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    book_size: float = float(
        _RAW_CFG.get("portfolio", {}).get("book_size", 20_000_000.0)
    )
    ann_days: int = int(_RAW_CFG.get("portfolio", {}).get("ann_days", 252))
    baseline_cost_bps: float = float(
        _RAW_CFG.get("portfolio", {}).get("baseline_cost_bps", 2.0)
    )
    cost_grid_bps: tuple[float, ...] = tuple(
        _RAW_CFG.get("portfolio", {}).get("cost_grid_bps", (0.0, 2.0, 5.0, 10.0, 15.0))
    )
    short_borrow_ann: float = float(
        _RAW_CFG.get("portfolio", {}).get("short_borrow_ann", 0.005)
    )
    impact_coeff: float = float(_RAW_CFG.get("portfolio", {}).get("impact_coeff", 0.10))
    warmup_days: int = int(_RAW_CFG.get("portfolio", {}).get("warmup_days", 252))
    purge_days: int = int(_RAW_CFG.get("portfolio", {}).get("purge_days", 6))
    buf_enter_q: float = float(_RAW_CFG.get("portfolio", {}).get("buf_enter_q", 0.10))
    buf_exit_q: float = float(_RAW_CFG.get("portfolio", {}).get("buf_exit_q", 0.20))
    hold_days: int = int(_RAW_CFG.get("portfolio", {}).get("hold_days", 5))


@dataclass(frozen=True)
class RiskOverlayConfig:
    target_vol: float = float(_RAW_CFG.get("risk", {}).get("target_vol", 0.09))
    vol_scale_min: float = float(_RAW_CFG.get("risk", {}).get("vol_scale_min", 0.35))
    vol_scale_max: float = float(_RAW_CFG.get("risk", {}).get("vol_scale_max", 1.35))
    max_gross: float = float(_RAW_CFG.get("risk", {}).get("max_gross", 1.80))
    max_single_w: float = float(_RAW_CFG.get("risk", {}).get("max_single_w", 0.025))
    max_abs_beta: float = float(_RAW_CFG.get("risk", {}).get("max_abs_beta", 0.08))
    max_sector_net: float = float(_RAW_CFG.get("risk", {}).get("max_sector_net", 0.15))
    max_adv_frac: float = float(_RAW_CFG.get("risk", {}).get("max_adv_frac", 0.05))
    dd_threshold: float = float(_RAW_CFG.get("risk", {}).get("dd_threshold", 0.08))


@dataclass(frozen=True)
class V27Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    portfolio: PortfolioBacktestConfig = field(default_factory=PortfolioBacktestConfig)
    risk: RiskOverlayConfig = field(default_factory=RiskOverlayConfig)
