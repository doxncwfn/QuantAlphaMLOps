"""Stock Ranking v27 — Clean Single-Responsibility Quantitative Alpha & Portfolio Architecture."""

from __future__ import annotations

__version__ = "27.0.0"

from v27.config import (
    CANDIDATES_1D_16,
    CANDIDATES_24,
    EXPERIMENTS_V7,
    FEATURE_CLUSTERS,
    STRATEGIES_V7,
    DataConfig,
    ModelConfig,
    PathConfig,
    PortfolioBacktestConfig,
    RiskOverlayConfig,
    TrainConfig,
    V27Config,
)

__all__ = [
    "CANDIDATES_1D_16",
    "CANDIDATES_24",
    "EXPERIMENTS_V7",
    "FEATURE_CLUSTERS",
    "STRATEGIES_V7",
    "DataConfig",
    "ModelConfig",
    "PathConfig",
    "PortfolioBacktestConfig",
    "RiskOverlayConfig",
    "TrainConfig",
    "V27Config",
    "__version__",
]
