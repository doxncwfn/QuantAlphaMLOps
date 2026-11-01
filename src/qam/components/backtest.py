"""Backtest engines. PLACEHOLDER: the interface is fixed, the engine is not implemented yet.

Contract: a BacktestEngine subclass. run(scores, load_returns) -> BacktestResult where
    scores        pd.DataFrame[date, security_id, score]   out-of-sample predictions
    load_returns  () -> pd.DataFrame[date, security_id, ret_total], daily returns with
                  delisting returns included; reads the warehouse, so call it only when needed
P&L must include delisting returns of held stocks (context/04).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from qam.registry import register


@dataclass
class BacktestResult:
    nav: pd.Series
    positions: pd.DataFrame | None = None
    stats: dict[str, float] = field(default_factory=dict)


class BacktestEngine(ABC):
    @abstractmethod
    def run(self, scores: pd.DataFrame, load_returns: Callable[[], pd.DataFrame]) -> BacktestResult: ...


@register("backtest", "placeholder")
class PlaceholderBacktest(BacktestEngine):
    def __init__(self, **params):
        self.params = params

    def run(self, scores: pd.DataFrame, load_returns: Callable[[], pd.DataFrame]) -> BacktestResult:
        raise NotImplementedError("backtest engine not implemented yet (placeholder)")
