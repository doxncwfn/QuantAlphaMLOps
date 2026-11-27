"""Market-guided feature gating component (MASTER Gate)."""

from __future__ import annotations

import torch
from torch import nn


class MarketFeatureGate(nn.Module):
    """Map macro market context mkt -> cross-feature importance multipliers alpha_t."""

    def __init__(self, m_dim: int, n_features: int, gate_temp: float = 5.0) -> None:
        super().__init__()
        self.n_features = n_features
        self.gate_temp = gate_temp
        self.proj = nn.Linear(m_dim, n_features)

    def forward(self, mkt: torch.Tensor) -> torch.Tensor:
        return self.n_features * torch.softmax(self.proj(mkt) / self.gate_temp, dim=-1)
