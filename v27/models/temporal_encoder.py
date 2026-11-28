"""Temporal LSTM + last-step attention pooling encoder."""

from __future__ import annotations

import math

import torch
from torch import nn


class TemporalLstmAttention(nn.Module):
    """Encode per-stock lookback sequence (B, K, L, F) into stock tokens (B, K, hidden)."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU())
        self.lstm = nn.LSTM(
            hidden,
            hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.pool_proj = nn.Linear(2 * hidden, hidden)
        self.pool_ln = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_size, k_size, l_size, _ = x.shape
        h_all, _ = self.lstm(self.embed(x).reshape(b_size * k_size, l_size, -1))
        last = h_all[:, -1, :]
        att = torch.softmax(
            (h_all * last[:, None, :]).sum(-1) / math.sqrt(h_all.shape[-1]), dim=1
        )
        pooled = (att[:, :, None] * h_all).sum(1)
        return self.pool_ln(self.pool_proj(torch.cat([last, pooled], dim=-1))).reshape(
            b_size, k_size, -1
        )
