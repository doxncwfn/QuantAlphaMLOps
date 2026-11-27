"""Composite Hybrid LSTM-Transformer ranking architecture with MASTER feature gate."""

from __future__ import annotations

import torch
from torch import nn

from v27.config import ModelConfig
from v27.models.cross_sectional_tfm import CrossSectionalTransformerHead
from v27.models.master_gate import MarketFeatureGate
from v27.models.temporal_encoder import TemporalLstmAttention


class HybridLstmTfm(nn.Module):
    """Market-gated temporal LSTM + cross-sectional Transformer cross-sectional alpha model."""

    def __init__(
        self, n_features: int, m_dim: int, cfg: ModelConfig | None = None
    ) -> None:
        super().__init__()
        cfg = cfg or ModelConfig()
        self.gate = MarketFeatureGate(m_dim, n_features, gate_temp=cfg.gate_temp)
        self.temporal = TemporalLstmAttention(
            n_features,
            hidden=cfg.hidden_dim,
            lstm_layers=cfg.lstm_layers,
            dropout=cfg.dropout,
        )
        self.cs_head = CrossSectionalTransformerHead(
            hidden=cfg.hidden_dim,
            heads=cfg.heads,
            tfm_layers=cfg.tfm_layers,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mkt: torch.Tensor,
        return_alpha: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        alpha = self.gate(mkt)
        tokens = self.temporal(x * alpha[:, None, None, :])
        scores = self.cs_head(tokens, mask)
        return (scores, alpha) if return_alpha else scores
