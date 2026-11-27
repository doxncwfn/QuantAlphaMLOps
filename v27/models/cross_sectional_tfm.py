"""Cross-sectional Transformer encoder and MLP ranking head."""

from __future__ import annotations

import torch
from torch import nn


class CrossSectionalTransformerHead(nn.Module):
    """Contextualize active stock tokens within each date and project to cross-sectional alpha scores."""

    def __init__(
        self,
        hidden: int = 64,
        heads: int = 4,
        tfm_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            hidden,
            heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tfm = nn.TransformerEncoder(
            layer, num_layers=tfm_layers, enable_nested_tensor=False
        )
        self.out_ln = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pad = ~mask
        empty = pad.all(dim=1)
        if empty.any():
            pad = pad.clone()
            pad[empty, 0] = False
        z = self.out_ln(self.tfm(tokens, src_key_padding_mask=pad))
        return (
            self.head(torch.cat([tokens, z], dim=-1))
            .squeeze(-1)
            .masked_fill(~mask, 0.0)
        )
