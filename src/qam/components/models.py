"""Ranking models.

Contract: an nn.Module class. The framework constructs it with the spec's kwargs plus
n_features, n_mkt_ts, n_mkt_static, lookback and k, and calls forward(batch) with
    batch["x"]            [B, k, L, F]   stock windows (already normalised, zeros in masked slots)
    batch["mask"]         [B, k] bool    True for a real stock slot
    batch["x_mkt_ts"]     [B, L, m']     time-series indicators
    batch["x_mkt_static"] [B, m - m']    non-time-series indicators
It returns scores [B, k]. Scores of masked slots are ignored by losses and metrics.
Stock slots carry no positional or identity encoding: reordering stocks must reorder scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qam.registry import register


@register("model", "hybrid_lstm_tfm")
class HybridLstmTfm(nn.Module):
    """LSTM per series, then a Transformer across tokens, then an MLP scorer (context/05_model.md).

    Ported from legacy HybridAlphaRanker (no stock-ID embedding), extended with a padding mask
    and optional indicator tokens: each time-series indicator goes through its own LSTM pass,
    each static indicator is projected to one token; indicator tokens get a learned type embedding.
    """

    def __init__(
        self,
        n_features: int,
        n_mkt_ts: int = 0,
        n_mkt_static: int = 0,
        lookback: int | None = None,
        k: int | None = None,
        hidden_dim: int = 64,
        lstm_layers: int = 2,
        tfm_layers: int = 2,
        heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.stock_lstm = nn.LSTM(n_features, hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=lstm_dropout)
        self.n_mkt_ts = n_mkt_ts
        self.n_mkt_static = n_mkt_static
        if n_mkt_ts:
            self.mkt_lstm = nn.LSTM(1, hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=lstm_dropout)
            self.mkt_ts_type = nn.Parameter(torch.zeros(n_mkt_ts, hidden_dim))
        if n_mkt_static:
            self.mkt_static_proj = nn.Linear(1, hidden_dim)
            self.mkt_static_type = nn.Parameter(torch.zeros(n_mkt_static, hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=tfm_layers, enable_nested_tensor=False)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x, mask = batch["x"], batch["mask"]
        B, K, L, F = x.shape
        _, (h, _) = self.stock_lstm(x.reshape(B * K, L, F))
        tokens = [h[-1].reshape(B, K, -1)]
        ignore = [~mask]

        if self.n_mkt_ts:
            series = batch["x_mkt_ts"].permute(0, 2, 1).reshape(B * self.n_mkt_ts, L, 1)
            _, (hm, _) = self.mkt_lstm(series)
            tokens.append(hm[-1].reshape(B, self.n_mkt_ts, -1) + self.mkt_ts_type)
            ignore.append(torch.zeros(B, self.n_mkt_ts, dtype=torch.bool, device=x.device))
        if self.n_mkt_static:
            values = batch["x_mkt_static"].unsqueeze(-1)
            tokens.append(self.mkt_static_proj(values) + self.mkt_static_type)
            ignore.append(torch.zeros(B, self.n_mkt_static, dtype=torch.bool, device=x.device))

        z = torch.cat(tokens, dim=1)
        padding = torch.cat(ignore, dim=1)
        # A date with no real token would make attention undefined; let it attend to slot 0 (its score is ignored).
        empty = padding.all(dim=1)
        if empty.any():
            padding = padding.clone()
            padding[empty, 0] = False
        z = self.transformer(z, src_key_padding_mask=padding)
        scores = self.scorer(z[:, :K]).squeeze(-1)
        return scores.masked_fill(~mask, 0.0)
