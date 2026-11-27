"""Architecture Y: Stock LSTM + Temporal Attention + Context BiLSTM + Cross-Sectional Transformer (v1.3 §8.2 & §9.7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TemporalAttention(nn.Module):
    """Temporal self-attention and attention pooling over a sequence of hidden vectors."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_pool = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [N, L, d_model] where L is sequence length.

        Returns
        -------
        torch.Tensor
            Pooled embedding of shape [N, d_model].
        """
        # Self-attention over temporal dimension
        attn_out, _ = self.mha(x, x, x)
        # Attention pooling across sequence length L
        weights = F.softmax(self.attn_pool(attn_out), dim=1)  # [N, L, 1]
        pooled = (attn_out * weights).sum(dim=1)  # [N, d_model]
        return pooled


class BaselineArchitectureY(nn.Module):
    """Frozen v1.3 Baseline Model Architecture (v1.3 §8.2 & §9.7).

    Components:
    1. Stock Temporal Branch: Unidirectional LSTM(hidden_size=128, 2 layers) + Temporal Attention(4 heads).
    2. Context Branch: Bidirectional LSTM(hidden_size=32, 1 layer) -> FC(64) -> ReLU -> FC(128).
    3. Fusion: Concatenation [h_i ; z_ctx] in R^256 -> Linear(256 -> 128).
    4. Cross-Sectional Transformer: 2 Encoder layers, 4 heads, d_model=128, ff_dim=256, dropout=0.1.
    5. Scorer: Linear(128 -> 1) producing scalar ranking scores per stock.
    """

    def __init__(
        self,
        n_stock_features: int = 19,
        n_context_features: int = 11,
        d_model: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        context_hidden: int = 32,
        n_heads: int = 4,
        tfm_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Stock temporal branch (unidirectional: strictly causal over [t-59, t])
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.stock_lstm = nn.LSTM(
            input_size=n_stock_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=False,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.stock_attn = TemporalAttention(
            embed_dim=lstm_hidden,
            num_heads=n_heads,
            dropout=dropout,
        )

        # 2. Context branch (BiLSTM over market context [t-59, t])
        self.context_lstm = nn.LSTM(
            input_size=n_context_features,
            hidden_size=context_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.context_proj = nn.Sequential(
            nn.Linear(2 * context_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )

        # 3. Fusion: concatenation [h_i; z_ctx] -> Linear(2*d_model -> d_model)
        self.fusion = nn.Linear(lstm_hidden + d_model, d_model)

        # 4. Cross-Sectional Transformer: permutation-equivariant across stocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=tfm_layers,
            enable_nested_tensor=False,
        )

        # 5. Scorer: linear projection to scalar score
        self.scorer = nn.Linear(d_model, 1)

    def forward(
        self,
        x_stock: torch.Tensor,
        x_ctx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x_stock : torch.Tensor
            Stock features of shape [B, K, L, F_stock] (where L=60, F_stock=19).
        x_ctx : torch.Tensor
            Context features of shape [B, L, F_ctx] (where L=60, F_ctx=11).
        mask : torch.Tensor
            Boolean mask of shape [B, K] where True indicates real constituent.

        Returns
        -------
        torch.Tensor
            Predicted ranking scores of shape [B, K].
        """
        B, K, L, F_stock = x_stock.shape

        # Step 1: Stock branch (chunked across stocks to bound peak memory)
        flat_stocks = x_stock.reshape(B * K, L, F_stock)
        chunk_size = 2048
        if flat_stocks.shape[0] <= chunk_size:
            lstm_out, _ = self.stock_lstm(flat_stocks)
            h_flat = self.stock_attn(lstm_out)
        else:
            chunks = []
            for i in range(0, flat_stocks.shape[0], chunk_size):
                sub_out, _ = self.stock_lstm(flat_stocks[i : i + chunk_size])
                chunks.append(self.stock_attn(sub_out))
            h_flat = torch.cat(chunks, dim=0)

        h_stocks = h_flat.reshape(B, K, self.d_model)

        # Step 2: Context branch
        _ctx_out, (h_ctx, _) = self.context_lstm(x_ctx)
        # Concatenate forward and backward last hidden states
        z_ctx_raw = torch.cat([h_ctx[0], h_ctx[1]], dim=-1)  # [B, 2 * context_hidden]
        z_ctx = self.context_proj(z_ctx_raw)  # [B, d_model]

        # Step 3: Fusion by concatenation
        z_ctx_expanded = z_ctx.unsqueeze(1).expand(-1, K, -1)  # [B, K, d_model]
        fused = torch.cat([h_stocks, z_ctx_expanded], dim=-1)  # [B, K, 2 * d_model]
        e = self.fusion(fused)  # [B, K, d_model]

        # Step 4: Cross-Sectional Transformer
        # PyTorch src_key_padding_mask requires True for padded slots to ignore
        padding_mask = ~mask
        # Guard against all-padded row to avoid NaN in attention softmax
        all_padded = padding_mask.all(dim=1)
        if all_padded.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padded, 0] = False

        tfm_out = self.transformer(e, src_key_padding_mask=padding_mask)

        # Step 5: Score projection
        scores = self.scorer(tfm_out).squeeze(-1)  # [B, K]
        # Padded slots output 0.0
        scores = scores.masked_fill(~mask, 0.0)

        return scores
