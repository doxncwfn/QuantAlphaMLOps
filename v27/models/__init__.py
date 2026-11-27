"""Model architecture blocks and differentiable ranking losses for v27."""

from __future__ import annotations

from v27.models.cross_sectional_tfm import CrossSectionalTransformerHead
from v27.models.hybrid_lstm_tfm import HybridLstmTfm
from v27.models.losses import compute_spearman_loss, masked_corr, masked_rank, soft_rank
from v27.models.master_gate import MarketFeatureGate
from v27.models.temporal_encoder import TemporalLstmAttention

__all__ = [
    "CrossSectionalTransformerHead",
    "HybridLstmTfm",
    "MarketFeatureGate",
    "TemporalLstmAttention",
    "compute_spearman_loss",
    "masked_corr",
    "masked_rank",
    "soft_rank",
]
