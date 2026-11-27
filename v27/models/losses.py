"""Differentiable Spearman correlation and cross-sectional ranking losses."""

from __future__ import annotations

import torch


def masked_corr(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Compute per-row Pearson correlation restricted to boolean mask m."""
    mf = m.float()
    n = mf.sum(1, keepdim=True).clamp(min=1.0)
    a_c = (a - (a * mf).sum(1, keepdim=True) / n) * mf
    b_c = (b - (b * mf).sum(1, keepdim=True) / n) * mf
    return (a_c * b_c).sum(1) / (
        torch.sqrt((a_c * a_c).sum(1) + 1e-12) * torch.sqrt((b_c * b_c).sum(1) + 1e-12)
    )


def masked_rank(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Compute integer cross-sectional ranks across valid entries per row."""
    xf = x.float().masked_fill(~m, float("-inf"))
    rank = xf.argsort(dim=1).argsort(dim=1).float()
    return rank - (~m).sum(1, keepdim=True).float()


def soft_rank(p: torch.Tensor, m: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Smooth differentiable approximation to cross-sectional ranks."""
    mf = m.float()
    n = mf.sum(1, keepdim=True).clamp(min=2.0)
    mu = (p * mf).sum(1, keepdim=True) / n
    sd = torch.sqrt((((p - mu) * mf) ** 2).sum(1, keepdim=True) / (n - 1) + 1e-8)
    z = (p - mu) / sd
    diff = (z.unsqueeze(2) - z.unsqueeze(1)) / tau
    return (torch.sigmoid(diff) * mf.unsqueeze(1)).sum(2) - 0.5


def compute_spearman_loss(
    scores: torch.Tensor,
    y: torch.Tensor,
    valid: torch.Tensor,
    day_weights: torch.Tensor,
    min_stocks: int = 20,
    tau: float = 0.1,
) -> torch.Tensor:
    """Weighted daily 1 - Spearman RankIC loss via differentiable soft ranking."""
    wt = day_weights * (valid.sum(1) >= min_stocks).float()
    if float(wt.sum()) <= 0:
        return scores.sum() * 0.0
    target_r = torch.where(valid, masked_rank(y, valid), torch.zeros_like(y))
    per_day = 1.0 - masked_corr(soft_rank(scores, valid, tau=tau), target_r, valid)
    return (per_day * wt).sum() / wt.sum()
