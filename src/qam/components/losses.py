"""Training losses.

Contract: a callable class; __call__(scores [B, k], labels [B, k], mask [B, k] bool) -> scalar tensor.
mask is True where the slot is a real stock AND its label exists. Every statistic is computed
within one date (one row) and then averaged over dates; never pool stocks across dates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from qam.registry import register


def _rows(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, min_n: int = 2):
    for b in range(scores.shape[0]):
        valid = mask[b]
        if int(valid.sum()) >= min_n:
            yield scores[b][valid], labels[b][valid]


def _mean(values: list[torch.Tensor], scores: torch.Tensor) -> torch.Tensor:
    if not values:
        return scores.sum() * 0.0  # keeps the graph when a batch has no usable date
    return torch.stack(values).mean()


@register("loss", "hybrid_hinge")
class HybridHingeLoss:
    """Pairwise hinge between true top and bottom deciles + lambda * MSE (legacy HybridAlphaLoss).

    target="zscore" z-scores the labels within each date (sample std + 1e-8, like the legacy
    Target_5D_Z) before both terms; target="raw" uses the labels as given, which reproduces
    the legacy loss exactly when the labels are already z-scored.
    """

    def __init__(self, margin: float = 0.1, decile: float = 0.1, lambda_mse: float = 1.0, target: str = "zscore"):
        if target not in ("zscore", "raw"):
            raise ValueError("target must be 'zscore' or 'raw'")
        self.margin, self.decile, self.lambda_mse, self.target = margin, decile, lambda_mse, target

    def __call__(self, scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        losses = []
        for pred, y in _rows(scores, labels, mask):
            if self.target == "zscore":
                y = (y - y.mean()) / (y.std() + 1e-8)
            n_tail = max(1, int(len(y) * self.decile))
            order = torch.argsort(y)
            longs, shorts = pred[order[-n_tail:]], pred[order[:n_tail]]
            rank = torch.clamp(self.margin - (longs.unsqueeze(1) - shorts.unsqueeze(0)), min=0.0).mean()
            losses.append(rank + self.lambda_mse * F.mse_loss(pred, y))
        return _mean(losses, scores)


@register("loss", "neg_pearson_rank")
class NegPearsonRankLoss:
    """-Pearson(scores, rank(labels)) per date: a simple differentiable surrogate for Spearman IC."""

    def __call__(self, scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        losses = []
        for pred, y in _rows(scores, labels, mask):
            rank = torch.argsort(torch.argsort(y)).to(pred.dtype)
            p, r = pred - pred.mean(), rank - rank.mean()
            corr = (p * r).sum() / (p.norm() * r.norm() + 1e-8)
            losses.append(-corr)
        return _mean(losses, scores)
