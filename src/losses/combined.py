"""Loss functions for v1.3 ranking model: ListMLE + Softplus Tail loss.

Bug fixes applied:
- BUG-02: Softplus tail loss per v1.3 §0.5.1, 1D tensor handling in ListMLE and Tail loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def listmle_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Listwise Maximum Likelihood Estimation (ListMLE) ranking loss per v1.3 §0.5.1.

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores of shape [B, K] or [K].
    targets : torch.Tensor
        Realized targets (target_5d or y_star) of shape [B, K] or [K].
    mask : torch.Tensor
        Boolean mask of shape [B, K] or [K] where True indicates valid constituent.
    eps : float
        Numerical stability epsilon.

    Returns
    -------
    torch.Tensor
        Scalar ListMLE loss averaged across valid dates.
    """
    was_1d = scores.dim() == 1
    if was_1d:
        scores = scores.unsqueeze(0)
        targets = targets.unsqueeze(0)
        mask = mask.unsqueeze(0)

    batch_size = scores.shape[0]
    date_losses = []

    for b in range(batch_size):
        valid = mask[b].bool()
        n_valid = int(valid.sum())
        if n_valid < 2:
            continue

        s = scores[b][valid]
        y = targets[b][valid]

        # Sort scores in descending order of true target (highest return first)
        order = torch.argsort(y, descending=True)
        s_sorted = s[order]

        # Numerically stable suffix logsumexp:
        # suffix_lse[k] = log sum_{j=k}^{n-1} exp(s_sorted[j])
        rev_s = torch.flip(s_sorted, dims=[0])
        rev_lse = torch.logcumsumexp(rev_s, dim=0)
        suffix_lse = torch.flip(rev_lse, dims=[0])

        # Negative log-likelihood per element: suffix_lse[k] - s_sorted[k]
        nll = suffix_lse - s_sorted
        date_losses.append(nll.mean())

    if not date_losses:
        return scores.sum() * 0.0

    return torch.stack(date_losses).mean()


def tail_loss(
    scores: torch.Tensor,
    y_star: torch.Tensor,
    mask: torch.Tensor,
    tail_threshold: float = 0.4,
    tail_multiplier: float = 3.0,
    soft_scale: float = 4.0,
) -> torch.Tensor:
    """v1.3 §0.5.1 Softplus tail loss.

    L_tail = sum_i w_i * softplus(-y_tilde_i * s_i) / sum_i w_i
    y_tilde_i = tanh(soft_scale * y_star_i)
    w_i = 1 + tail_multiplier * 1[|y_star_i| >= tail_threshold]

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores of shape [B, K] or [K].
    y_star : torch.Tensor
        Cross-sectional rank target in [-0.5, +0.5] of shape [B, K] or [K].
    mask : torch.Tensor
        Boolean mask of shape [B, K] or [K].
    tail_threshold : float
        Threshold for tail deciles (|y_star| >= 0.4).
    tail_multiplier : float
        Weight added to tail constituents (yielding weight 1 + 3 = 4).
    soft_scale : float
        Scale factor in soft label tanh(scale * y_star).

    Returns
    -------
    torch.Tensor
        Scalar weighted Softplus loss averaged across valid dates.
    """
    # Handle 1D input by unsqueezing to [1, K]
    was_1d = scores.dim() == 1
    if was_1d:
        scores = scores.unsqueeze(0)
        y_star = y_star.unsqueeze(0)
        mask = mask.unsqueeze(0)

    B = scores.shape[0]
    date_losses = []
    for b in range(B):
        valid = mask[b].bool()
        if int(valid.sum()) < 2:
            continue
        s = scores[b][valid]
        y = y_star[b][valid]
        y_tilde = torch.tanh(soft_scale * y)
        is_tail = (y.abs() >= tail_threshold).float()
        weights = 1.0 + tail_multiplier * is_tail
        sp = F.softplus(-y_tilde * s)
        date_losses.append((weights * sp).sum() / (weights.sum() + 1e-8))

    if not date_losses:
        return scores.sum() * 0.0

    return torch.stack(date_losses).mean()


class CombinedRankingLoss(nn.Module):
    """Combined loss: L = L_ListMLE + 1.5 * L_tail + lambda_hub * L_hub per v1.3 / v1.4 specification."""

    def __init__(
        self,
        lambda_tail: float = 1.5,
        tail_threshold: float = 0.4,
        tail_multiplier: float = 3.0,
        soft_scale: float = 4.0,
        lambda_hub: float = 0.0,
    ):
        super().__init__()
        self.lambda_tail = lambda_tail
        self.tail_threshold = tail_threshold
        self.tail_multiplier = tail_multiplier
        self.soft_scale = soft_scale
        self.lambda_hub = lambda_hub

    def forward(
        self,
        scores: torch.Tensor,
        target_5d: torch.Tensor,
        y_star: torch.Tensor,
        mask: torch.Tensor,
        y_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss and return (total_loss, l_listmle, l_tail)."""
        y_hat = scores
        y = y_norm if y_norm is not None else y_star
        assert y_hat.dtype == torch.float32
        assert y.dtype == torch.float32
        assert target_5d.dtype == torch.float32
        l_mle = listmle_loss(scores, y, mask)
        l_tail = tail_loss(
            scores,
            y_star,
            mask,
            tail_threshold=self.tail_threshold,
            tail_multiplier=self.tail_multiplier,
            soft_scale=self.soft_scale,
        )
        total_loss = l_mle + self.lambda_tail * l_tail
        if self.lambda_hub > 0.0:
            l_hub = F.smooth_l1_loss(
                scores * mask, y * mask, beta=1.0, reduction="sum"
            ) / (mask.sum() + 1e-8)
            total_loss = total_loss + self.lambda_hub * l_hub
        return total_loss, l_mle, l_tail
