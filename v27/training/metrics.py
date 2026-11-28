"""Cross-sectional validation ranking metrics (RankIC, TailIC, Top/Bottom Hit)."""

from __future__ import annotations

import torch

from v27.models.losses import masked_corr, masked_rank


@torch.no_grad()
def daily_rank_ic(
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    min_stocks: int = 20,
) -> torch.Tensor:
    """Compute daily cross-sectional Spearman Rank IC on valid rows."""
    ic = masked_corr(masked_rank(scores, valid), masked_rank(labels, valid), valid)
    return ic[valid.sum(1) >= min_stocks]


@torch.no_grad()
def daily_tail_ic(
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    q: float = 0.05,
    min_stocks: int = 20,
) -> torch.Tensor:
    """Compute Spearman Rank IC restricted to top/bottom q predicted score tails."""
    n = valid.sum(1, keepdim=True).float().clamp(min=1.0)
    pct = (masked_rank(scores, valid) + 0.5) / n
    tail_mask = valid & ((pct >= 1.0 - q) | (pct <= q))
    tic = masked_corr(
        masked_rank(scores, tail_mask), masked_rank(labels, tail_mask), tail_mask
    )
    ok = (valid.sum(1) >= min_stocks) & (tail_mask.sum(1) >= 6)
    return tic[ok]


@torch.no_grad()
def daily_top_bot_hit(
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    q: float = 0.10,
    min_stocks: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute daily Top-q and Bottom-q directional classification hit rates."""
    n = valid.sum(1, keepdim=True).float().clamp(min=1.0)
    s_pct = (masked_rank(scores, valid) + 0.5) / n
    y_pct = (masked_rank(labels, valid) + 0.5) / n
    p_top, t_top = valid & (s_pct >= 1.0 - q), valid & (y_pct >= 1.0 - q)
    p_bot, t_bot = valid & (s_pct <= q), valid & (y_pct <= q)
    ok = (valid.sum(1) >= min_stocks) & (p_top.sum(1) > 0) & (p_bot.sum(1) > 0)
    top_hit = (p_top & t_top).sum(1).float() / p_top.sum(1).float().clamp(min=1.0)
    bot_hit = (p_bot & t_bot).sum(1).float() / p_bot.sum(1).float().clamp(min=1.0)
    return top_hit[ok], bot_hit[ok]
