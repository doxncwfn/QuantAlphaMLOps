"""Tests for HybridLstmTfm component blocks, empty-mask guard, MASTER gate, and Spearman loss."""

from __future__ import annotations

import torch

from v27.config import ModelConfig
from v27.models.hybrid_lstm_tfm import HybridLstmTfm
from v27.models.losses import compute_spearman_loss
from v27.training.folds import build_walk_forward_folds
from v27.training.metrics import daily_rank_ic, daily_tail_ic, daily_top_bot_hit


def test_hybrid_lstm_tfm_forward_and_gate_sum():
    cfg = ModelConfig(
        hidden_dim=16, lstm_layers=2, tfm_layers=1, heads=2, dropout=0.1, gate_temp=5.0
    )
    model = HybridLstmTfm(n_features=34, m_dim=28, cfg=cfg)
    x = torch.randn(3, 12, 8, 34)
    mask = torch.ones(3, 12, dtype=torch.bool)
    mask[0, -3:] = False
    mask[2, :] = False  # verify all-empty row guard
    mkt = torch.randn(3, 28)

    scores, alpha = model(x, mask, mkt, return_alpha=True)
    assert scores.shape == (3, 12)
    assert alpha.shape == (3, 34)
    assert torch.allclose(alpha.sum(dim=-1), torch.full((3,), 34.0), atol=1e-4)
    assert torch.all(scores[2] == 0.0)
    assert torch.all(scores[0, -3:] == 0.0)


def test_spearman_loss_backprops_and_ic_metrics():
    scores = torch.randn(4, 30, requires_grad=True)
    y = scores.detach() + 0.05 * torch.randn(4, 30)
    valid = torch.ones(4, 30, dtype=torch.bool)
    day_w = torch.ones(4)

    loss = compute_spearman_loss(scores, y, valid, day_w, min_stocks=20, tau=0.1)
    assert loss.item() < 0.25
    loss.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()

    ic = daily_rank_ic(scores.detach(), y, valid, min_stocks=20)
    tic = daily_tail_ic(scores.detach(), y, valid, q=0.10, min_stocks=20)
    th, bh = daily_top_bot_hit(scores.detach(), y, valid, q=0.10, min_stocks=20)
    assert float(ic.mean()) > 0.85
    assert len(tic) == 4
    assert float(th.mean()) > 0.7 and float(bh.mean()) > 0.7


def test_walk_forward_folds_respect_purge_and_embargo():
    import numpy as np

    cal = np.array(
        [np.datetime64("2005-01-03") + np.timedelta64(i, "D") for i in range(6000)]
    )
    dec = np.arange(100, 5000)
    folds = build_walk_forward_folds(
        dec,
        cal,
        test_start="2010-01-01",
        test_end="2018-01-01",
        n_folds=5,
        purge=6,
        embargo=5,
        val_days=200,
    )
    assert len(folds) == 5
    for f in folds:
        assert f["train"][-1] < f["val"][0]
        assert f["val"][-1] < f["test"][0]
