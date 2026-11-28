"""Tests for strict determinism across model initialization, loss, and evaluation."""

from __future__ import annotations

import numpy as np
import torch

from src.data.normalize_schema import seed_everything
from src.losses.combined import CombinedRankingLoss
from src.models.architecture import BaselineArchitectureY
from src.training.evaluate import evaluate_predictions


def run_one_step(seed: int = 42) -> tuple[float, list[torch.Tensor]]:
    """Execute a single forward-backward optimization step with deterministic seeding."""
    seed_everything(seed)
    model = BaselineArchitectureY(
        d_model=32,
        lstm_hidden=32,
        context_hidden=16,
        n_heads=2,
        tfm_layers=1,
        ff_dim=64,
    )
    loss_fn = CombinedRankingLoss()

    torch.manual_seed(seed)
    B, K, L = 2, 20, 60
    x_stock = torch.randn(B, K, L, 19)
    x_ctx = torch.randn(B, L, 11)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[:, 15:] = False
    t5 = torch.randn(B, K)
    ys = torch.clamp(torch.randn(B, K) * 0.3, -0.5, 0.5)

    scores = model(x_stock, x_ctx, mask)
    loss, _, _ = loss_fn(scores, t5, ys, mask)
    loss.backward()

    grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    return float(loss.item()), grads


def test_training_step_is_strictly_deterministic() -> None:
    """Verify two runs with the same random seed produce bitwise identical loss and gradients."""
    loss1, grads1 = run_one_step(seed=42)
    loss2, grads2 = run_one_step(seed=42)

    assert loss1 == loss2
    assert len(grads1) == len(grads2)
    for g1, g2 in zip(grads1, grads2):
        torch.testing.assert_close(g1, g2, rtol=0.0, atol=0.0)


def test_evaluation_metrics_are_strictly_deterministic() -> None:
    """Verify evaluation metric calculations are strictly deterministic given the same inputs."""
    np.random.seed(42)
    T, K = 30, 100
    scores = np.random.randn(T, K)
    targets = np.random.randn(T, K)
    mask = np.ones((T, K), dtype=bool)
    mask[:, 80:] = False

    res1 = evaluate_predictions(scores, targets, mask)
    res2 = evaluate_predictions(scores, targets, mask)

    assert res1 == res2
