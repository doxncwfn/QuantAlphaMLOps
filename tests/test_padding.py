"""Tests for model padding invariance, permutation equivariance, and fully masked dates."""

from __future__ import annotations

import torch

from src.models.architecture import BaselineArchitectureY


def create_sample_batch(B: int = 2, K: int = 16, L: int = 60, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[:, -4:] = False  # last 4 slots are padding
    return {
        "x_stock": torch.randn(B, K, L, 19, generator=g),
        "x_ctx": torch.randn(B, L, 11, generator=g),
        "mask": mask,
    }


def test_padding_content_does_not_affect_real_stock_scores() -> None:
    """Verify that changing values in padded slots (mask=False) leaves real stock scores unchanged."""
    torch.manual_seed(42)
    model = BaselineArchitectureY(
        d_model=32,
        lstm_hidden=32,
        context_hidden=16,
        n_heads=2,
        tfm_layers=1,
        ff_dim=64,
    ).eval()

    b = create_sample_batch(B=2, K=16, L=60, seed=10)
    with torch.no_grad():
        base_scores = model(b["x_stock"], b["x_ctx"], b["mask"])

        # Inject extreme noise into padded slots
        noisy_x = b["x_stock"].clone()
        noisy_x[:, -4:] = 1e5
        noisy_scores = model(noisy_x, b["x_ctx"], b["mask"])

    # Real stock scores (first 12) must be identical
    torch.testing.assert_close(
        base_scores[:, :-4], noisy_scores[:, :-4], rtol=1e-5, atol=1e-5
    )
    # Masked slots must evaluate to 0.0
    assert (base_scores[:, -4:] == 0.0).all()
    assert (noisy_scores[:, -4:] == 0.0).all()


def test_stock_permutation_equivariance() -> None:
    """Verify permutation equivariance: permuting input stocks permutes output scores identically."""
    torch.manual_seed(42)
    model = BaselineArchitectureY(
        d_model=32,
        lstm_hidden=32,
        context_hidden=16,
        n_heads=2,
        tfm_layers=1,
        ff_dim=64,
    ).eval()

    b = create_sample_batch(B=1, K=12, L=60, seed=20)
    b["mask"][:] = True  # all real stocks
    perm = torch.randperm(12)

    with torch.no_grad():
        base = model(b["x_stock"], b["x_ctx"], b["mask"])
        permuted_x = b["x_stock"][:, perm]
        permuted_mask = b["mask"][:, perm]
        permuted_scores = model(permuted_x, b["x_ctx"], permuted_mask)

    torch.testing.assert_close(base[:, perm], permuted_scores, rtol=1e-5, atol=1e-5)


def test_fully_masked_date_gives_finite_zeros() -> None:
    """Verify that a degenerate batch containing an all-masked date produces finite zeros without crashing."""
    torch.manual_seed(42)
    model = BaselineArchitectureY(
        d_model=32,
        lstm_hidden=32,
        context_hidden=16,
        n_heads=2,
        tfm_layers=1,
        ff_dim=64,
    ).eval()

    b = create_sample_batch(B=2, K=10, L=60, seed=30)
    b["mask"][0] = False  # entire first date is masked out

    with torch.no_grad():
        scores = model(b["x_stock"], b["x_ctx"], b["mask"])

    assert torch.isfinite(scores).all()
    assert (scores[0] == 0.0).all()
