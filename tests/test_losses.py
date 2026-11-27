"""Tests for ranking loss functions: ListMLE numerical sanity, tail loss anti-symmetry, and padding."""

from __future__ import annotations

import torch

from src.losses.combined import CombinedRankingLoss, listmle_loss, tail_loss


def test_listmle_numerical_sanity_perfect_vs_reversed() -> None:
    """Verify that perfect ranking predictions produce lower ListMLE loss than reversed predictions."""
    B, K = 2, 20
    targets = torch.linspace(-0.5, 0.5, K).unsqueeze(0).expand(B, -1)
    mask = torch.ones(B, K, dtype=torch.bool)

    # Perfect scores: aligned with targets
    perfect_scores = targets * 5.0
    loss_perfect = listmle_loss(perfect_scores, targets, mask)

    # Inverted scores: opposite of targets
    inverted_scores = -targets * 5.0
    loss_inverted = listmle_loss(inverted_scores, targets, mask)

    assert loss_perfect < loss_inverted
    assert torch.isfinite(loss_perfect)
    assert torch.isfinite(loss_inverted)


def test_listmle_shift_invariance() -> None:
    """Verify that adding an arbitrary constant scalar to scores leaves ListMLE unchanged."""
    B, K = 2, 25
    torch.manual_seed(42)
    scores = torch.randn(B, K)
    targets = torch.randn(B, K)
    mask = torch.ones(B, K, dtype=torch.bool)

    base_loss = listmle_loss(scores, targets, mask)
    shifted_loss = listmle_loss(scores + 42.0, targets, mask)

    torch.testing.assert_close(base_loss, shifted_loss, rtol=1e-5, atol=1e-5)


def test_listmle_extreme_logits_no_nan() -> None:
    """Verify that extreme positive/negative logits do not overflow or produce NaNs."""
    B, K = 2, 30
    scores = torch.tensor([[-100.0, -50.0, 0.0, 50.0, 100.0] * 6] * B)
    targets = torch.randn(B, K)
    mask = torch.ones(B, K, dtype=torch.bool)

    loss = listmle_loss(scores, targets, mask)
    assert torch.isfinite(loss).all()


def test_tail_loss_anti_symmetry_and_weights() -> None:
    """Verify mathematical properties of tail loss:

    1. Soft labels y_tilde = tanh(4 * y*) are strictly anti-symmetric: y_tilde(-y*) == -y_tilde(y*).
    2. Tail weights w(y*) are symmetric: w(-y*) == w(y*), with w=4.0 for |y*| >= 0.4.
    3. Gradient of tail loss with respect to scores is anti-symmetric under sign flip:
       grad(-scores, -y*) == -grad(scores, y*).
    4. Tail loss value is symmetric under simultaneous sign flip:
       L_tail(-scores, -y*) == L_tail(scores, y*).
    """
    y_star_vals = torch.tensor([-0.45, -0.40, -0.20, 0.0, 0.20, 0.40, 0.45])
    soft_pos = torch.tanh(4.0 * y_star_vals)
    soft_neg = torch.tanh(4.0 * (-y_star_vals))
    # 1. Anti-symmetry of soft labels
    torch.testing.assert_close(soft_pos, -soft_neg, atol=1e-7, rtol=1e-7)

    # 2. Tail weight symmetry
    w_pos = 1.0 + 3.0 * (y_star_vals.abs() >= 0.4).float()
    w_neg = 1.0 + 3.0 * ((-y_star_vals).abs() >= 0.4).float()
    torch.testing.assert_close(w_pos, w_neg)
    assert (w_pos[:2] == 4.0).all()
    assert (w_pos[-2:] == 4.0).all()
    assert (w_pos[2:5] == 1.0).all()

    # 3 & 4. Loss symmetry and gradient anti-symmetry
    B, K = 2, 20
    torch.manual_seed(99)
    scores1 = torch.randn(B, K, requires_grad=True)
    targets1 = torch.clamp(torch.randn(B, K) * 0.3, -0.5, 0.5)
    mask = torch.ones(B, K, dtype=torch.bool)

    loss1 = tail_loss(scores1, targets1, mask)
    loss1.backward()
    grad1 = scores1.grad.clone()

    scores2 = (-scores1.detach()).clone().requires_grad_(True)
    targets2 = -targets1.clone()

    loss2 = tail_loss(scores2, targets2, mask)
    loss2.backward()
    grad2 = scores2.grad.clone()

    # Loss is symmetric
    torch.testing.assert_close(loss1, loss2, atol=1e-6, rtol=1e-6)
    # Gradients are anti-symmetric: dL/d(-s) = - dL/ds
    torch.testing.assert_close(grad1, -grad2, atol=1e-6, rtol=1e-6)


def test_combined_loss_composition_and_padding_mask() -> None:
    """Verify CombinedRankingLoss = ListMLE + 1.5 * TailLoss and mask exclusion."""
    B, K = 3, 30
    torch.manual_seed(123)
    scores = torch.randn(B, K, requires_grad=True)
    t5 = torch.randn(B, K)
    ys = torch.clamp(torch.randn(B, K) * 0.3, -0.5, 0.5)

    mask = torch.ones(B, K, dtype=torch.bool)
    mask[:, 20:] = False  # slots 20-29 are padding

    comb_fn = CombinedRankingLoss(lambda_tail=1.5)
    tot, lmle, ltail = comb_fn(scores, t5, ys, mask)

    torch.testing.assert_close(tot, lmle + 1.5 * ltail)

    # Padding slots produce zero gradient
    tot.backward()
    assert torch.all(scores.grad[:, 20:] == 0.0)
    assert torch.any(scores.grad[:, :20].abs() > 0.0)
