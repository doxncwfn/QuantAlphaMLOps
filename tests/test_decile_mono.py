"""Unit tests for §1 Decile Monotonicity + Long-Short Spread Loss.

Verifies:
- Test 1: Perfect prediction (monotonic, high positive spread)
- Test 2: Random/Zero prediction (flat spread)
- Test 3: Adversarial prediction (inverted ranks, high penalty)
- Test 4: Masking equivalence (padded stocks do not affect valid loss)
- Test 5: Toy soft rank test from spec
"""

import torch

from src.losses.decile_mono import DecileMonotonicityLoss, decile_mono_loss, soft_rank


def test_soft_rank_toy():
    """Verify toy test from Part 1.2: y_hat = [3, 2, 1, 0, -1] -> r ≈ [5, 4, 3, 2, 1]."""
    y_hat = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])
    r = soft_rank(y_hat, regularization=0.01)
    expected = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    assert torch.allclose(r, expected, atol=0.1), f"Expected {expected}, got {r}"


def test_perfect_prediction():
    """Test 1: Perfect prediction gives near-zero L_mono, L_ls ≈ -3 (in y_star units), L_hub near zero."""
    N = 1000
    y_star = torch.linspace(-0.5, 0.5, N).unsqueeze(0)
    # y_hat matches order of y_star
    y_hat = torch.linspace(-3.0, 3.0, N).unsqueeze(0)
    mask = torch.ones(1, N, dtype=torch.bool)

    _loss, diag = decile_mono_loss(
        y_hat=y_hat,
        y_star=y_star,
        mask=mask,
        K=10,
        tau_sort=0.01,
        tau_assign=30.0,
        m_mono=0.02,
        lam_ls=0.5,
        lam_hub=0.2,
        beta=0.1,
    )

    # L_mono should be near zero (no inversions, diff > 0.02)
    assert diag["L_mono"].item() < 1e-3, f"L_mono too large: {diag['L_mono'].item()}"
    # L_ls should be strongly negative (approx -3 to -3.5)
    assert diag["L_ls"].item() < -2.5, (
        f"L_ls not sufficiently negative: {diag['L_ls'].item()}"
    )


def test_random_zeros_prediction():
    """Test 2: Random/zero prediction gives L_mono ≈ 0, L_ls = 0, L_hub high."""
    N = 1000
    y_star = torch.linspace(-0.5, 0.5, N).unsqueeze(0)
    y_hat = torch.zeros(1, N)
    mask = torch.ones(1, N, dtype=torch.bool)

    _loss, diag = decile_mono_loss(
        y_hat=y_hat,
        y_star=y_star,
        mask=mask,
        K=10,
        tau_sort=0.01,
        tau_assign=30.0,
        m_mono=0.02,
        lam_ls=0.5,
        lam_hub=0.2,
        beta=0.1,
    )

    # Monotonicity trivial margin penalty (0.02^2 = 0.0004)
    assert diag["L_mono"].item() <= 0.001
    # LS spread should be essentially 0
    assert abs(diag["L_ls"].item()) < 1e-4
    # Huber loss should be significant (~0.20 for zeros vs [-0.5, 0.5])
    assert diag["L_hub"].item() > 0.15


def test_adversarial_prediction():
    """Test 3: Adversarial prediction gives high L_mono, L_ls > 0, high L_hub."""
    N = 1000
    y_star = torch.linspace(-0.5, 0.5, N).unsqueeze(0)
    y_hat = torch.linspace(3.0, -3.0, N).unsqueeze(0)  # Inverted
    mask = torch.ones(1, N, dtype=torch.bool)

    loss, diag = decile_mono_loss(
        y_hat=y_hat,
        y_star=y_star,
        mask=mask,
        K=10,
        tau_sort=0.01,
        tau_assign=30.0,
        m_mono=0.02,
        lam_ls=0.5,
        lam_hub=0.2,
        beta=0.1,
    )

    # Monotonicity penalty should be high (violations in every adjacent pair)
    assert diag["L_mono"].item() > 0.015, f"L_mono too low: {diag['L_mono'].item()}"
    # L_ls should be positive (negative spread: top bucket has lower returns)
    assert diag["L_ls"].item() > 2.5, f"L_ls should be positive: {diag['L_ls'].item()}"
    # Total loss must be substantially worse than perfect prediction
    assert loss.item() > 1.0


def test_masking_equivalence():
    """Test 4: Masking — if half the stocks are masked, loss equals unmasked calculation on valid half."""
    N_total = 1000
    N_valid = 600
    torch.manual_seed(42)

    s_valid = torch.randn(1, N_valid)
    y_valid = torch.linspace(-0.5, 0.5, N_valid).unsqueeze(0)

    s_padded = torch.zeros(1, N_total)
    y_padded = torch.zeros(1, N_total)
    mask = torch.zeros(1, N_total, dtype=torch.bool)

    s_padded[0, :N_valid] = s_valid[0]
    y_padded[0, :N_valid] = y_valid[0]
    mask[0, :N_valid] = True

    loss_valid, _ = decile_mono_loss(
        y_hat=s_valid,
        y_star=y_valid,
        mask=torch.ones(1, N_valid, dtype=torch.bool),
        lam_hub=0.2,
    )
    loss_masked, _ = decile_mono_loss(
        y_hat=s_padded,
        y_star=y_padded,
        mask=mask,
        lam_hub=0.2,
    )

    assert abs(loss_valid.item() - loss_masked.item()) < 1e-5


def test_module_wrapper():
    """Verify DecileMonotonicityLoss PyTorch nn.Module interface."""
    loss_module = DecileMonotonicityLoss(lam_hub=0.2)
    s = torch.randn(2, 100, requires_grad=True)
    t5 = torch.randn(2, 100)
    ys = torch.linspace(-0.5, 0.5, 100).repeat(2, 1)
    mask = torch.ones(2, 100, dtype=torch.bool)

    loss, _l_mono, _l_ls = loss_module(s, t5, ys, mask)
    loss.backward()

    assert s.grad is not None
    assert torch.isfinite(s.grad).all()
    assert loss.item() != 0.0


def test_c7_huber_target_and_beta():
    """Fix 1.3 (C7): Assert default beta == 1.0 and the target passed to Huber has per-date std in [0.95, 1.05]."""
    import inspect
    from unittest.mock import patch

    import scipy.special as sp
    import torch.nn.functional as F

    sig = inspect.signature(decile_mono_loss)
    assert sig.parameters["beta"].default == 1.0
    mod = DecileMonotonicityLoss(lam_hub=0.2)
    assert mod.beta == 1.0

    B, N = 2, 500
    y_star = torch.linspace(-0.5, 0.5, N, dtype=torch.float32).unsqueeze(0).repeat(B, 1)
    ranks = torch.arange(1, N + 1, dtype=torch.float32) / (N + 1.0)
    y_norm = (
        torch.from_numpy(sp.ndtri(ranks.numpy()).astype("float32"))
        .unsqueeze(0)
        .repeat(B, 1)
    )
    y_hat = torch.randn(B, N, dtype=torch.float32)
    mask = torch.ones(B, N, dtype=torch.bool)

    captured_targets = []
    captured_betas = []
    orig_smooth_l1 = F.smooth_l1_loss

    def spy_smooth_l1(input, target, beta=1.0, **kwargs):
        captured_targets.append(target.detach().clone())
        captured_betas.append(beta)
        return orig_smooth_l1(input, target, beta=beta, **kwargs)

    with patch("src.losses.decile_mono.F.smooth_l1_loss", side_effect=spy_smooth_l1):
        _loss, _diag = decile_mono_loss(
            y_hat=y_hat,
            y_star=y_star,
            mask=mask,
            lam_hub=0.2,
            y_norm=y_norm,
        )

    assert len(captured_targets) == B
    for b_idx, t_b in enumerate(captured_targets):
        std_val = float(torch.std(t_b, unbiased=True).item())
        assert 0.95 <= std_val <= 1.05, (
            f"Date {b_idx} Huber target std={std_val} outside [0.95, 1.05]"
        )
        assert captured_betas[b_idx] == 1.0


def test_c8_variance_clamp_inside_sqrt():
    """Fix 1.4 (C8): Feed constant y_hat (var = 0) and constant y_star; assert loss and gradients are finite."""
    N = 100
    y_hat = torch.zeros(1, N, dtype=torch.float32, requires_grad=True)
    y_star = torch.linspace(-0.5, 0.5, N, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, N, dtype=torch.bool)

    loss, _diag = decile_mono_loss(
        y_hat=y_hat,
        y_star=y_star,
        mask=mask,
        lam_hub=0.2,
    )
    loss.backward()

    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert y_hat.grad is not None
    assert torch.isfinite(y_hat.grad).all(), (
        "Gradients contain NaN or Inf on constant y_hat"
    )

    # Also test constant y_star (var = 0)
    y_hat2 = torch.randn(1, N, dtype=torch.float32, requires_grad=True)
    y_star_const = torch.zeros(1, N, dtype=torch.float32)
    loss2, _ = decile_mono_loss(
        y_hat=y_hat2, y_star=y_star_const, mask=mask, lam_hub=0.2
    )
    loss2.backward()
    assert torch.isfinite(loss2)
    assert y_hat2.grad is not None
    assert torch.isfinite(y_hat2.grad).all()


def test_c2_mps_cuda_cpu_soft_rank_equivalence():
    """Fix 1.5 (C2): Given identical inputs, assert max |L_cpu - L_mps| < 1e-5,
    max |L_cpu - L_cuda| < 1e-5, and gradient norms agree within 1e-5.
    (Skip CUDA branch if CUDA unavailable; print SKIP with reason.)
    """
    torch.manual_seed(123)
    B, N = 2, 200
    s_base = torch.randn(B, N, dtype=torch.float32)
    y_base = torch.linspace(-0.5, 0.5, N, dtype=torch.float32).unsqueeze(0).repeat(B, 1)
    yn_base = (
        torch.linspace(-2.0, 2.0, N, dtype=torch.float32).unsqueeze(0).repeat(B, 1)
    )
    m_base = torch.ones(B, N, dtype=torch.bool)

    s_cpu = s_base.clone().requires_grad_(True)
    loss_cpu, _ = decile_mono_loss(
        y_hat=s_cpu,
        y_star=y_base,
        mask=m_base,
        lam_hub=0.2,
        y_norm=yn_base,
    )
    loss_cpu.backward()
    grad_norm_cpu = float(s_cpu.grad.norm().item())

    if torch.backends.mps.is_available():
        s_mps = s_base.to("mps").detach().requires_grad_(True)
        loss_mps, _ = decile_mono_loss(
            y_hat=s_mps,
            y_star=y_base.to("mps"),
            mask=m_base.to("mps"),
            lam_hub=0.2,
            y_norm=yn_base.to("mps"),
        )
        loss_mps.backward()
        grad_norm_mps = float(s_mps.grad.cpu().norm().item())
        diff_mps = abs(float(loss_cpu.item()) - float(loss_mps.cpu().item()))
        grad_diff_mps = abs(grad_norm_cpu - grad_norm_mps)
        assert diff_mps < 1e-5, f"max |L_cpu - L_mps| = {diff_mps} >= 1e-5"
        assert grad_diff_mps < 1e-5, f"|grad_cpu - grad_mps| = {grad_diff_mps} >= 1e-5"
    else:
        print("SKIP MPS: torch.backends.mps.is_available() is False")

    if torch.cuda.is_available():
        s_cuda = s_base.to("cuda").detach().requires_grad_(True)
        loss_cuda, _ = decile_mono_loss(
            y_hat=s_cuda,
            y_star=y_base.to("cuda"),
            mask=m_base.to("cuda"),
            lam_hub=0.2,
            y_norm=yn_base.to("cuda"),
        )
        loss_cuda.backward()
        grad_norm_cuda = float(s_cuda.grad.cpu().norm().item())
        diff_cuda = abs(float(loss_cpu.item()) - float(loss_cuda.cpu().item()))
        grad_diff_cuda = abs(grad_norm_cpu - grad_norm_cuda)
        assert diff_cuda < 1e-5, f"max |L_cpu - L_cuda| = {diff_cuda} >= 1e-5"
        assert grad_diff_cuda < 1e-5, (
            f"|grad_cpu - grad_cuda| = {grad_diff_cuda} >= 1e-5"
        )
    else:
        print(
            "SKIP CUDA: torch.cuda.is_available() is False on Apple Silicon macOS host"
        )


def test_c4_empty_bucket_mk_less_than_5_guard():
    """Fix 1.7 (C4): Extreme predictions producing one empty bucket (M_k < 5.0);
    assert loss is finite and L_mono equals the mean over non-skipped dates, and returns 0 when all dates skipped.
    """
    N = 100
    # Date 0: well-separated predictions -> all M_k >= 5.0 (each ~10.0)
    s_good = torch.linspace(-100.0, 100.0, N, dtype=torch.float32)
    # Date 1: extreme tied predictions -> middle buckets have M_k == 0.0 < 5.0
    s_bad = torch.zeros(N, dtype=torch.float32)

    y_star_0 = torch.linspace(
        0.5, -0.5, N, dtype=torch.float32
    )  # non-zero L_mono on Date 0
    y_star_1 = torch.linspace(0.5, -0.5, N, dtype=torch.float32)
    mask_good = torch.ones(N, dtype=torch.bool)

    s_batch = torch.stack([s_good, s_bad], dim=0)
    y_batch = torch.stack([y_star_0, y_star_1], dim=0)
    mask_batch = torch.stack([mask_good, mask_good], dim=0)

    loss_batch, diag_batch = decile_mono_loss(
        y_hat=s_batch,
        y_star=y_batch,
        mask=mask_batch,
        tau_assign=1.5,
        lam_ls=0.0,
        lam_hub=0.0,
    )
    loss_single, diag_single = decile_mono_loss(
        y_hat=s_good.unsqueeze(0),
        y_star=y_star_0.unsqueeze(0),
        mask=mask_good.unsqueeze(0),
        tau_assign=1.5,
        lam_ls=0.0,
        lam_hub=0.0,
    )

    assert torch.isfinite(loss_batch)
    assert diag_single["L_mono"].item() > 0.0
    assert abs(float(loss_batch.item()) - float(loss_single.item())) < 1e-6
    assert (
        abs(float(diag_batch["L_mono"].item()) - float(diag_single["L_mono"].item()))
        < 1e-6
    )

    # All dates skipped -> L_mono returns 0.0
    loss_all_skipped, diag_skipped = decile_mono_loss(
        y_hat=s_bad.unsqueeze(0),
        y_star=y_star_1.unsqueeze(0),
        mask=mask_good.unsqueeze(0),
        tau_assign=1.5,
        lam_ls=0.0,
        lam_hub=0.0,
    )
    assert torch.isfinite(loss_all_skipped)
    assert float(loss_all_skipped.item()) == 0.0
    assert float(diag_skipped["L_mono"].item()) == 0.0
