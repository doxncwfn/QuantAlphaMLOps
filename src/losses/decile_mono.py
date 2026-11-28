"""§1 Soft-Decile Monotonicity + Long-Short Spread Loss.

Implementation of Section 1 of specs/proposed_loss.md adapted for 5-day horizon and y_star target:
- Step A: Soft rank using torchsort (with grid softmax fallback)
- Step B: Soft bucket assignment with centroids c_k = (k - 0.5) * (N / K)
- Step C: Bucket means mu_k of y_star
- Step D: Monotonicity hinge with margin m_mono = 0.02
- Step E: Standardized long-short spread L_ls = -(mu_K - mu_1) / (sigma_y + eps)
- Step F: Optional Huber auxiliary loss (lambda_hub * L_hub)
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

try:
    import torchsort

    _HAS_TORCHSORT = True
except ImportError:
    torchsort = None
    _HAS_TORCHSORT = False


def soft_rank(
    y_hat: torch.Tensor,
    regularization: float = 0.01,
    allow_fallback: bool = False,
) -> torch.Tensor:
    """Compute soft rank along dimension -1 (ascending: smallest value -> rank ~1, largest -> rank ~N).

    Uses torchsort when available; on MPS transfers to CPU for torchsort and back to MPS.
    Grid-softmax fallback is disabled by default unless allow_fallback=True.
    """
    std_y = (
        torch.sqrt(torch.var(y_hat, dim=-1, keepdim=True, unbiased=True) + 1e-6)
        .clamp(min=1e-2)
        .detach()
    )
    if _HAS_TORCHSORT and torchsort is not None and not allow_fallback:
        N = float(y_hat.shape[-1])
        scale = max(1.0, N / 400.0)
        if y_hat.device.type == "mps":
            y_cpu = y_hat.float().cpu()
            std_cpu = (
                torch.sqrt(torch.var(y_cpu, dim=-1, keepdim=True, unbiased=True) + 1e-6)
                .clamp(min=1e-2)
                .detach()
            )
            s_unit = (y_cpu / std_cpu) * scale
            s_plain = y_cpu * scale
            s_scaled = s_plain + (s_unit - s_plain).detach()
            r_cpu = torchsort.soft_rank(
                s_scaled, regularization_strength=regularization
            )
            return r_cpu.to(device=y_hat.device, dtype=torch.float32)
        s_unit = (y_hat.float() / std_y.float()) * scale
        s_plain = y_hat.float() * scale
        s_scaled = s_plain + (s_unit - s_plain).detach()
        return torchsort.soft_rank(s_scaled, regularization_strength=regularization).to(
            dtype=torch.float32
        )

    if not allow_fallback:
        raise RuntimeError(
            "torchsort is required for soft_rank when allow_fallback=False."
        )

    # Explicit fallback: grid-based soft rank approximation (only when allow_fallback=True)
    _B, N = y_hat.shape
    g = torch.arange(1, N + 1, device=y_hat.device, dtype=torch.float32)  # [N]
    s_norm = (y_hat / std_y) * (N / 6.0) + (N / 2.0)  # [B, N]
    diff = g.unsqueeze(0).unsqueeze(1) - s_norm.unsqueeze(-1)  # [B, N, N_g]
    weights = torch.softmax(-(diff**2) / (2.0 * ((regularization * N) ** 2)), dim=-1)
    return (weights * g.unsqueeze(0).unsqueeze(1)).sum(dim=-1)


def decile_mono_loss(
    y_hat: torch.Tensor,
    y_star: torch.Tensor,
    mask: torch.Tensor,
    K: int = 10,
    tau_sort: float = 0.01,
    tau_assign: float | None = None,
    m_mono: float = 0.02,  # adapted for 5-day y_star in [-0.5, 0.5]
    lam_ls: float = 0.5,  # adapted for 5-day
    lam_hub: float = 0.0,  # 0.0 for Variant A, 0.2 for Variant B
    beta: float = 1.0,
    y_norm: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute §1 Soft-Decile Monotonicity + Long-Short Spread Loss.

    Parameters
    ----------
    y_hat : torch.Tensor
        Model prediction scores [B, N] or [N].
    y_star : torch.Tensor
        Target cross-sectional ranks in [-0.5, 0.5] [B, N] or [N].
    mask : torch.Tensor
        Validity mask [B, N] or [N].
    K : int
        Number of soft deciles (default 10).
    tau_sort : float
        Regularization parameter for soft ranking (default 0.01).
    tau_assign : float | None
        Bandwidth for Gaussian bucket assignment. Defaults to 0.3 * (N / K).
    m_mono : float
        Minimum separation margin between adjacent soft deciles (default 0.02).
    lam_ls : float
        Weight on standardized long-short spread (default 0.5).
    lam_hub : float
        Weight on auxiliary Huber loss (0.0 for §1-A, 0.2 for §1-B).
    beta : float
        Smooth L1 Huber threshold (default 0.1).

    Returns
    -------
    tuple[torch.Tensor, dict[str, torch.Tensor]]
        total_loss, diagnostic dict with L_mono, L_ls, L_hub, and mu.
    """
    y = y_norm if y_norm is not None else y_star
    assert y_hat.dtype == torch.float32
    assert y.dtype == torch.float32

    was_1d = y_hat.dim() == 1
    if was_1d:
        y_hat = y_hat.unsqueeze(0)
        y_star = y_star.unsqueeze(0)
        y = y.unsqueeze(0)
        mask = mask.unsqueeze(0)

    B = y_hat.shape[0]
    eps = 1e-8

    l_mono_list = []
    l_ls_list = []
    l_hub_list = []
    all_mus = []

    for b in range(B):
        valid = mask[b].bool()
        n_valid = int(valid.sum())
        if n_valid < K:
            continue

        s = y_hat[b][valid].unsqueeze(0)  # [1, N_v]
        y_star[b][valid].unsqueeze(0)  # [1, N_v]
        y_norm_b = y[b][valid].unsqueeze(0)  # [1, N_v]
        N_v = float(n_valid)

        # Step A: Soft rank
        r = soft_rank(s, regularization=tau_sort)  # [1, N_v]

        # Step B: Soft bucket assignment
        eff_tau_assign = tau_assign if tau_assign is not None else 0.3 * (N_v / K)
        c = (torch.arange(1, K + 1, device=y_hat.device, dtype=torch.float32) - 0.5) * (
            N_v / K
        )  # [K]
        d = r.unsqueeze(-1) - c.unsqueeze(0).unsqueeze(0)  # [1, N_v, K]
        logits = -(d**2) / (2.0 * (eff_tau_assign**2))
        p = torch.softmax(logits, dim=-1)  # [1, N_v, K]

        # Step C: Bucket means
        M = p.sum(dim=1)  # [1, K]
        num = (p * y_norm_b.unsqueeze(-1)).sum(dim=1)  # [1, K]
        mu = num / (M + eps)  # [1, K]
        all_mus.append(mu.squeeze(0))

        # Step D: Monotonicity hinge with margin (guarded against empty bucket M_k < 5.0)
        if float(M.min().item()) >= 5.0:
            diff = mu[:, 1:] - mu[:, :-1]  # [1, K-1]
            l_mono_b = (torch.relu(m_mono - diff) ** 2).mean()
            l_mono_list.append(l_mono_b)
        else:
            logger.warning(
                "Date %d has min soft bucket mass M_k=%.4f < 5.0; excluding date from L_mono.",
                b,
                float(M.min().item()),
            )

        # Step E: LS spread (standardized by masked std of y_norm)
        sigma_y = torch.sqrt(torch.var(y_norm_b, unbiased=True) + 1e-6).clamp(min=1e-2)
        spread = mu[:, -1] - mu[:, 0]  # [1]
        l_ls_b = -(spread / sigma_y).mean()
        l_ls_list.append(l_ls_b)

        # Step F: Huber auxiliary
        if lam_hub > 0.0:
            l_hub_b = F.smooth_l1_loss(s, y_norm_b, beta=beta, reduction="mean")
            l_hub_list.append(l_hub_b)

    if not l_ls_list:
        logger.warning("All dates had n_valid < K in decile_mono_loss; returning 0.")
        return y_hat.sum() * 0.0, {
            "L_mono": torch.tensor(0.0, device=y_hat.device),
            "L_ls": torch.tensor(0.0, device=y_hat.device),
            "L_hub": torch.tensor(0.0, device=y_hat.device),
            "mu": torch.empty(0, device=y_hat.device),
        }

    if l_mono_list:
        L_mono = torch.stack(l_mono_list).mean()
    else:
        logger.warning(
            "All dates skipped for L_mono (M_k < 5.0); setting L_mono = 0.0."
        )
        L_mono = y_hat.sum() * 0.0
    L_ls = torch.stack(l_ls_list).mean()
    L_hub = (
        torch.stack(l_hub_list).mean()
        if l_hub_list
        else torch.tensor(0.0, device=y_hat.device)
    )

    total_loss = L_mono + lam_ls * L_ls + lam_hub * L_hub
    diag = {
        "L_mono": L_mono.detach(),
        "L_ls": L_ls.detach(),
        "L_hub": L_hub.detach(),
        "mu": torch.stack(all_mus).detach()
        if all_mus
        else torch.empty(0, device=y_hat.device),
    }
    return total_loss, diag


class DecileMonotonicityLoss(nn.Module):
    """PyTorch Module wrapper for §1 Decile Monotonicity + LS Spread loss."""

    def __init__(
        self,
        K: int = 10,
        tau_sort: float = 0.01,
        tau_assign: float | None = None,
        m_mono: float = 0.02,
        lam_ls: float = 0.5,
        lam_hub: float = 0.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.K = K
        self.tau_sort = tau_sort
        self.tau_assign = tau_assign
        self.m_mono = m_mono
        self.lam_ls = lam_ls
        self.lam_hub = lam_hub
        self.beta = beta
        self.last_diag: dict[str, torch.Tensor] = {}

    def forward(
        self,
        scores: torch.Tensor,
        target_5d: torch.Tensor,
        y_star: torch.Tensor,
        mask: torch.Tensor,
        y_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss and return (total_loss, L_mono, L_ls)."""
        loss, diag = decile_mono_loss(
            y_hat=scores,
            y_star=y_star,
            mask=mask,
            K=self.K,
            tau_sort=self.tau_sort,
            tau_assign=self.tau_assign,
            m_mono=self.m_mono,
            lam_ls=self.lam_ls,
            lam_hub=self.lam_hub,
            beta=self.beta,
            y_norm=y_norm,
        )
        self.last_diag = diag
        return loss, diag["L_mono"], diag["L_ls"]
