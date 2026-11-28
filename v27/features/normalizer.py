"""Cross-sectional normalization for dense feature tensors."""

from __future__ import annotations

import math

import numpy as np
import torch


@torch.no_grad()
def cs_normalize(
    X: np.ndarray,
    mem: np.ndarray,
    mode: str = "rankgauss",
    feature_clip: float = 3.0,
    chunk: int = 128,
) -> torch.Tensor:
    """Normalize each (date, feature) cross-section across active universe members."""
    D_, N_, NF_ = X.shape
    out = torch.empty((D_, N_, NF_), dtype=torch.float16)
    for s in range(0, D_, chunk):
        e = min(D_, s + chunk)
        x = torch.from_numpy(X[s:e]).float()
        m = torch.from_numpy(mem[s:e]).unsqueeze(-1).expand_as(x)
        valid = torch.isfinite(x) & m
        use = valid if valid.any() else torch.isfinite(x)
        if mode == "rankgauss":
            xt = x.permute(0, 2, 1)
            ut = use.permute(0, 2, 1)
            big = torch.tensor(1e30, dtype=torch.float32)
            v_hi = torch.where(ut, xt, big)
            v_lo = torch.where(ut, xt, -big)
            idx_hi = v_hi.argsort(dim=-1, descending=False)
            cnt_valid = ut.float().sum(dim=-1, keepdim=True)
            n_ref = cnt_valid.clamp(min=1.0)
            pos = torch.arange(N_, dtype=torch.float32).view(1, 1, N_).expand_as(xt)
            hi_c = torch.empty_like(xt).scatter_(-1, idx_hi, pos)
            idx_lo = v_lo.argsort(dim=-1, descending=True)
            pos_desc = cnt_valid - 1.0 - pos
            lo_c = torch.empty_like(xt).scatter_(-1, idx_lo, pos_desc)
            u = 0.5 * (lo_c + hi_c) / n_ref
            lo_b = 0.5 / n_ref
            u = torch.minimum(torch.maximum(u, lo_b), 1.0 - lo_b)
            z = (math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)).permute(0, 2, 1)
        else:
            xm = x.masked_fill(~use, float("nan"))
            med = torch.nanmedian(xm, dim=1, keepdim=True).values
            mad = torch.nanmedian((xm - med).abs(), dim=1, keepdim=True).values * 1.4826
            z = ((x - med) / mad.clamp(min=1e-8)).clamp(-feature_clip, feature_clip)
        z = torch.where(valid, z, torch.zeros_like(z))
        out[s:e] = z.half().cpu()
    return out
