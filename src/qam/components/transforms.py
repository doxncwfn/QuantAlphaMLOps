"""Cross-sectional transforms, applied at load time on one decision date's window.

Contract: fn(x: np.ndarray[k, L, F], mask: np.ndarray[k] bool, **kwargs) -> np.ndarray[k, L, F].
Statistics are computed across the stock axis (axis 0), separately for every day and
feature, using only the real slots (mask True), i.e. within the k stocks the model sees.
Missing cells arrive as NaN: ignore them in statistics and leave them NaN; the framework
sets them to 0 after the last transform. Masked slots must not influence the result; the
framework zeroes them afterwards.
"""

from __future__ import annotations

import warnings

import numpy as np

from qam.registry import register


@register("transform", "cs_zscore")
def cs_zscore(x: np.ndarray, mask: np.ndarray, eps: float = 1e-8, ddof: int = 1) -> np.ndarray:
    """(x - mean) / (std + eps) per day and feature over the values present. ddof=1 matches
    the legacy polars z-score. A day and feature with ddof or fewer values becomes NaN."""
    real = x[mask]
    n = (~np.isnan(real)).sum(axis=0, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty or single-value cells
        mean = np.nanmean(real, axis=0, keepdims=True)
        std = np.nanstd(real, axis=0, ddof=ddof, keepdims=True)
        out = (x - mean) / (std + eps)
    return np.where(n > ddof, out, np.nan).astype(x.dtype)


@register("transform", "clip")
def clip(x: np.ndarray, mask: np.ndarray, lo: float = -3.0, hi: float = 3.0) -> np.ndarray:
    return np.clip(x, lo, hi)


@register("transform", "cs_rank")
def cs_rank(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Uniform rank in [-0.5, 0.5] among the real slots with a value (ties broken by position)."""
    out = np.full_like(x, np.nan)
    real = x[mask]
    valid = ~np.isnan(real)
    n = valid.sum(axis=0, keepdims=True)
    ranks = np.where(valid, real, np.inf).argsort(axis=0, kind="stable").argsort(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[mask] = np.where(valid & (n >= 2), ranks / (n - 1) - 0.5, np.nan)
    return out
