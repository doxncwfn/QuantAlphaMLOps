"""Target & zero-leakage (t-1) risk matrix construction for raw_1d, risk_1d, raw_5d, risk_5d."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from v27.config import DataConfig
from v27.features.market_context import build_market_features
from v27.features.normalizer import cs_normalize


@dataclass
class PreparedAlphaDataset:
    """Device-staged normalized features, 1d/5d raw & risk-adjusted targets, and lagged risk arrays."""

    calendar: np.ndarray
    active_ids: np.ndarray
    features: Sequence[str]
    slots: np.ndarray
    decision_indices: np.ndarray
    D: int
    N: int
    n_feat: int
    m_dim: int
    Xn_g: torch.Tensor
    Y_map: dict[str, torch.Tensor]
    sigma_prev: np.ndarray
    beta_prev: np.ndarray
    mom60_prev: np.ndarray
    slots_g: torch.Tensor
    bad_cs_g: torch.Tensor
    M_g: torch.Tensor
    look_off: torch.Tensor


def build_targets_and_lagged_risk(
    X: np.ndarray,
    Y: np.ndarray,
    features: Sequence[str],
    vol_floor: float = 0.005,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Construct raw/risk-adjusted 1d and 5d targets using strictly lagged (t-1) volatility."""
    D_, N_, _ = X.shape
    fi = {n: i for i, n in enumerate(features)}
    nan_row = np.full((1, N_), np.nan, dtype=np.float32)

    sigma_raw = np.fmax(X[:, :, fi["vol_20"]], X[:, :, fi["vol_60"]])
    sigma_prev = np.vstack([nan_row, sigma_raw[:-1]])
    sigma_prev = np.where(
        np.isnan(sigma_prev), np.nan, np.maximum(sigma_prev, vol_floor)
    ).astype(np.float32)
    beta_prev = np.vstack([nan_row, X[:-1, :, fi["beta_60"]]]).astype(np.float32)
    mom60_prev = np.vstack([nan_row, X[:-1, :, fi["mom_60"]]]).astype(np.float32)

    one_p = 1.0 + Y.astype(np.float64)
    y_raw_5d = np.full((D_, N_), np.nan, dtype=np.float32)
    if D_ > 5:
        y_raw_5d[:-4] = (
            one_p[:-4] * one_p[1:-3] * one_p[2:-2] * one_p[3:-1] * one_p[4:] - 1.0
        ).astype(np.float32)
    y_risk_1d = np.where(
        np.isfinite(Y) & np.isfinite(sigma_prev), Y / sigma_prev, np.nan
    ).astype(np.float32)
    y_risk_5d = np.where(
        np.isfinite(y_raw_5d) & np.isfinite(sigma_prev),
        y_raw_5d / (sigma_prev * math.sqrt(5.0)),
        np.nan,
    ).astype(np.float32)

    targets = {
        "raw_1d": Y.astype(np.float32, copy=False),
        "risk_1d": y_risk_1d,
        "raw_5d": y_raw_5d,
        "risk_5d": y_risk_5d,
    }
    return targets, sigma_prev, beta_prev, mom60_prev


def prepare_alpha_dataset(
    cache: dict[str, Any],
    data_cfg: DataConfig | None = None,
    device: torch.device | None = None,
) -> PreparedAlphaDataset:
    """Assemble normalized cross-sectional features, macro context, targets, and indexers."""
    data_cfg = data_cfg or DataConfig()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y, slots, features = (
        cache["X_dense"],
        cache["Y_dense"],
        cache["slots"],
        cache["FEATURES"],
    )
    D_, N_, NF_ = X.shape

    mem = np.zeros((D_, N_), dtype=bool)
    rr = np.repeat(np.arange(D_), slots.shape[1])
    cc = slots.reshape(-1)
    ok = cc >= 0
    mem[rr[ok], cc[ok]] = True

    targets_np, sigma_prev, beta_prev, mom60_prev = build_targets_and_lagged_risk(
        X, Y, features, vol_floor=data_cfg.vol_floor
    )
    M = build_market_features(X, mem, features, mkt_windows=data_cfg.mkt_windows)
    Xn = cs_normalize(
        X, mem, mode=data_cfg.feature_norm, feature_clip=data_cfg.feature_clip
    )

    return PreparedAlphaDataset(
        calendar=cache["calendar"],
        active_ids=cache["active_ids"],
        features=features,
        slots=slots,
        decision_indices=cache["decision_indices"],
        D=D_,
        N=N_,
        n_feat=NF_,
        m_dim=M.shape[1],
        Xn_g=Xn.to(dev),
        Y_map={k: torch.from_numpy(v).to(dev) for k, v in targets_np.items()},
        sigma_prev=sigma_prev,
        beta_prev=beta_prev,
        mom60_prev=mom60_prev,
        slots_g=torch.from_numpy(slots).long().to(dev),
        bad_cs_g=torch.from_numpy(cache["bad_cs"]).to(dev),
        M_g=torch.from_numpy(M).to(dev),
        look_off=torch.arange(-data_cfg.lookback, 0, device=dev),
    )
