"""Tests for features pipeline, normalization, market context, and causal targets."""

from __future__ import annotations

import math

import numpy as np
import torch

from v27.config import FEATURE_CLUSTERS
from v27.features.market_context import build_market_features
from v27.features.normalizer import cs_normalize
from v27.features.stock_features import compute_34_stock_features
from v27.targets.target_builder import build_targets_and_lagged_risk


def test_compute_34_features_matches_expected_clusters():
    T, N = 270, 25
    rng = np.random.default_rng(11)
    rt = rng.normal(0.0004, 0.015, size=(T, N)).astype(np.float32)
    adj = 100.0 * np.cumprod(1.0 + rt, axis=0).astype(np.float32)
    op = (adj * (1.0 + rng.normal(0.0, 0.002, size=(T, N)))).astype(np.float32)
    hi = (
        np.maximum(op, adj) * (1.0 + np.abs(rng.normal(0.0, 0.004, size=(T, N))))
    ).astype(np.float32)
    lo = (
        np.minimum(op, adj) * (1.0 - np.abs(rng.normal(0.0, 0.004, size=(T, N))))
    ).astype(np.float32)
    vo = rng.uniform(1e5, 5e6, size=(T, N)).astype(np.float32)

    feat_dict, x_gpu, y_gpu = compute_34_stock_features(
        torch.from_numpy(op),
        torch.from_numpy(hi),
        torch.from_numpy(lo),
        torch.from_numpy(adj),
        torch.from_numpy(vo),
        torch.from_numpy(adj),
        torch.from_numpy(rt),
    )
    expected = [f for cols in FEATURE_CLUSTERS.values() for f in cols]
    assert len(feat_dict) == 34
    assert set(feat_dict.keys()) == set(expected)
    assert x_gpu.shape == (T, N, 34)
    assert y_gpu.shape == (T, N)


def test_cs_normalize_rankgauss_zeroes_non_members():
    rng = np.random.default_rng(3)
    X = rng.normal(0.0, 1.0, size=(20, 250, 4)).astype(np.float32)
    mem = np.zeros((20, 250), dtype=bool)
    mem[:, :200] = True
    Xn = cs_normalize(X, mem, mode="rankgauss")
    assert Xn.dtype == torch.float16
    assert torch.all(Xn[:, 200:, :] == 0.0)
    assert float(Xn[:, :200, :].float().mean().abs()) < 0.02
    assert 0.90 < float(Xn[:, :200, :].float().std()) < 1.05


def test_targets_use_strictly_lagged_sigma_prev_and_compounded_5d():
    names = [
        "log_ret",
        "vol_20",
        "vol_ratio_20",
        "mom_20",
        "hl_range",
        "vol_60",
        "beta_60",
        "mom_60",
    ]
    D, N = 12, 4
    X = np.zeros((D, N, len(names)), dtype=np.float32)
    X[:, :, 1] = np.arange(D)[:, None] * 0.01 + 0.01  # vol_20 increases each day
    X[:, :, 5] = 0.015  # vol_60
    X[:, :, 6] = 1.1  # beta_60
    Y = np.full((D, N), 0.01, dtype=np.float32)

    mem = np.ones((D, N), dtype=bool)
    M = build_market_features(X, mem, names)
    assert M.shape == (D, 28)

    targets, sig_prev, beta_prev, _ = build_targets_and_lagged_risk(
        X, Y, names, vol_floor=0.005
    )
    assert np.all(np.isnan(sig_prev[0]))
    assert np.all(np.isnan(beta_prev[0]))
    # At t=3, sigma_prev comes from t=2: max(0.03, 0.015) = 0.03
    assert math.isclose(float(sig_prev[3, 0]), 0.03, rel_tol=1e-5)
    expected_5d = (1.01**5) - 1.0
    assert math.isclose(float(targets["raw_5d"][2, 0]), expected_5d, rel_tol=1e-5)
    assert math.isclose(
        float(targets["risk_5d"][3, 0]),
        expected_5d / (0.03 * math.sqrt(5.0)),
        rel_tol=1e-4,
    )
