"""Causal macro market-context feature builder (28 rolling expanding z-scored features)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def build_market_features(
    X: np.ndarray,
    mem: np.ndarray,
    feature_names: Sequence[str],
    mkt_windows: Sequence[int] = (1, 5, 20, 60),
) -> np.ndarray:
    """Compute strictly lagged market-level summary features for MASTER feature gating."""
    fi = {n: i for i, n in enumerate(feature_names)}

    def _col(name: str) -> np.ndarray:
        return np.where(mem, X[:, :, fi[name]], np.nan)

    lr = _col("log_ret")
    series = {
        "mkt_ret": np.nanmean(lr, axis=1),
        "disp": np.nanstd(lr, axis=1),
        "breadth": np.nanmean(
            np.where(np.isnan(lr), np.nan, (lr > 0).astype(np.float32)), axis=1
        ),
        "vol20": np.nanmean(_col("vol_20"), axis=1),
        "vratio": np.nanmean(_col("vol_ratio_20"), axis=1),
        "mom20": np.nanmean(_col("mom_20"), axis=1),
        "hl": np.nanmean(_col("hl_range"), axis=1),
    }
    base_df = pd.DataFrame(series)
    feats = [
        base_df.rolling(w, min_periods=w).mean().shift(1).add_suffix(f"_{w}")
        for w in mkt_windows
    ]
    merged = pd.concat(feats, axis=1)
    mu = merged.expanding(min_periods=60).mean().shift(1)
    sd = merged.expanding(min_periods=60).std().shift(1)
    z = ((merged - mu) / (sd + 1e-8)).clip(-5, 5).fillna(0.0)
    return z.values.astype(np.float32)
