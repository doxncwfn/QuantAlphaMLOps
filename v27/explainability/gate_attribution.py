"""MASTER Gate feature-cluster importance aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from v27.config import FEATURE_CLUSTERS


def summarize_master_gate_clusters(
    alpha_df: pd.DataFrame,
    clusters: dict[str, list[str]] = FEATURE_CLUSTERS,
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """Aggregate per-feature MASTER gate multipliers alpha_t into 5 economic cluster weights."""
    a_dt = alpha_df.copy()
    a_dt.index = pd.to_datetime(a_dt.index)
    a_dt = a_dt.sort_index()

    cluster_cols = {}
    for cname, cols in clusters.items():
        present = [c for c in cols if c in a_dt.columns]
        if present:
            cluster_cols[cname] = a_dt[present].mean(axis=1)
    cluster_df = pd.DataFrame(cluster_cols, index=a_dt.index)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cluster_df.to_csv(out_csv)
    return cluster_df


def cluster_share_table(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Compute time-average multiplier, relative share (%), and volatility per cluster."""
    mu = cluster_df.mean(axis=0)
    sd = cluster_df.std(axis=0)
    share = 100.0 * mu / max(float(mu.sum()), 1e-12)
    return pd.DataFrame(
        {
            "Cluster": cluster_df.columns,
            "MeanAlphaMult": mu.to_numpy(dtype=np.float64),
            "StdAlphaMult": sd.to_numpy(dtype=np.float64),
            "RelativeShare(%)": share.to_numpy(dtype=np.float64),
        }
    )
