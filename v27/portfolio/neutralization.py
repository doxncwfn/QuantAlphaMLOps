"""Portfolio leg renormalization and lagged-beta residualization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def renorm_legs(w: pd.DataFrame) -> pd.DataFrame:
    """Normalize long leg sum = +1.0 and short leg sum = -1.0 (gross = 2.0)."""
    W = np.nan_to_num(w.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    pos = np.maximum(W, 0.0)
    neg = np.maximum(-W, 0.0)
    ps = pos.sum(axis=1, keepdims=True)
    ns = neg.sum(axis=1, keepdims=True)
    pos = np.where(ps > 1e-12, pos / np.maximum(ps, 1e-12), 0.0)
    neg = np.where(ns > 1e-12, neg / np.maximum(ns, 1e-12), 0.0)
    out = np.where((ps > 1e-12) & (ns > 1e-12), pos - neg, 0.0)
    return pd.DataFrame(out, index=w.index, columns=w.columns)


def neutralize_beta(
    sc_df: pd.DataFrame, beta_df: pd.DataFrame, min_stocks: int = 20
) -> pd.DataFrame:
    """Cross-sectional OLS residualization of prediction scores against lagged beta_60."""
    S = sc_df.to_numpy(dtype=np.float64)
    B = beta_df.to_numpy(dtype=np.float64)
    out = np.full_like(S, np.nan)
    valid = np.isfinite(S) & np.isfinite(B)
    for t in range(S.shape[0]):
        v = valid[t]
        if v.sum() < min_stocks:
            out[t] = S[t]
            continue
        sv, bv = S[t, v], B[t, v]
        bc = bv - bv.mean()
        denom = float(np.dot(bc, bc))
        if denom > 1e-12:
            slope = float(np.dot(sv - sv.mean(), bc) / denom)
            out[t, v] = sv - slope * bc
        else:
            out[t, v] = sv
    return pd.DataFrame(out, index=sc_df.index, columns=sc_df.columns)
