"""Position-level sleeve aggregation and internal trade netting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def consolidate_positions(
    alloc_schedule: pd.DataFrame,
    cand_weights: dict[str, np.ndarray],
) -> np.ndarray:
    """Construct consolidated security positions W_{p,i,t} = sum_k a_{k,t} W_{k,i,t}."""
    cols = list(alloc_schedule.columns)
    first_w = next(iter(cand_weights.values()))
    T, N = first_w.shape
    assert alloc_schedule.shape[0] == T, (
        f"Schedule length {alloc_schedule.shape[0]} != {T}"
    )
    A = np.nan_to_num(alloc_schedule.to_numpy(dtype=np.float64), nan=0.0)
    W_cons = np.zeros((T, N), dtype=np.float64)
    for j, cname in enumerate(cols):
        aj = A[:, j]
        if float(np.abs(aj).max()) > 1e-12:
            W_cons += aj[:, None] * cand_weights[cname]
    return W_cons
