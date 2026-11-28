"""Decile portfolio monotonicity and D10-D1 spread diagnostics for cross-sectional signals."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from v27.config import EXPERIMENTS_V7
from v27.data.bundle import V7DataBundle


def compute_decile_monotonicity(
    scores_df: pd.DataFrame,
    ret1d_df: pd.DataFrame,
    n_deciles: int = 10,
    ann_days: int = 252,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute equal-weight D1..D10 daily/annualized returns, D10-D1 spread, and monotonicity."""
    S = scores_df.to_numpy(dtype=np.float64)
    R = ret1d_df.to_numpy(dtype=np.float64)
    T, _ = S.shape
    dec_rets = np.full((T, n_deciles), np.nan, dtype=np.float64)

    for t in range(T):
        v = np.isfinite(S[t]) & np.isfinite(R[t])
        if v.sum() < n_deciles * 2:
            continue
        sv, rv = S[t, v], R[t, v]
        ranks = np.argsort(np.argsort(sv))
        bins = np.floor(ranks * n_deciles / len(sv)).astype(int)
        bins = np.clip(bins, 0, n_deciles - 1)
        for d in range(n_deciles):
            m = bins == d
            if m.any():
                dec_rets[t, d] = float(rv[m].mean())

    dec_df = pd.DataFrame(
        dec_rets, index=scores_df.index, columns=[f"D{d + 1}" for d in range(n_deciles)]
    )
    mean_ann = dec_df.mean() * ann_days * 100.0
    vol_ann = dec_df.std(ddof=1) * math.sqrt(ann_days) * 100.0
    sh_ann = (dec_df.mean() / (dec_df.std(ddof=1) + 1e-12)) * math.sqrt(ann_days)

    summary_df = pd.DataFrame(
        {
            "Decile": [f"D{d + 1}" for d in range(n_deciles)],
            "AnnRet(%)": mean_ann.to_numpy(dtype=np.float64),
            "AnnVol(%)": vol_ann.to_numpy(dtype=np.float64),
            "Sharpe": sh_ann.to_numpy(dtype=np.float64),
        }
    )

    rho_mono, _ = spearmanr(
        np.arange(1, n_deciles + 1), mean_ann.to_numpy(dtype=np.float64)
    )
    spread_ser = dec_df[f"D{n_deciles}"] - dec_df["D1"]
    spread_ann = float(spread_ser.mean() * ann_days * 100.0)
    spread_sh = float(
        spread_ser.mean() / (spread_ser.std(ddof=1) + 1e-12) * math.sqrt(ann_days)
    )
    adj_step_rate = float(
        np.mean(np.diff(mean_ann.to_numpy(dtype=np.float64)) > 0) * 100.0
    )

    stats = {
        "Mono_Spearman": float(rho_mono) if np.isfinite(rho_mono) else 0.0,
        "Adj_Step_Rate(%)": adj_step_rate,
        "D10_Minus_D1_Ann(%)": spread_ann,
        "D10_Minus_D1_Sharpe": spread_sh,
    }
    return summary_df, stats


def audit_decile_monotonicity_all(
    bundle: V7DataBundle,
    tab_dir: Path | None = None,
    ann_days: int = 252,
) -> pd.DataFrame:
    """Run decile monotonicity diagnostics across all prediction experiments."""
    rows = []
    for exp in EXPERIMENTS_V7:
        dec_df, st = compute_decile_monotonicity(
            bundle.scores[exp], bundle.ret1d_df, ann_days=ann_days
        )
        if tab_dir is not None:
            dec_df.to_csv(tab_dir / f"sec3_decile_returns_{exp}.csv", index=False)
        rows.append({"Experiment": exp, **{k: round(v, 4) for k, v in st.items()}})
    out = pd.DataFrame(rows)
    if tab_dir is not None:
        out.to_csv(tab_dir / "sec3_decile_monotonicity.csv", index=False)
    return out
