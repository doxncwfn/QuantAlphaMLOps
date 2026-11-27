"""Cross-model score rank correlation and dynamic Sharpe persistence pre-check."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from v27.config import EXPERIMENTS_V7
from v27.data.bundle import V7DataBundle


def compute_score_rank_correlations(
    bundle: V7DataBundle, tab_dir: Path | None = None
) -> pd.DataFrame:
    """Compute average daily cross-sectional Spearman rank correlation across experiment scores."""
    sc_corrs = pd.DataFrame(
        index=EXPERIMENTS_V7, columns=EXPERIMENTS_V7, dtype=np.float64
    )
    for e1 in EXPERIMENTS_V7:
        for e2 in EXPERIMENTS_V7:
            c_ser = (
                bundle.scores[e1]
                .corrwith(bundle.scores[e2], axis=1, method="spearman")
                .dropna()
            )
            sc_corrs.loc[e1, e2] = float(c_ser.mean())
    if tab_dir is not None:
        sc_corrs.to_csv(tab_dir / "sec4_model_score_rank_correlations.csv")
    return sc_corrs


def compute_dynamic_persistence_precheck(
    ret_net_df: pd.DataFrame,
    tab_dir: Path | None = None,
    purge_days: int = 6,
    ann_days: int = 252,
) -> pd.DataFrame:
    """Test whether past strategy Sharpe predicts future Sharpe (persistence vs mean reversion)."""
    rows = []
    R = ret_net_df.to_numpy(dtype=np.float64)
    T, _ = R.shape
    for lb in (21, 63, 126, 252):
        for fwd in (21, 63):
            cs_corrs = []
            top_f_sh, bot_f_sh = [], []
            for t in range(lb + purge_days, T - fwd, fwd):
                past = R[t - lb - purge_days : t - purge_days]
                fut = R[t : t + fwd]
                p_sh = (
                    np.mean(past, axis=0)
                    / (np.std(past, axis=0, ddof=1) + 1e-12)
                    * math.sqrt(ann_days)
                )
                f_sh = (
                    np.mean(fut, axis=0)
                    / (np.std(fut, axis=0, ddof=1) + 1e-12)
                    * math.sqrt(ann_days)
                )
                if np.std(p_sh) > 1e-9 and np.std(f_sh) > 1e-9:
                    cs_corrs.append(float(np.corrcoef(p_sh, f_sh)[0, 1]))
                rk = np.argsort(p_sh)
                top_f_sh.append(float(np.mean(f_sh[rk[-6:]])))
                bot_f_sh.append(float(np.mean(f_sh[rk[:6]])))
            arr = np.asarray(cs_corrs, dtype=np.float64)
            rows.append(
                {
                    "Lookback(d)": lb,
                    "Forward(d)": fwd,
                    "N_Periods": len(arr),
                    "Mean_RankIC": round(float(arr.mean()), 4),
                    "IC_tstat": round(
                        float(
                            arr.mean()
                            / (arr.std(ddof=1) / math.sqrt(max(len(arr), 1)) + 1e-12)
                        ),
                        2,
                    ),
                    "Pos_IC_Frac(%)": round(float((arr > 0).mean() * 100.0), 1),
                    "TopQuartile_FwdSh": round(float(np.mean(top_f_sh)), 3),
                    "BotQuartile_FwdSh": round(float(np.mean(bot_f_sh)), 3),
                    "Spread_TopMinusBot": round(
                        float(np.mean(top_f_sh) - np.mean(bot_f_sh)), 3
                    ),
                    "Verdict": "Persistence"
                    if np.mean(arr) > 0.03
                    else ("MeanReversion" if np.mean(arr) < -0.03 else "Neutral"),
                }
            )
    df = pd.DataFrame(rows)
    if tab_dir is not None:
        df.to_csv(tab_dir / "tab05_dynamic_persistence_precheck.csv", index=False)
    return df
