"""Component risk contribution, tail risk, and realized holdings exposure diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from v27.backtest.metrics import compounded_dd_episodes
from v27.config import CANDIDATES_24
from v27.data.bundle import V7DataBundle


def compute_risk_contribution_and_exposures(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    cand_bts: dict[str, dict[str, pd.Series]],
    net_df: pd.DataFrame,
    tail_dep_df: pd.DataFrame,
    tab_dir: Path | None = None,
    ann_days: int = 252,
) -> pd.DataFrame:
    """Compute MCR, CCR, EW-24 risk share, CVaR95, drawdown episodes, and lagged holdings tilts."""
    cids = list(CANDIDATES_24)
    n_c = len(cids)
    cov_ann = net_df.cov().to_numpy(dtype=np.float64) * ann_days
    w_eq24 = np.ones(n_c) / n_c
    p_vol24 = math.sqrt(float(w_eq24 @ cov_ann @ w_eq24))
    mcr24 = (cov_ann @ w_eq24) / max(p_vol24, 1e-12)
    rc_pct24 = (w_eq24 * mcr24) / max(p_vol24, 1e-12) * 100.0

    B60_np = np.nan_to_num(bundle.beta60_df.to_numpy(dtype=np.float64), nan=1.0)
    S60_np = np.nan_to_num(bundle.sigma_prev_df.to_numpy(dtype=np.float64), nan=0.02)
    M252_np = np.nan_to_num(bundle.mom252_df.to_numpy(dtype=np.float64), nan=0.0)
    sec_masks = [bundle.sectors == s for s in np.unique(bundle.sectors)]
    tail_dep = tail_dep_df.to_numpy(dtype=np.float64)

    rows = []
    for i, cid in enumerate(cids):
        r = net_df[cid]
        bt = cand_bts[cid]
        W_c = cand_weights[cid]
        dd_i = compounded_dd_episodes(r)
        ann_ret = float(r.mean() * ann_days * 100.0)
        ann_vol = float(r.std(ddof=1) * math.sqrt(ann_days) * 100.0)
        down = r[r < 0]
        down_dev = (
            math.sqrt(float((down**2).mean()) * ann_days) * 100.0 if len(down) else 0.0
        )
        q05 = float(r.quantile(0.05))
        cvar95 = float(r[r <= q05].mean() * 100.0)
        port_b60 = float(np.mean(np.einsum("ti,ti->t", W_c, B60_np)))
        port_vol_exp = float(np.mean(np.einsum("ti,ti->t", W_c, S60_np)) * 100.0)
        port_mom_exp = float(np.mean(np.einsum("ti,ti->t", W_c, M252_np)) * 100.0)
        max_sec_net = float(
            max(np.abs(W_c[:, sm].sum(axis=1)).mean() for sm in sec_masks) * 100.0
        )
        rows.append(
            {
                "Candidate": cid,
                "AnnRet2(%)": ann_ret,
                "AnnVol2(%)": ann_vol,
                "NetSh2": float(
                    r.mean() / (r.std(ddof=1) + 1e-12) * math.sqrt(ann_days)
                ),
                "DownDev(%)": down_dev,
                "MCR_EW24(%)": mcr24[i] * 100.0,
                "RiskShare_EW24(%)": rc_pct24[i],
                "Comp_MaxDD(%)": dd_i["Comp_MaxDD(%)"],
                "MaxDD_Dur(d)": dd_i["MaxDD_Dur(d)"],
                "MaxRecov(d)": dd_i["MaxRecov(d)"],
                "CVaR95_1d(%)": cvar95,
                "Avg_TailDep": float((tail_dep[i].sum() - 1.0) / (n_c - 1)),
                "AvgGross": float(bt["gross_exp"].mean()),
                "AvgNet": float(bt["net_exp"].mean()),
                "Hold_Beta60": port_b60,
                "Hold_VolExp(%)": port_vol_exp,
                "Hold_MomExp(%)": port_mom_exp,
                "MaxAvgSecNet(%)": max_sec_net,
                "Turn_1Way(%)": float(0.5 * bt["traded"].mean() * 100.0),
            }
        )
    df_rc = pd.DataFrame(rows).set_index("Candidate")
    if tab_dir is not None:
        df_rc.to_csv(tab_dir / "sec4_risk_contribution_and_exposures_24.csv")
    return df_rc
