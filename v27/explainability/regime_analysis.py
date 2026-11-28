"""Causal regime classification and regime-conditioned Sharpe performance matrices."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from v27.config import CANDIDATES_24
from v27.data.bundle import V7DataBundle


def classify_market_regimes(bundle: V7DataBundle) -> dict[str, np.ndarray]:
    """Construct causal macro/breadth/volatility/momentum flags plus named crisis/rally episodes."""
    mkt = bundle.factors_3f["MKT"]
    mkt_nav = (1.0 + mkt).cumprod()
    is_bear = (
        mkt_nav.shift(1).bfill()
        < mkt_nav.rolling(200, min_periods=40).mean().shift(1).bfill()
    ) & (
        ((1.0 + mkt).rolling(63, min_periods=20).apply(np.prod, raw=True) - 1.0)
        .shift(1)
        .fillna(0.0)
        < 0.0
    )
    vol20_l = mkt.rolling(20, min_periods=5).std().shift(1).bfill()
    br_l = (
        (bundle.ret1d_df > 0)
        .mean(axis=1)
        .rolling(20, min_periods=5)
        .mean()
        .shift(1)
        .bfill()
    )
    mom_l = (
        bundle.factors_3f["MOM"].rolling(63, min_periods=20).sum().shift(1).fillna(0.0)
    )
    idx_d = bundle.dates
    return {
        "Bull": np.asarray(~is_bear, dtype=bool),
        "Bear": np.asarray(is_bear, dtype=bool),
        "Vol_Low": np.asarray(vol20_l <= vol20_l.quantile(1 / 3), dtype=bool),
        "Vol_High": np.asarray(vol20_l >= vol20_l.quantile(2 / 3), dtype=bool),
        "Breadth_Low": np.asarray(br_l <= br_l.quantile(1 / 3), dtype=bool),
        "Breadth_High": np.asarray(br_l >= br_l.quantile(2 / 3), dtype=bool),
        "Mom_Pos": np.asarray(mom_l > 0.0, dtype=bool),
        "Mom_Neg": np.asarray(mom_l <= 0.0, dtype=bool),
        "Euro_2011": np.asarray(
            (idx_d >= "2011-05-01") & (idx_d <= "2011-12-31"), dtype=bool
        ),
        "ChinaOil_15_16": np.asarray(
            (idx_d >= "2015-08-01") & (idx_d <= "2016-03-31"), dtype=bool
        ),
        "COVID_2020": np.asarray(
            (idx_d >= "2020-02-15") & (idx_d <= "2020-04-30"), dtype=bool
        ),
        "RateHike_2022": np.asarray(
            (idx_d >= "2022-01-01") & (idx_d <= "2022-10-31"), dtype=bool
        ),
        "AI_Bull_23_24": np.asarray(
            (idx_d >= "2023-01-01") & (idx_d <= "2024-12-31"), dtype=bool
        ),
    }


def compute_regime_sharpes(
    net_df: pd.DataFrame,
    regimes: dict[str, np.ndarray],
    tab_dir: Path | None = None,
    ann_days: int = 252,
) -> pd.DataFrame:
    """Evaluate annualized Net Sharpe ratio across every regime mask per candidate strategy."""
    cids = list(CANDIDATES_24)
    reg_sh = pd.DataFrame(index=cids, columns=list(regimes.keys()), dtype=np.float64)
    for rname, mask in regimes.items():
        sub = net_df.iloc[mask]
        for cid in cids:
            s = sub[cid]
            reg_sh.loc[cid, rname] = float(
                s.mean() / (s.std(ddof=1) + 1e-12) * math.sqrt(ann_days)
            )
    if tab_dir is not None:
        reg_sh.to_csv(tab_dir / "sec4_regime_sharpes_24.csv")
    return reg_sh
