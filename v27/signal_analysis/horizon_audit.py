"""Section 3 signal horizon audit: 1d vs 5d IC, Top/Bot hit rates, and risk_t5 accounting."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from v27.backtest.metrics import compounded_dd_episodes
from v27.backtest.simulator import backtest_weights_daily
from v27.config import EXPERIMENTS_V7, PortfolioBacktestConfig
from v27.data.bundle import V7DataBundle
from v27.portfolio.constructors import apply_overlapping_hold, strat_top10_buf
from v27.portfolio.neutralization import neutralize_beta, renorm_legs


def non_overlapping_rebal_weights(
    w_daily_df: pd.DataFrame, r_1d_df: pd.DataFrame, stride: int = 5
) -> pd.DataFrame:
    """Rebalance strictly every `stride` days and drift portfolio weights daily on 1d returns."""
    W_tar = np.nan_to_num(w_daily_df.to_numpy(dtype=np.float64), nan=0.0)
    R = np.nan_to_num(r_1d_df.to_numpy(dtype=np.float64), nan=0.0)
    T, N = W_tar.shape
    W_out = np.zeros((T, N), dtype=np.float64)
    for t in range(T):
        if t % stride == 0:
            W_out[t] = W_tar[t]
        else:
            gp = float(np.dot(W_out[t - 1], R[t - 1]))
            W_out[t] = W_out[t - 1] * (1.0 + R[t - 1]) / max(1.0 + gp, 0.25)
    return pd.DataFrame(W_out, index=w_daily_df.index, columns=w_daily_df.columns)


def audit_signal_1d_vs_5d(
    bundle: V7DataBundle, tab_dir: Path | None = None, ann_days: int = 252
) -> pd.DataFrame:
    """Measure 1-day vs 5-day predictive RankIC, ICIR, and Top-Bottom 10% hit rate per experiment."""
    rows = []
    for exp in EXPERIMENTS_V7:
        sc = bundle.scores[exp]
        act = sc.notna()
        ic_1d = (
            sc.rank(axis=1)
            .corrwith(bundle.ret1d_df.where(act).rank(axis=1), axis=1)
            .dropna()
        )
        ic_5d = (
            sc.rank(axis=1)
            .corrwith(bundle.ret5d_fwd_df.where(act).rank(axis=1), axis=1)
            .dropna()
        )
        p = sc.rank(axis=1, pct=True)
        w_t10 = renorm_legs(
            ((p > 0.90).astype(float) - (p <= 0.10).astype(float))
            .where(act)
            .fillna(0.0)
        )
        spread_1d = (w_t10 * bundle.ret1d_df.fillna(0.0)).sum(axis=1)
        spread_5d = (w_t10 * bundle.ret5d_fwd_df.fillna(0.0)).sum(axis=1).iloc[:-4]
        rows.append(
            {
                "Experiment": exp,
                "IC_vs_1d": float(ic_1d.mean()),
                "ICIR_1d": float(
                    ic_1d.mean() / (ic_1d.std() + 1e-12) * math.sqrt(ann_days)
                ),
                "TopBot10_Hit_1d(%)": float((spread_1d > 0).mean() * 100.0),
                "IC_vs_5d": float(ic_5d.mean()),
                "ICIR_5d_OverlapAdj": float(
                    ic_5d.mean() / (ic_5d.std() + 1e-12) * math.sqrt(ann_days / 5.0)
                ),
                "TopBot10_Hit_5d(%)": float((spread_5d > 0).mean() * 100.0),
            }
        )
    df = pd.DataFrame(rows)
    if tab_dir is not None:
        df.to_csv(tab_dir / "sec3_signal_1d_vs_5d_audit.csv", index=False)
    return df


def audit_risk_t5_horizon_accounting(
    bundle: V7DataBundle,
    tab_dir: Path | None = None,
    cfg: PortfolioBacktestConfig | None = None,
) -> pd.DataFrame:
    """Compare valid daily/non-overlap/HOLD5 MTM accounting against invalid 5d-label MTM."""
    cfg = cfg or PortfolioBacktestConfig()
    sc5 = bundle.scores["risk_t5"]
    inv_vol = (1.0 / bundle.sigma_prev_df).where(sc5.notna()).fillna(0.0)
    w_t5_daily = strat_top10_buf(neutralize_beta(sc5, bundle.beta60_df), inv_vol)
    w_t5_5d_nonov = non_overlapping_rebal_weights(w_t5_daily, bundle.ret1d_df, stride=5)
    w_t5_hold5 = apply_overlapping_hold(
        w_t5_daily, bundle.ret1d_df, hold_days=cfg.hold_days
    )
    r1_np = bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64)

    rows = []
    for tag, W_m, ret_mat, lbl in [
        ("DailyRebal_Realized1d", w_t5_daily.to_numpy(), r1_np, "Valid daily 1d MTM"),
        (
            "NonOverlap5d_Realized1d",
            w_t5_5d_nonov.to_numpy(),
            r1_np,
            "Valid 5d-stride MTM",
        ),
        (
            "OverlapHOLD5_Realized1d",
            w_t5_hold5.to_numpy(),
            r1_np,
            "Valid 5-tranche MTM",
        ),
        (
            "INVALID_5dFwdLabel_MTM",
            w_t5_daily.to_numpy(),
            np.nan_to_num(bundle.ret5d_fwd_df.to_numpy(), nan=0.0),
            "DISTORTED (5d fwd as 1d)",
        ),
    ]:
        bt_m = backtest_weights_daily(
            W_m, ret_mat, bundle.dates, cost_bps=cfg.baseline_cost_bps
        )
        rn = bt_m["net"]
        nav = (1.0 + rn.clip(lower=-0.95)).cumprod()
        rows.append(
            {
                "Mode": tag,
                "Note": lbl,
                "GrossSh": float(
                    bt_m["gross"].mean()
                    / (bt_m["gross"].std(ddof=1) + 1e-12)
                    * math.sqrt(cfg.ann_days)
                ),
                "NetSh@2bps": float(
                    rn.mean() / (rn.std(ddof=1) + 1e-12) * math.sqrt(cfg.ann_days)
                ),
                "CAGR2(%)": float(
                    (nav.iloc[-1] ** (1.0 / (len(rn) / cfg.ann_days)) - 1.0) * 100.0
                ),
                "Turn_1Way(%)": float(0.5 * bt_m["traded"].mean() * 100.0),
                "CompMaxDD(%)": compounded_dd_episodes(rn)["Comp_MaxDD(%)"],
            }
        )
    df = pd.DataFrame(rows)
    if tab_dir is not None:
        df.to_csv(tab_dir / "sec3_risk_t5_horizon_accounting_audit.csv", index=False)
    return df
