"""Alpha decay analysis across forward lag days (t+1..t+5) and overlapping hold tranches."""

from __future__ import annotations

import pandas as pd

from v27.backtest.metrics import compute_legacy_brain_stats
from v27.backtest.simulator import backtest_weights_daily
from v27.portfolio.constructors import apply_overlapping_hold, strat_top10_invvol


def compute_alpha_decay(
    scores_df: pd.DataFrame,
    ret1d_df: pd.DataFrame,
    sigma_prev_df: pd.DataFrame,
    cost_bps: float = 2.0,
    max_hold: int = 5,
    ann_days: int = 252,
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Compute lagged daily Rank IC (t+1..t+H) and overlapping HOLD=1..H Sharpe vs Turnover."""
    act_mask = scores_df.notna()
    inv_vol = (1.0 / sigma_prev_df).where(act_mask).fillna(0.0)
    w_base = strat_top10_invvol(scores_df, inv_vol)
    r_np = ret1d_df.fillna(0.0).to_numpy(dtype=float)

    ic_by_lag: dict[int, float] = {}
    rows: dict[int, dict[str, float]] = {}
    for h in range(1, max_hold + 1):
        r_lag = ret1d_df.shift(-(h - 1)).where(act_mask)
        ic_by_lag[h] = float(
            scores_df.rank(axis=1).corrwith(r_lag.rank(axis=1), axis=1).dropna().mean()
        )
        w_h = (
            w_base if h == 1 else apply_overlapping_hold(w_base, ret1d_df, hold_days=h)
        )
        bt = backtest_weights_daily(
            w_h.to_numpy(dtype=float),
            r_np,
            ret1d_df.index,
            cost_bps=cost_bps,
            inception_zero_trade=False,
        )
        leg = compute_legacy_brain_stats(bt, cost_bps=cost_bps, ann_days=ann_days)
        rows[h] = {
            "Lag_RankIC": ic_by_lag[h],
            "GrossSharpe": leg["Legacy_GrossSh"],
            f"NetSharpe@{cost_bps:g}bps": leg[f"Legacy_NetSh@{cost_bps:g}bps"],
            "Turnover(%)": leg["Legacy_Turn(%)"],
            "Drawdown(%)": leg["Legacy_ArithDD(%)"],
        }
    return pd.DataFrame(rows).T, ic_by_lag
