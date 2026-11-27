"""Section 3 24-candidate portfolio construction and grand_summary_v7 parity audit."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from v27.backtest.metrics import compounded_dd_episodes, compute_legacy_brain_stats
from v27.backtest.simulator import backtest_weights_daily
from v27.config import EXPERIMENTS_V7, STRATEGIES_V7, PortfolioBacktestConfig
from v27.data.bundle import V7DataBundle
from v27.portfolio.constructors import build_8_strategies_for_score


def run_24_candidate_parity_audit(
    bundle: V7DataBundle,
    qh_dir: Path,
    tab_dir: Path | None = None,
    bt_cfg: PortfolioBacktestConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, pd.Series]], pd.DataFrame]:
    """Reconstruct all 24 base candidates and verify legacy parity against grand_summary_v7.csv."""
    bt_cfg = bt_cfg or PortfolioBacktestConfig()
    r1_np = bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64)
    adv_np = bundle.adv20_df.to_numpy(dtype=np.float64)
    sig_np = bundle.sigma_prev_df.to_numpy(dtype=np.float64)
    rf_np = bundle.ff6_df["rf"].to_numpy(dtype=np.float64)

    gs_path = qh_dir / "grand_summary_v7.csv"
    gs_map: dict[tuple[str, str], dict[str, float]] = {}
    if gs_path.exists():
        gs_pl = pl.read_csv(gs_path)
        for row in gs_pl.iter_rows(named=True):
            gs_map[(str(row["experiment"]), str(row["portfolio"]))] = row

    cand_weights: dict[str, np.ndarray] = {}
    cand_bts: dict[str, dict[str, pd.Series]] = {}
    audit_rows = []

    for exp in EXPERIMENTS_V7:
        w_dict = build_8_strategies_for_score(
            bundle.scores[exp],
            bundle.ret1d_df,
            bundle.sigma_prev_df,
            bundle.beta60_df,
            hold_days=bt_cfg.hold_days,
        )
        for sname in STRATEGIES_V7:
            cid = f"{exp}__{sname}"
            W_np = w_dict[sname].to_numpy(dtype=np.float64)
            cand_weights[cid] = W_np
            bt = backtest_weights_daily(
                W_np,
                r1_np,
                bundle.dates,
                cost_bps=bt_cfg.baseline_cost_bps,
                adv20_np=adv_np,
                sigma_np=sig_np,
                rf_np=rf_np,
                book_size=bt_cfg.book_size,
                short_borrow_ann=bt_cfg.short_borrow_ann,
                impact_eta=bt_cfg.impact_coeff,
                ann_days=bt_cfg.ann_days,
                inception_zero_trade=False,
            )
            cand_bts[cid] = bt
            leg = compute_legacy_brain_stats(
                bt, cost_bps=bt_cfg.baseline_cost_bps, ann_days=bt_cfg.ann_days
            )
            rn = bt["net"]
            nav = (1.0 + rn.clip(lower=-0.95)).cumprod()
            cagr = float((nav.iloc[-1] ** (bt_cfg.ann_days / len(rn)) - 1.0) * 100.0)
            dd_i = compounded_dd_episodes(rn)
            gs_row = gs_map.get((exp, sname))

            audit_rows.append(
                {
                    "Candidate": cid,
                    "Experiment": exp,
                    "Strategy": sname,
                    "Legacy_GrossSh": leg["Legacy_GrossSh"],
                    "GS_v7_GrossSh": float(gs_row["Sharpe"])
                    if gs_row is not None
                    else np.nan,
                    "Legacy_NetSh2": leg[
                        f"Legacy_NetSh@{bt_cfg.baseline_cost_bps:g}bps"
                    ],
                    "GS_v7_NetSh2": float(gs_row["NetSharpe@2bps"])
                    if gs_row is not None
                    else np.nan,
                    "Legacy_Turn(%)": leg["Legacy_Turn(%)"],
                    "GS_v7_Turn(%)": float(gs_row["Turnover(%)"])
                    if gs_row is not None
                    else np.nan,
                    "Legacy_ArithDD(%)": leg["Legacy_ArithDD(%)"],
                    "Comp_NetSh2": float(
                        rn.mean()
                        / (rn.std(ddof=1) + 1e-12)
                        * math.sqrt(bt_cfg.ann_days)
                    ),
                    "Comp_CAGR2(%)": cagr,
                    "Comp_MaxDD(%)": dd_i["Comp_MaxDD(%)"],
                    "Comp_Calmar2": cagr / max(dd_i["Comp_MaxDD(%)"], 0.01),
                }
            )

    if tab_dir is not None:
        pl.DataFrame(audit_rows).write_csv(
            tab_dir / "sec3_audit_24_candidates_parity.csv"
        )
    return cand_weights, cand_bts, pd.DataFrame(audit_rows).set_index("Candidate")
