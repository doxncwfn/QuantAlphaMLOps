"""Section 10 ablation study quantifying marginal impact of signals, constructions, and overlays."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v27.config import CANDIDATES_24, PortfolioBacktestConfig
from v27.data.bundle import V7DataBundle
from v27.experiments.common import run_consolidated_strategy
from v27.optimization.estimators import ledoit_wolf_cov
from v27.optimization.schedule import causal_wf_schedule, static_schedule
from v27.optimization.solvers import solve_erc


def run_section10_ablations(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    ret_net_df: pd.DataFrame,
    flagship_res: dict[str, Any],
    tab_dir: Path | None = None,
    bt_cfg: PortfolioBacktestConfig | None = None,
) -> pd.DataFrame:
    """Measure Sharpe/CAGR/MaxDD deltas when dropping individual features, horizons, or overlays."""
    bt_cfg = bt_cfg or PortfolioBacktestConfig()
    dates = bundle.dates
    wu, pg, ann = bt_cfg.warmup_days, bt_cfg.purge_days, bt_cfg.ann_days

    def _oos_row(
        label: str, res: dict[str, Any], b_sh: float, b_cagr: float, b_mdd: float
    ) -> dict[str, Any]:
        r = res["bt"]["net"].iloc[wu:]
        nav = (1.0 + r).cumprod()
        cagr = float((nav.iloc[-1] ** (ann / len(r)) - 1.0) * 100.0)
        mdd = float(
            -(nav / np.maximum.accumulate(np.maximum(nav, 1.0)) - 1.0).min() * 100.0
        )
        sh = float(r.mean() / (r.std(ddof=1) + 1e-12) * math.sqrt(ann))
        tvr = float(res["bt"]["traded"].iloc[wu:].mean() * 100.0)
        return {
            "Ablation": label,
            "OOS_Net2_Sharpe": round(sh, 4),
            "Delta_Sharpe": round(sh - b_sh, 4),
            "OOS_CAGR(%)": round(cagr, 2),
            "Delta_CAGR(%)": round(cagr - b_cagr, 2),
            "OOS_MaxDD(%)": round(mdd, 2),
            "Delta_MaxDD(%)": round(mdd - b_mdd, 2),
            "OOS_Turnover(%)": round(tvr, 2),
        }

    ref_sched = causal_wf_schedule(
        ret_net_df[list(CANDIDATES_24)],
        dates,
        lambda X, wp, t: solve_erc(ledoit_wolf_cov(X), w_max=0.22),
        warmup=wu,
        purge=pg,
    )
    ref_res = run_consolidated_strategy(
        "Abl_Ref_ERC_All24", "Ablation", ref_sched, cand_weights, bundle, bt_cfg=bt_cfg
    )
    r_ref = ref_res["bt"]["net"].iloc[wu:]
    nav_ref = (1.0 + r_ref).cumprod()
    b_sh = float(r_ref.mean() / (r_ref.std(ddof=1) + 1e-12) * math.sqrt(ann))
    b_cagr = float((nav_ref.iloc[-1] ** (ann / len(r_ref)) - 1.0) * 100.0)
    b_mdd = float(
        -(nav_ref / np.maximum.accumulate(np.maximum(nav_ref, 1.0)) - 1.0).min() * 100.0
    )

    abl_specs = {
        "Baseline_ERC_All24": list(CANDIDATES_24),
        "Drop_BetaNeu": [c for c in CANDIDATES_24 if "BetaNeu" not in c],
        "Drop_InvVol": [c for c in CANDIDATES_24 if "InvVol" not in c],
        "Drop_Buffer": [c for c in CANDIDATES_24 if "Buf" not in c],
        "Drop_HOLD5": [c for c in CANDIDATES_24 if "HOLD5" not in c],
        "Drop_risk_t1": [c for c in CANDIDATES_24 if not c.startswith("risk_t1__")],
        "Drop_raw_t1": [c for c in CANDIDATES_24 if not c.startswith("raw_t1__")],
        "Drop_risk_t5": [c for c in CANDIDATES_24 if not c.startswith("risk_t5__")],
    }
    rows = []
    for label, cols in abl_specs.items():
        if label == "Baseline_ERC_All24":
            rows.append(_oos_row(label, ref_res, b_sh, b_cagr, b_mdd))
            continue
        sch = causal_wf_schedule(
            ret_net_df[cols],
            dates,
            lambda X, wp, t: solve_erc(ledoit_wolf_cov(X), w_max=0.25),
            warmup=wu,
            purge=pg,
        )
        res = run_consolidated_strategy(
            f"Abl_{label}", "Ablation", sch, cand_weights, bundle, bt_cfg=bt_cfg
        )
        rows.append(_oos_row(label, res, b_sh, b_cagr, b_mdd))

    ew_res = run_consolidated_strategy(
        "Abl_EW_All24",
        "Ablation",
        static_schedule(dates, {c: 1.0 / len(CANDIDATES_24) for c in CANDIDATES_24}),
        cand_weights,
        bundle,
        bt_cfg=bt_cfg,
    )
    rows.append(_oos_row("Replace_CovOpt_With_EW", ew_res, b_sh, b_cagr, b_mdd))
    rows.append(_oos_row("Final_Flagship_Full", flagship_res, b_sh, b_cagr, b_mdd))

    df = pd.DataFrame(rows)
    if tab_dir is not None:
        df.to_csv(tab_dir / "tab10_ablation_study.csv", index=False)
    return df
