"""Method D: Causal expanding risk-based and return-risk portfolio optimization."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from v27.config import CANDIDATES_1D_16, CANDIDATES_24
from v27.data.bundle import V7DataBundle
from v27.experiments.common import run_consolidated_strategy
from v27.optimization.estimators import bayes_stein_mu, cov_corr_shrink, ledoit_wolf_cov
from v27.optimization.schedule import causal_wf_schedule
from v27.optimization.solvers import (
    solve_erc,
    solve_inv_vol,
    solve_max_sharpe,
    solve_mean_cvar_lp,
    solve_min_var,
    solve_regularized_return_risk,
)


def build_method_d_risk_and_opt(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    ret_net_df: pd.DataFrame,
    ortho_cols: list[str],
    warmup: int = 252,
    purge: int = 6,
) -> list[dict[str, Any]]:
    """Run Method D expanding causal optimizers across 1d16, All24, and Ortho-8 pools."""
    dates = bundle.dates
    out: list[dict[str, Any]] = []
    pools = {
        "1d16": list(CANDIDATES_1D_16),
        "All24": list(CANDIDATES_24),
        "Ortho": ortho_cols,
    }
    w_caps = {"1d16": 0.25, "All24": 0.22, "Ortho": 0.35}

    for pname, cols in pools.items():
        R_df = ret_net_df[cols]
        w_max = w_caps[pname]

        sched_iv = causal_wf_schedule(
            R_df,
            dates,
            lambda X, wp, t, m=w_max: solve_inv_vol(X, w_max=m),
            warmup=warmup,
            purge=purge,
        )
        out.append(
            run_consolidated_strategy(
                f"D1_ExpInvVol_{pname}",
                "Method_D_RiskOpt",
                sched_iv,
                cand_weights,
                bundle,
            )
        )

        sched_erc = causal_wf_schedule(
            R_df,
            dates,
            lambda X, wp, t, m=w_max: solve_erc(ledoit_wolf_cov(X), w_max=m),
            warmup=warmup,
            purge=purge,
        )
        out.append(
            run_consolidated_strategy(
                f"D2_ExpERC_LW_{pname}",
                "Method_D_RiskOpt",
                sched_erc,
                cand_weights,
                bundle,
            )
        )

        if pname in ("1d16", "Ortho"):
            sched_mv = causal_wf_schedule(
                R_df,
                dates,
                lambda X, wp, t, m=w_max: solve_min_var(
                    ledoit_wolf_cov(X), w_max=m, l2_reg=0.10
                ),
                warmup=warmup,
                purge=purge,
            )
            out.append(
                run_consolidated_strategy(
                    f"D3_ExpMinVar_LW_{pname}",
                    "Method_D_RiskOpt",
                    sched_mv,
                    cand_weights,
                    bundle,
                )
            )

            sched_cs = causal_wf_schedule(
                R_df,
                dates,
                lambda X, wp, t, m=w_max: solve_min_var(
                    cov_corr_shrink(X, 0.30), w_max=m, l2_reg=0.10
                ),
                warmup=warmup,
                purge=purge,
            )
            out.append(
                run_consolidated_strategy(
                    f"D4_ExpCovCorrShrink_{pname}",
                    "Method_D_RiskOpt",
                    sched_cs,
                    cand_weights,
                    bundle,
                )
            )

            sched_sh = causal_wf_schedule(
                R_df,
                dates,
                lambda X, wp, t, m=w_max: solve_max_sharpe(
                    bayes_stein_mu(X, ledoit_wolf_cov(X)), ledoit_wolf_cov(X), w_max=m
                ),
                warmup=warmup,
                purge=purge,
            )
            out.append(
                run_consolidated_strategy(
                    f"D5_ExpMaxSharpe_BS_{pname}",
                    "Method_D_ReturnRisk",
                    sched_sh,
                    cand_weights,
                    bundle,
                )
            )

            sched_cvar = causal_wf_schedule(
                R_df,
                dates,
                lambda X, wp, t, m=w_max: solve_mean_cvar_lp(
                    X[-504:], bayes_stein_mu(X, ledoit_wolf_cov(X)), w_max=m
                ),
                warmup=warmup,
                purge=purge,
            )
            out.append(
                run_consolidated_strategy(
                    f"D6_ExpMeanCVaR95_LP_{pname}",
                    "Method_D_ReturnRisk",
                    sched_cvar,
                    cand_weights,
                    bundle,
                )
            )

            sched_reg = causal_wf_schedule(
                R_df,
                dates,
                lambda X, wp, t, m=w_max: solve_regularized_return_risk(
                    bayes_stein_mu(X, ledoit_wolf_cov(X)),
                    ledoit_wolf_cov(X),
                    wp,
                    w_max=m,
                ),
                warmup=warmup,
                purge=purge,
            )
            out.append(
                run_consolidated_strategy(
                    f"D7_ExpRegRetRisk_{pname}",
                    "Method_D_ReturnRisk",
                    sched_reg,
                    cand_weights,
                    bundle,
                )
            )

    return out
