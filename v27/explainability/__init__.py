"""Alpha explainability components for v27."""

from __future__ import annotations

from v27.explainability.complementarity import (
    build_pairwise_complementarity,
    compute_position_overlap_and_tail,
    compute_return_correlations,
)
from v27.explainability.factor_exposure import (
    compute_3f_attribution,
    compute_ff6_and_sector_exposure,
    newey_west_ols,
)
from v27.explainability.gate_attribution import (
    cluster_share_table,
    summarize_master_gate_clusters,
)
from v27.explainability.regime_analysis import (
    classify_market_regimes,
    compute_regime_sharpes,
)
from v27.explainability.risk_contribution import compute_risk_contribution_and_exposures

__all__ = [
    "build_pairwise_complementarity",
    "classify_market_regimes",
    "cluster_share_table",
    "compute_3f_attribution",
    "compute_ff6_and_sector_exposure",
    "compute_position_overlap_and_tail",
    "compute_regime_sharpes",
    "compute_return_correlations",
    "compute_risk_contribution_and_exposures",
    "newey_west_ols",
    "summarize_master_gate_clusters",
]
