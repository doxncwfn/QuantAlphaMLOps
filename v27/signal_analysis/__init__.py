"""Signal strength analysis modules for v27."""

from __future__ import annotations

from v27.signal_analysis.alpha_decay import compute_alpha_decay
from v27.signal_analysis.decile_monotonicity import (
    audit_decile_monotonicity_all,
    compute_decile_monotonicity,
)
from v27.signal_analysis.horizon_audit import (
    audit_risk_t5_horizon_accounting,
    audit_signal_1d_vs_5d,
    non_overlapping_rebal_weights,
)
from v27.signal_analysis.score_persistence import (
    compute_dynamic_persistence_precheck,
    compute_score_rank_correlations,
)

__all__ = [
    "audit_decile_monotonicity_all",
    "audit_risk_t5_horizon_accounting",
    "audit_signal_1d_vs_5d",
    "compute_alpha_decay",
    "compute_decile_monotonicity",
    "compute_dynamic_persistence_precheck",
    "compute_score_rank_correlations",
    "non_overlapping_rebal_weights",
]
