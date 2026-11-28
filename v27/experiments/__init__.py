"""Strategy experiment runners (24-candidate parity, Methods A/B/E/F, Section 10 ablations)."""

from __future__ import annotations

from v27.experiments.ablations import run_section10_ablations
from v27.experiments.common import run_consolidated_strategy
from v27.experiments.method_a_static import build_method_a_static
from v27.experiments.method_b_score_blend import build_method_b_score_ensembles
from v27.experiments.method_e_dynamic import build_method_e_dynamic
from v27.experiments.method_f_hierarchical import build_method_f_hierarchical
from v27.experiments.parity_audit import run_24_candidate_parity_audit

__all__ = [
    "build_method_a_static",
    "build_method_b_score_ensembles",
    "build_method_e_dynamic",
    "build_method_f_hierarchical",
    "run_24_candidate_parity_audit",
    "run_consolidated_strategy",
    "run_section10_ablations",
]
