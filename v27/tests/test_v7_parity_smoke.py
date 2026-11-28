"""Integration smoke test verifying exact numeric parity against outputs/QH_v7 baselines."""

from __future__ import annotations

from pathlib import Path

import pytest

from v27.config import PathConfig
from v27.data.bundle import load_v7_data_bundle
from v27.experiments.parity_audit import run_24_candidate_parity_audit

QH_DIR = PathConfig().qh_v7_dir


@pytest.mark.skipif(
    not (QH_DIR / "grand_summary_v7.csv").exists(),
    reason="outputs/QH_v7 artifacts not present",
)
def test_v7_24_candidates_parity_matches_semester_261():
    bundle = load_v7_data_bundle(
        qh_dir=QH_DIR, warehouse_path=Path("nonexistent_skip_duckdb")
    )
    _, _, audit_df = run_24_candidate_parity_audit(bundle, QH_DIR)
    max_delta = float(
        (audit_df["Legacy_NetSh2"] - audit_df["GS_v7_NetSh2"]).abs().max()
    )
    assert max_delta <= 0.004, f"Parity drift {max_delta:.4f} > 0.004"
    best_ind = audit_df.loc["risk_t5__top10_BetaNeu_InvVol_Buf"]
    assert (
        abs(float(best_ind["Legacy_NetSh2"]) - float(best_ind["GS_v7_NetSh2"])) <= 0.004
    )
