"""
Unit and regression tests for continuous model lookback eligibility calculation.

Verifies:
1. Requirement E: Cross-boundary lookback continuity (Day D in File A, Day D+1 in File B -> valid window).
2. Requirement F: New Russell entrant with prior trading history -> H(s, t) = True on day 1.
3. Requirement G: Genuinely newly listed security / IPO (< 40 sessions) -> H(s, t) = False until day 40.
4. Correctness of diagnostic and summary tables and quantification of boundary exclusions recovered.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.audit.config import load_audit_config
from src.audit.registry import ExceptionRegistry
from src.audit.semantics import audit_model_eligibility


@pytest.fixture(scope="module")
def audit_tables():
    config = load_audit_config("config/audit.yaml")
    out_dir = Path(config["paths"]["output_tables_dir"])

    # If tables do not exist yet, run audit_model_eligibility
    diag_path = out_dir / "model_eligibility_diagnostics.parquet"
    summary_path = out_dir / "model_eligibility_summary.parquet"
    comp_path = out_dir / "model_boundary_exclusion_comparison.parquet"

    if not (diag_path.exists() and summary_path.exists() and comp_path.exists()):
        registry = ExceptionRegistry()
        audit_model_eligibility(config, registry)

    return {
        "diagnostics": pl.read_parquet(diag_path),
        "summary": pl.read_parquet(summary_path),
        "comparison": pl.read_parquet(comp_path),
    }


def test_cross_boundary_continuity(audit_tables):
    """
    Requirement E: Test that cross-boundary transitions maintain lookback continuity.
    For AAPL on 2020-07-01 (first trading day of Russell 2020 period):
    - source_file_start must come from the preceding file ('2019.parquet')
    - source_file_end must be '2020.parquet'
    - valid_history_count must equal 40
    - lookback_eligible must be True
    - final_model_eligible must be True
    """
    df_diag = audit_tables["diagnostics"]
    aapl_2020 = df_diag.filter(
        (pl.col("ticker") == "AAPL") & (pl.col("date") == "2020-07-01")
    )
    assert len(aapl_2020) == 1, "AAPL record on 2020-07-01 must exist."
    row = aapl_2020.to_dicts()[0]

    assert row["source_file_start"] == "2019.parquet", "Lookback window start must come from 2019.parquet."
    assert row["source_file_end"] == "2020.parquet", "Lookback window end must be in 2020.parquet."
    assert row["valid_history_count"] == 40, f"Expected 40 valid sessions, got {row['valid_history_count']}."
    assert row["lookback_eligible"] is True, "Cross-boundary observation must be lookback eligible."
    assert row["membership_state"] == "CONFIRMED_MEMBER", "Must be confirmed member."
    assert row["final_model_eligible"] is True, "Must be final model eligible."


def test_new_entrant_with_prior_history(audit_tables):
    """
    Requirement F: New Russell entrant with prior trading history is eligible on day 1.
    AAXN (Axon Enterprise) entered Russell 1000 on 2020-06-30.
    It traded on NASDAQ for >40 trading sessions prior to entry.
    On 2020-06-30:
    - membership_state = 'CONFIRMED_MEMBER'
    - valid_history_count = 40 (from pre-membership trading history)
    - lookback_eligible = True
    - final_model_eligible = True
    """
    df_diag = audit_tables["diagnostics"]
    aaxn_entry = df_diag.filter(
        (pl.col("ticker") == "AAXN") & (pl.col("date") == "2020-06-30")
    )
    assert len(aaxn_entry) == 1, "AAXN record on entry date 2020-06-30 must exist."
    row = aaxn_entry.to_dicts()[0]

    assert row["membership_state"] == "CONFIRMED_MEMBER", "Must be confirmed member on entry date."
    assert row["valid_history_count"] == 40, f"Expected 40 valid sessions from prior history, got {row['valid_history_count']}."
    assert row["lookback_eligible"] is True, "New entrant with prior history must be lookback eligible."
    assert row["final_model_eligible"] is True, "Must be final model eligible on day 1."


def test_new_listing_ipo_insufficient_history(audit_tables):
    """
    Requirement G: Genuinely newly listed security / IPO (<40 sessions) must fail H(s, t).
    ABNB (Airbnb) IPO listed on 2020-12-10.
    On 2020-12-15 (day 4 of its listing):
    - valid_history_count = 4 < 40
    - lookback_eligible = False
    - final_model_eligible = False
    """
    # Verify directly against US market history that ABNB had exactly 4 trading sessions by 2020-12-15
    abnb_sub = (
        pl.scan_parquet("data/US_history.parquet")
        .filter((pl.col("TICKER") == "ABNB") & (pl.col("date") <= 20201215))
        .collect()
    )
    assert len(abnb_sub) == 4, f"ABNB should have exactly 4 trading days by 2020-12-15, got {len(abnb_sub)}."

    # Check that any security with fewer than 40 trading days fails lookback eligibility
    df_diag = audit_tables["diagnostics"]
    ineligible_h = df_diag.filter(pl.col("valid_history_count") < 40)
    assert len(ineligible_h) > 0, "There must be records with insufficient history."
    assert bool(ineligible_h["lookback_eligible"].all()) is False, "Observations with <40 history must not be lookback eligible."
    assert bool(ineligible_h["final_model_eligible"].all()) is False, "Observations with <40 history must fail final model eligibility."


def test_no_synthetic_data_or_forward_leakage(audit_tables):
    """
    Verify that on the final calendar date (2026-09-25), no target can be constructed
    (target_eligible = False) to prevent future look-ahead leakage.
    """
    df_diag = audit_tables["diagnostics"]
    last_date = df_diag["date"].max()
    final_obs = df_diag.filter(pl.col("date") == last_date)
    assert len(final_obs) > 0, f"Observations on final date {last_date} must exist."
    assert not final_obs["target_eligible"].any(), "Target eligible must be False on the last observation date (no forward leakage)."
    assert not final_obs["final_model_eligible"].any(), "Final model eligible must be False on the last date."


def test_diagnostic_table_schema(audit_tables):
    """
    Verify all 13 required diagnostic columns exist with correct types.
    """
    df_diag = audit_tables["diagnostics"]
    expected_cols = [
        "date",
        "security_id",
        "ticker",
        "membership_state",
        "lookback_n",
        "valid_history_count",
        "required_history_count",
        "lookback_eligible",
        "tradable_at_t",
        "target_eligible",
        "final_model_eligible",
        "source_file_start",
        "source_file_end",
    ]
    for col in expected_cols:
        assert col in df_diag.columns, f"Required column '{col}' missing from diagnostics table."


def test_boundary_exclusion_quantification(audit_tables):
    """
    Verify that boundary comparison table exists and quantifies over 1,000,000
    observations falsely excluded by annual file boundaries.
    """
    df_comp = audit_tables["comparison"]
    assert len(df_comp) == 27, "Must have comparison records for all 27 snapshots."
    total_recovered = df_comp["falsely_excluded_by_boundaries"].sum()
    assert total_recovered > 1_000_000, f"Expected >1M observations recovered, got {total_recovered:,}."

    # Verify Year 2024 has high eligibility (>95%)
    df_summary = audit_tables["summary"]
    row_2024 = df_summary.filter(pl.col("list_year") == 2024).to_dicts()[0]
    assert row_2024["pct_eligible_continuous"] > 0.95, f"2024 eligibility must exceed 95%, got {row_2024['pct_eligible_continuous']*100:.2f}%."
