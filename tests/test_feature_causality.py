"""Causality verification suite: 30 tests (one per v1.3 feature) per §3.1.

Verifies that for every feature f in Layer 1 (6), Layer 2 (10), Layer 4 (3), and Layer 5 (11),
computing f on the full dataset produces the exact same values at date T as computing f on
raw data truncated to date <= T.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.data.build_universe import assign_in_universe
from src.data.compute_features import ALL_FEATURE_COLS, compute_all_features
from src.data.normalize_schema import normalize_schema, seed_everything

TEST_DATE = pd.Timestamp("2018-06-15")


@pytest.fixture(scope="module")
def causality_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build full vs truncated feature panels around TEST_DATE=2018-06-15 for 60 Russell 1000 stocks."""
    seed_everything(42)
    con = duckdb.connect("data/warehouse/russell1000.duckdb", read_only=True)
    try:
        # Select 60 active Russell 1000 constituents in 2017-2018
        russell_df = con.execute(
            """
            SELECT year AS Year, ticker AS Ticker, security_id AS permno
            FROM silver.russell
            WHERE year IN (2017, 2018) AND security_id IS NOT NULL
            """
        ).df()
        sample_permnos = sorted(russell_df["permno"].dropna().astype(int).unique())[:60]
        permno_list_sql = ",".join(str(p) for p in sample_permnos)

        raw_crsp = con.execute(
            f"""
            SELECT PERMNO,
                   strptime(cast(date AS VARCHAR), '%Y%m%d')::DATE AS date,
                   TICKER, PRC, OPENPRC, ASKHI, BIDLO, VOL, RET, RETX,
                   CFACPR, CFACSHR, SHROUT, DLSTCD, DLRET, DLRETX, SHRCD
            FROM bronze.crsp_daily
            WHERE PERMNO IN ({permno_list_sql})
              AND date >= 20160101 AND date <= 20180930
              AND PRC IS NOT NULL AND abs(PRC) > 0
            ORDER BY PERMNO, date
            """
        ).df()

        factors_df = con.execute(
            """
            SELECT date, mktrf, smb, hml, rmw, cma, umd, rf
            FROM silver.market_factor
            WHERE date >= DATE '2016-01-01' AND date <= DATE '2018-09-30'
            ORDER BY date
            """
        ).df()
        factors_df["date"] = pd.to_datetime(factors_df["date"])
    finally:
        con.close()

    norm_full = normalize_schema(raw_crsp, "CRSP")
    norm_full["in_universe"] = assign_in_universe(norm_full, russell_df)

    # 1. Full dataset computation (includes future dates up to 2018-09-30)
    feat_full = compute_all_features(norm_full, factors_df)
    full_at_T = (
        feat_full.loc[feat_full["date"] == TEST_DATE]
        .sort_values("permno")
        .reset_index(drop=True)
    )

    # 2. Truncated dataset computation (strictly date <= TEST_DATE)
    raw_trunc = raw_crsp.loc[pd.to_datetime(raw_crsp["date"]) <= TEST_DATE].copy()
    factors_trunc = factors_df.loc[factors_df["date"] <= TEST_DATE].copy()
    norm_trunc = normalize_schema(raw_trunc, "CRSP")
    norm_trunc["in_universe"] = assign_in_universe(norm_trunc, russell_df)

    feat_trunc = compute_all_features(norm_trunc, factors_trunc)
    trunc_at_T = (
        feat_trunc.loc[feat_trunc["date"] == TEST_DATE]
        .sort_values("permno")
        .reset_index(drop=True)
    )

    return full_at_T, trunc_at_T


@pytest.mark.parametrize("feature_name", ALL_FEATURE_COLS)
def test_feature_causality_at_T(
    causality_pair: tuple[pd.DataFrame, pd.DataFrame],
    feature_name: str,
) -> None:
    """Verify feature at date T is identical whether future data (> T) is present or truncated."""
    full_at_T, trunc_at_T = causality_pair
    assert len(full_at_T) == len(trunc_at_T) and len(full_at_T) >= 50

    s_full = full_at_T[feature_name].to_numpy(dtype=np.float64)
    s_trunc = trunc_at_T[feature_name].to_numpy(dtype=np.float64)

    # Ensure the feature is actually populated (non-NaN) on TEST_DATE
    assert np.isfinite(s_full).sum() > 0, (
        f"Feature {feature_name} is all-NaN on {TEST_DATE}"
    )

    np.testing.assert_allclose(
        s_full,
        s_trunc,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
        err_msg=f"Causality violation detected for feature {feature_name} at {TEST_DATE}",
    )
