"""Future-noise leakage verification suite: 30 tests (one per v1.3 feature) per §3.2.

Verifies that replacing all price, volume, return, and factor data strictly after date T
with random noise leaves all 30 features on date T completely unchanged.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.data.build_universe import assign_in_universe
from src.data.compute_features import ALL_FEATURE_COLS, compute_all_features
from src.data.compute_target import compute_target
from src.data.normalize_schema import normalize_schema, seed_everything

TEST_DATE = pd.Timestamp("2018-06-15")


@pytest.fixture(scope="module")
def leakage_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build original vs future-perturbed feature panels around TEST_DATE=2018-06-15."""
    seed_everything(42)
    con = duckdb.connect("data/warehouse/russell1000.duckdb", read_only=True)
    try:
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

    # 1. Original computation
    norm_orig = normalize_schema(raw_crsp, "CRSP")
    norm_orig["in_universe"] = assign_in_universe(norm_orig, russell_df)
    feat_orig = compute_all_features(norm_orig, factors_df)
    feat_orig = compute_target(feat_orig)
    orig_at_T = (
        feat_orig.loc[feat_orig["date"] == TEST_DATE]
        .sort_values("permno")
        .reset_index(drop=True)
    )

    # 2. Replace all data after TEST_DATE with random noise and recompute from scratch
    rng = np.random.RandomState(999)
    raw_noisy = raw_crsp.copy()
    raw_dates = pd.to_datetime(raw_noisy["date"])
    after_T = raw_dates > TEST_DATE
    n_after = int(after_T.sum())

    noise_close = rng.uniform(10.0, 500.0, size=n_after)
    raw_noisy.loc[after_T, "PRC"] = noise_close
    raw_noisy.loc[after_T, "OPENPRC"] = noise_close * rng.uniform(
        0.95, 1.05, size=n_after
    )
    raw_noisy.loc[after_T, "ASKHI"] = noise_close * rng.uniform(
        1.05, 1.15, size=n_after
    )
    raw_noisy.loc[after_T, "BIDLO"] = noise_close * rng.uniform(
        0.85, 0.95, size=n_after
    )
    raw_noisy.loc[after_T, "VOL"] = rng.randint(100_000, 100_000_000, size=n_after)
    raw_noisy.loc[after_T, "RETX"] = [
        f"{v:.6f}" for v in rng.normal(0.0, 0.08, size=n_after)
    ]
    raw_noisy.loc[after_T, "RET"] = [
        f"{v:.6f}" for v in rng.normal(0.0, 0.08, size=n_after)
    ]
    raw_noisy.loc[after_T, "CFACPR"] = rng.choice([1.0, 2.0, 4.0], size=n_after)

    factors_noisy = factors_df.copy()
    f_after = factors_noisy["date"] > TEST_DATE
    n_f_after = int(f_after.sum())
    for col in ["mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]:
        factors_noisy.loc[f_after, col] = rng.normal(0.0, 0.05, size=n_f_after)

    norm_noisy = normalize_schema(raw_noisy, "CRSP")
    norm_noisy["in_universe"] = assign_in_universe(norm_noisy, russell_df)
    feat_noisy = compute_all_features(norm_noisy, factors_noisy)
    feat_noisy = compute_target(feat_noisy)
    noisy_at_T = (
        feat_noisy.loc[feat_noisy["date"] == TEST_DATE]
        .sort_values("permno")
        .reset_index(drop=True)
    )

    return orig_at_T, noisy_at_T


@pytest.mark.parametrize("feature_name", ALL_FEATURE_COLS)
def test_feature_no_future_leakage(
    leakage_pair: tuple[pd.DataFrame, pd.DataFrame],
    feature_name: str,
) -> None:
    """Verify feature at date T is unaffected when all data after T is replaced with noise."""
    orig_at_T, noisy_at_T = leakage_pair
    assert len(orig_at_T) == len(noisy_at_T) and len(orig_at_T) >= 50

    s_orig = orig_at_T[feature_name].to_numpy(dtype=np.float64)
    s_noisy = noisy_at_T[feature_name].to_numpy(dtype=np.float64)

    assert np.isfinite(s_orig).sum() > 0, (
        f"Feature {feature_name} is all-NaN on {TEST_DATE}"
    )

    np.testing.assert_allclose(
        s_orig,
        s_noisy,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
        err_msg=f"Future data leakage detected for feature {feature_name} at {TEST_DATE}",
    )
