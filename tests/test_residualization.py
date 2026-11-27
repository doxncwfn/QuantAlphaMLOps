"""Lagged-beta verification suite for Layer 4 six-factor residualization per §3.4.

For 5 selected (permno, date) pairs:
1. Extracts the 252 trading sessions strictly preceding `date` ([s-252, s-1]).
2. Manually runs `np.linalg.lstsq(X, y)` where `X = [1, mktrf, smb, hml, rmw, cma, umd]`
   and `y = R^e = ret_total - rf` on `[s-252, s-1]`.
3. Computes `epsilon_manual = R^e_s - (alpha_hat + beta_hat @ F_s)`.
4. Asserts `abs(resid_ret - epsilon_manual) < 1e-6` against both `compute_layer4_features`
   (fast batched solver) and `compute_residuals` (single-stock reference).
5. Repeats OLS on `[s-251, s]` (same-day leaky window) and asserts
   `abs(epsilon_manual - epsilon_leaky) > 1e-8`.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.data.compute_features import (
    FACTOR_COLS,
    compute_layer4_features,
    compute_residuals,
)
from src.data.normalize_schema import normalize_schema

SAMPLE_PAIRS = [
    # (permno, ticker_label, eval_date, query_start, query_end)
    (14593, "AAPL", "2015-06-15", 20130101, 20151231),
    (10107, "MSFT", "2010-06-15", 20080101, 20101231),
    (12490, "IBM", "2005-06-15", 20030101, 20051231),
    (11850, "XOM", "2018-09-19", 20160101, 20181231),
    (47896, "JPM", "2021-11-10", 20190101, 20211231),
]


@pytest.mark.parametrize(
    "permno,label,eval_date,q_start,q_end",
    SAMPLE_PAIRS,
    ids=[f"{p[1]}_{p[2]}" for p in SAMPLE_PAIRS],
)
def test_lagged_beta_residualization(
    permno: int,
    label: str,
    eval_date: str,
    q_start: int,
    q_end: int,
) -> None:
    """Verify lagged [s-252, s-1] beta estimation vs manual lstsq and vs [s-251, s] leaky beta."""
    con = duckdb.connect("data/warehouse/russell1000.duckdb", read_only=True)
    try:
        raw_df = con.execute(
            f"""
            SELECT PERMNO,
                   strptime(cast(date AS VARCHAR), '%Y%m%d')::DATE AS date,
                   TICKER, PRC, OPENPRC, ASKHI, BIDLO, VOL, RET, RETX,
                   CFACPR, CFACSHR, SHROUT, DLSTCD, DLRET, DLRETX, SHRCD
            FROM bronze.crsp_daily
            WHERE PERMNO = {permno}
              AND date >= {q_start} AND date <= {q_end}
              AND PRC IS NOT NULL AND abs(PRC) > 0
            ORDER BY date
            """
        ).df()

        factors_df = con.execute(
            """
            SELECT date, mktrf, smb, hml, rmw, cma, umd, rf
            FROM silver.market_factor
            WHERE date >= DATE '2002-01-01' AND date <= DATE '2022-12-31'
            ORDER BY date
            """
        ).df()
        factors_df["date"] = pd.to_datetime(factors_df["date"])
    finally:
        con.close()

    norm_df = normalize_schema(raw_df, "CRSP")
    feat_df = compute_layer4_features(norm_df, factors_df).reset_index(drop=True)

    target_ts = pd.Timestamp(eval_date)
    matches = feat_df.index[feat_df["date"] == target_ts].tolist()
    assert len(matches) == 1, f"Date {eval_date} not found for {label} ({permno})"
    s_idx = matches[0]
    assert s_idx >= 252

    # Align factors to stock series
    fac_indexed = factors_df.set_index("date")
    f_aligned = fac_indexed.reindex(feat_df["date"]).reset_index(drop=True)
    r_excess = feat_df["ret_total"].to_numpy(dtype=np.float64) - f_aligned[
        "rf"
    ].to_numpy(dtype=np.float64)
    F_mat = f_aligned[FACTOR_COLS].to_numpy(dtype=np.float64)

    # 1 & 2. Manual OLS on [s-252, s-1]
    y_lagged = r_excess[s_idx - 252 : s_idx]
    F_lagged = F_mat[s_idx - 252 : s_idx]
    X_lagged = np.column_stack([np.ones(252, dtype=np.float64), F_lagged])
    params_lagged = np.linalg.lstsq(X_lagged, y_lagged, rcond=None)[0]
    alpha_hat = params_lagged[0]
    beta_hat = params_lagged[1:]

    # 3. Compute manual residual at s
    F_s = F_mat[s_idx]
    R_e_s = r_excess[s_idx]
    epsilon_manual = float(R_e_s - (alpha_hat + np.dot(beta_hat, F_s)))

    # 4. Compare against pipeline resid_ret and reference compute_residuals()
    pipeline_resid = float(feat_df.loc[s_idx, "resid_ret"])
    assert abs(pipeline_resid - epsilon_manual) < 1e-6, (
        f"{label} {eval_date}: pipeline resid_ret ({pipeline_resid}) != manual ({epsilon_manual})"
    )

    ref_res = compute_residuals(
        pd.Series(r_excess, index=feat_df["date"]),
        fac_indexed[FACTOR_COLS],
    )
    ref_resid = float(ref_res["epsilon"].loc[target_ts])
    assert abs(ref_resid - epsilon_manual) < 1e-12

    # Verify stored historical residuals are used in resid_mom_20 (not recomputed with beta_s)
    expected_mom_20 = float(feat_df.loc[s_idx - 19 : s_idx, "resid_ret"].sum())
    pipeline_mom_20 = float(feat_df.loc[s_idx, "resid_mom_20"])
    assert abs(pipeline_mom_20 - expected_mom_20) < 1e-12

    # 5. Repeat OLS using [s-251, s] (including same-day s) to compute epsilon_leaky
    y_leaky = r_excess[s_idx - 251 : s_idx + 1]
    F_leaky = F_mat[s_idx - 251 : s_idx + 1]
    X_leaky = np.column_stack([np.ones(252, dtype=np.float64), F_leaky])
    params_leaky = np.linalg.lstsq(X_leaky, y_leaky, rcond=None)[0]
    epsilon_leaky = float(R_e_s - (params_leaky[0] + np.dot(params_leaky[1:], F_s)))

    assert abs(epsilon_manual - epsilon_leaky) > 1e-8, (
        f"{label} {eval_date}: lagged vs same-day beta residuals unexpectedly identical"
    )
