"""Split-contamination verification suite per §3.3.

Tests known stock split events:
- AAPL 2020-08-31 (4-for-1, PERMNO 14593)
- MSFT 2003-02-18 (2-for-1, PERMNO 10107)
- GOOGL 2014-03-27 record date / 2014-04-03 CRSP ex-distribution date (2-for-1, PERMNO 90319)

Verifies:
1. `price_z_20` is NaN for the 20 sessions following the split (`[s_0, ..., s_0+19]`).
2. `drawdown_20` is NaN for the same 20 sessions.
3. `r_cc`, `ret_5d`, `ret_20d`, `vol_20d` are NOT NaN across those 20 sessions.
4. `price_z_20` and `drawdown_20` resume non-NaN values on session 21 (`s_0+20`).
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from src.data.compute_features import compute_layer1_features, compute_layer2_features
from src.data.normalize_schema import normalize_schema

SPLIT_EVENTS = [
    # (label, permno, crsp_split_date, query_start, query_end)
    ("AAPL_2020_4to1", 14593, "2020-08-31", 20200101, 20201231),
    ("MSFT_2003_2to1", 10107, "2003-02-18", 20020601, 20030630),
    # Note: 2014-03-27 was the record date for Google's 2-for-1 Class C stock split;
    # in CRSP bronze.crsp_daily, the ex-distribution date where CFACPR halves (39.938 -> 20.0)
    # and price halves ($1135.10 -> $571.50) is 2014-04-03.
    ("GOOGL_2014_2to1", 90319, "2014-04-03", 20130901, 20140731),
]


@pytest.mark.parametrize(
    "label,permno,split_date,q_start,q_end",
    SPLIT_EVENTS,
    ids=[e[0] for e in SPLIT_EVENTS],
)
def test_split_contamination_event(
    label: str,
    permno: int,
    split_date: str,
    q_start: int,
    q_end: int,
) -> None:
    """Verify 20-session NaN mask on price_z_20 and drawdown_20 and session-21 resumption."""
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
    finally:
        con.close()

    norm_df = normalize_schema(raw_df, "CRSP")
    feat_df = compute_layer2_features(compute_layer1_features(norm_df)).reset_index(
        drop=True
    )

    target_ts = pd.Timestamp(split_date)
    matches = feat_df.index[feat_df["date"] == target_ts].tolist()
    assert len(matches) == 1, f"Split date {split_date} not found for {label}"
    s0_idx = matches[0]

    # Confirm CFACPR actually changed at s0_idx
    assert feat_df.loc[s0_idx, "cfacpr"] != feat_df.loc[s0_idx - 1, "cfacpr"], (
        f"Expected CFACPR change at {split_date} for {label}"
    )

    # Confirm pre-split session (s0_idx - 1) had valid price_z_20 and drawdown_20
    assert pd.notna(feat_df.loc[s0_idx - 1, "price_z_20"])
    assert pd.notna(feat_df.loc[s0_idx - 1, "drawdown_20"])

    # 1 & 2. Verify price_z_20 and drawdown_20 are NaN for the 20 sessions [s0_idx : s0_idx + 20]
    window_20 = feat_df.iloc[s0_idx : s0_idx + 20]
    assert len(window_20) == 20
    assert window_20["price_z_20"].isna().all(), (
        f"{label}: price_z_20 was not NaN for all 20 sessions following split"
    )
    assert window_20["drawdown_20"].isna().all(), (
        f"{label}: drawdown_20 was not NaN for all 20 sessions following split"
    )

    # 3. Verify r_cc, ret_5d, ret_20d, vol_20d are NOT NaN during the 20 sessions
    for col in ["r_cc", "ret_5d", "ret_20d", "vol_20d"]:
        assert window_20[col].notna().all(), (
            f"{label}: {col} unexpectedly contained NaN during split window"
        )

    # 4. Verify price_z_20 and drawdown_20 resume non-NaN values on session 21 (s0_idx + 20)
    s21_row = feat_df.iloc[s0_idx + 20]
    assert pd.notna(s21_row["price_z_20"]), (
        f"{label}: price_z_20 did not resume non-NaN on session 21 ({s21_row['date']})"
    )
    assert pd.notna(s21_row["drawdown_20"]), (
        f"{label}: drawdown_20 did not resume non-NaN on session 21 ({s21_row['date']})"
    )
