"""Identifies missing trading dates between observed OHLCV and expected matrix."""

from __future__ import annotations

import polars as pl

from src.market_data.config import CORRUPTED_DATES


def detect_missing_dates(
    expected_df: pl.DataFrame, observed_prices_df: pl.DataFrame
) -> pl.DataFrame:
    """Compares expected_security_dates against observed_prices to detect missing market dates.

    Preserves corrupted dates as UNKNOWN without fabricating data.
    """
    if expected_df.height == 0:
        return pl.DataFrame(
            schema={
                "security_id": pl.Utf8,
                "date": pl.Utf8,
                "expectation_state": pl.Utf8,
                "missing_reason": pl.Utf8,
            }
        )

    obs_dates_set = (
        set(
            zip(
                observed_prices_df["security_id"].to_list(),
                observed_prices_df["date"].to_list(),
            )
        )
        if observed_prices_df.height > 0
        else set()
    )

    missing_records = []
    for row in expected_df.iter_rows(named=True):
        sec_id = row["security_id"]
        dt = row["date"]
        state = row.get("expectation_state", "OBSERVED_ACTIVE")

        if (sec_id, dt) not in obs_dates_set:
            if dt in CORRUPTED_DATES:
                reason = "CORRUPTED_MASSIVE_SNAPSHOT_KNOWN_OUTAGE"
            elif state == "INFERRED_ACTIVE":
                reason = "INFERRED_GAP_NO_PRICE_RECORDED"
            else:
                reason = "OBSERVED_ACTIVE_NO_PRICE_RETURNED"

            missing_records.append(
                {
                    "security_id": sec_id,
                    "date": dt,
                    "ticker": row.get("ticker", "UNKNOWN"),
                    "expectation_state": state,
                    "missing_reason": reason,
                }
            )

    return pl.DataFrame(missing_records)
