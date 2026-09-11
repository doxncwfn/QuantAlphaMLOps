"""Constructs the prioritized market data acquisition queue."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import polars as pl

from src.market_data.config import (
    AVAILABILITY_EPISODES_PARQUET,
    CALENDAR_BUFFER_DAYS,
    MARKET_DATA_QUEUE_CSV,
    MARKET_DATA_QUEUE_PARQUET,
    MARKET_DIR,
    SECURITY_MASTER_PARQUET,
)

logger = logging.getLogger(__name__)


def build_market_data_queue() -> pl.DataFrame:
    """Builds prioritized market_data_queue from availability_episodes and security_master."""
    logger.info(
        "Building market data acquisition queue from %s...",
        AVAILABILITY_EPISODES_PARQUET,
    )

    episodes = pl.read_parquet(AVAILABILITY_EPISODES_PARQUET)
    sec_master = pl.read_parquet(SECURITY_MASTER_PARQUET)

    # Join with security_master to get share_class_figi, cik, security_type
    joined = episodes.join(
        sec_master.select(["security_id", "share_class_figi", "cik", "security_type"]),
        on="security_id",
        how="left",
    )

    records = []
    for row in joined.iter_rows(named=True):
        sec_id = row["security_id"]
        ep_id = row["episode_id"]
        pri_ticker = row["primary_ticker"]
        st_date = row["start_date"]
        en_date = row["end_date"]
        figi = row.get("share_class_figi")
        cik = row.get("cik")
        sec_type = row.get("security_type", "UNKNOWN")
        conf = row.get("identity_confidence", "UNRESOLVED")

        # Buffer dates by 30 days
        try:
            dt_start = (
                datetime.strptime(st_date, "%Y-%m-%d")
                - timedelta(days=CALENDAR_BUFFER_DAYS)
            ).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            dt_start = st_date

        try:
            dt_end = (
                datetime.strptime(en_date, "%Y-%m-%d")
                + timedelta(days=CALENDAR_BUFFER_DAYS)
            ).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            dt_end = en_date

        # Priority stratification
        if figi is not None:
            priority = "PRIORITY_1_FIGI_ACTIVE"
        elif cik is not None:
            priority = "PRIORITY_2_CIK_BACKED"
        else:
            priority = "PRIORITY_3_UNRESOLVED"

        records.append(
            {
                "security_id": sec_id,
                "ticker": pri_ticker,
                "date_start": dt_start,
                "date_end": dt_end,
                "source_priority": priority,
                "security_type": sec_type,
                "confidence": conf,
                "episode_id": ep_id,
            }
        )

    df_queue = pl.DataFrame(records)

    # Sort by priority and start date
    df_queue = df_queue.sort(["source_priority", "date_start", "ticker"])

    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    df_queue.write_parquet(MARKET_DATA_QUEUE_PARQUET)
    df_queue.write_csv(MARKET_DATA_QUEUE_CSV)

    logger.info(
        "Saved %d queue records to %s and %s",
        df_queue.height,
        MARKET_DATA_QUEUE_PARQUET,
        MARKET_DATA_QUEUE_CSV,
    )

    # Log priority breakdown
    p_counts = df_queue["source_priority"].value_counts().sort("count", descending=True)
    for p in p_counts.iter_rows(named=True):
        logger.info(
            "  - %s: %d episodes (%.1f%%)",
            p["source_priority"],
            p["count"],
            p["count"] / df_queue.height * 100,
        )

    return df_queue


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_market_data_queue()
