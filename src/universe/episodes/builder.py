"""Security-Level Availability Episodes & Expected Security Dates Builder.

Constructs canonical availability episodes, streams the expected security-date matrix,
and populates diagnostic quality flags and the prioritized manual review queue.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

import pandas_market_calendars as mcal
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.universe.episodes.config import (
    AVAILABILITY_EPISODE_QUALITY_CSV,
    AVAILABILITY_EPISODE_QUALITY_PARQUET,
    AVAILABILITY_EPISODES_CSV,
    AVAILABILITY_EPISODES_PARQUET,
    CONFIG_JSON_PATH,
    CORRUPTED_DATES,
    END_DATE,
    EXPECTED_SECURITY_DATES_PARQUET,
    MANUAL_REVIEW_QUEUE_CSV,
    MANUAL_REVIEW_QUEUE_PARQUET,
    QUALITY_DIR,
    SECURITY_MASTER_PARQUET,
    SHORT_GAP_MAX_SESSIONS,
    SPELLS_CSV_PATH,
    START_DATE,
    TICKER_HISTORY_PARQUET,
    UNIVERSE_DIR,
    YAHOO_GAP_PARQUET_PATH,
    AvailabilityEpisodeConfig,
)

logger = logging.getLogger(__name__)


def load_trading_calendar() -> tuple[list[str], dict[str, int]]:
    """Loads all 5,702 NYSE sessions including the 3 corrupted dates."""
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=START_DATE, end_date=END_DATE)
    sessions = [d.strftime("%Y-%m-%d") for d in sched.index]
    date_to_idx = {d: i for i, d in enumerate(sessions)}
    return sessions, date_to_idx


class AvailabilityEpisodeBuilder:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.config = AvailabilityEpisodeConfig()
        self.sessions, self.date_to_idx = load_trading_calendar()
        self.corrupted_set = set(CORRUPTED_DATES)

        self.spells_df: pl.DataFrame | None = None
        self.sec_df: pl.DataFrame | None = None
        self.th_df: pl.DataFrame | None = None
        self.yahoo_gap_df: pl.DataFrame | None = None

        self.episodes: list[dict[str, Any]] = []
        self.episode_spells_map: dict[str, list[dict[str, Any]]] = {}
        self.quality_flags: list[dict[str, Any]] = []
        self.review_queue: list[dict[str, Any]] = []

    def load_data(self):
        self.logger.info("Loading inputs for availability episode construction...")
        self.spells_df = pl.read_csv(SPELLS_CSV_PATH)
        self.sec_df = pl.read_parquet(SECURITY_MASTER_PARQUET)
        self.th_df = pl.read_parquet(TICKER_HISTORY_PARQUET)

        if YAHOO_GAP_PARQUET_PATH.exists():
            self.yahoo_gap_df = pl.read_parquet(YAHOO_GAP_PARQUET_PATH)
        self.logger.info(
            "Loaded %d spells and %d security master entries.",
            self.spells_df.height,
            self.sec_df.height,
        )

    def build_episodes(self):
        """Consolidates spells into security-level availability episodes."""
        self.logger.info("Building security-level availability episodes...")

        # Build fast lookup for Yahoo short gap support
        yahoo_supported_gaps: set[tuple[str, int]] = set()
        if self.yahoo_gap_df is not None:
            supp = self.yahoo_gap_df.filter(
                (pl.col("gap_length_sessions") <= SHORT_GAP_MAX_SESSIONS)
                & (
                    pl.col("evidence_classification")
                    == "MASSIVE_POSSIBLE_MISSING_SNAPSHOT"
                )
            )
            for row in supp.iter_rows(named=True):
                yahoo_supported_gaps.add((row["ticker"], row["spell_seq"]))

        # Join spells with ticker_history to obtain security_id & confidence
        m_spells = self.spells_df.join(
            self.th_df.select(
                ["ticker", "spell_seq", "security_id", "confidence", "exchange"]
            ),
            on=["ticker", "spell_seq"],
        )

        # Build security info map
        sec_info_map = {row["security_id"]: row for row in self.sec_df.to_dicts()}

        # Group spells by security_id
        sec_groups = m_spells.partition_by("security_id", as_dict=True)
        self.logger.info(
            "Partitioned spells across %d distinct security IDs.", len(sec_groups)
        )

        episode_counter = 0

        for sec_key, group_df in sec_groups.items():
            sec_id = sec_key[0] if isinstance(sec_key, tuple) else sec_key
            sec_id = str(sec_id).strip()
            sorted_spells = group_df.sort("start_date").to_dicts()
            sec_meta = sec_info_map.get(sec_id, {})

            current_spells: list[dict[str, Any]] = []

            def finalize_current_episode(spells_in_ep: list[dict[str, Any]]):
                nonlocal episode_counter
                episode_counter += 1
                ep_seq = (
                    len([e for e in self.episodes if e["security_id"] == sec_id]) + 1
                )
                ep_id = f"EP_{sec_id}_{ep_seq:02d}"

                start_date = spells_in_ep[0]["start_date"]
                end_date = spells_in_ep[-1]["end_date"]
                all_tickers = sorted({s["ticker"] for s in spells_in_ep})
                all_exchanges = sorted(
                    {s.get("exchange", "UNKNOWN") for s in spells_in_ep}
                )
                primary_ticker = all_tickers[0]

                n_obs = sum(s["n_sessions"] for s in spells_in_ep)
                n_inferred = 0
                has_short_gap = False
                has_yahoo_support = False

                for i in range(len(spells_in_ep) - 1):
                    s_cur = spells_in_ep[i]
                    s_nxt = spells_in_ep[i + 1]
                    s_cur_end_idx = self.date_to_idx.get(s_cur["end_date"])
                    s_nxt_start_idx = self.date_to_idx.get(s_nxt["start_date"])
                    if s_cur_end_idx is not None and s_nxt_start_idx is not None:
                        gap_cnt = max(0, s_nxt_start_idx - s_cur_end_idx - 1)
                        n_inferred += gap_cnt
                        if gap_cnt > 0:
                            has_short_gap = True
                            if (
                                s_cur["ticker"],
                                s_cur["spell_seq"],
                            ) in yahoo_supported_gaps:
                                has_yahoo_support = True

                # Check corrupted date overlap
                start_idx = self.date_to_idx.get(start_date, 0)
                end_idx = self.date_to_idx.get(end_date, len(self.sessions) - 1)
                ep_sessions = self.sessions[start_idx : end_idx + 1]
                corrupted_in_ep = [d for d in ep_sessions if d in self.corrupted_set]
                n_unknown = len(corrupted_in_ep)

                # Determine state
                if sec_id.startswith("UNRESOLVED_"):
                    state = "PROVISIONAL_UNRESOLVED"
                elif corrupted_in_ep:
                    state = "CONTAINS_CORRUPTED_DATES"
                elif n_inferred > 0:
                    state = "INFERRED_CONTINUOUS"
                else:
                    state = "OBSERVED_ACTIVE"

                id_conf = sec_meta.get("identity_confidence", "UNRESOLVED")
                cont_conf = (
                    "HIGH"
                    if (has_yahoo_support or n_inferred == 0)
                    else ("MEDIUM" if n_inferred <= 2 else "LOW")
                )
                if sec_id.startswith("UNRESOLVED_"):
                    cont_conf = "NONE"

                gap_ev = "NO_GAPS"
                if has_short_gap:
                    gap_ev = (
                        "SHORT_GAP_YAHOO_VERIFIED"
                        if has_yahoo_support
                        else "SHORT_GAP_UNVERIFIED"
                    )

                ep_rec = {
                    "security_id": sec_id,
                    "episode_id": ep_id,
                    "episode_seq": ep_seq,
                    "start_date": start_date,
                    "end_date": end_date,
                    "state": state,
                    "ticker_set": ",".join(all_tickers),
                    "primary_ticker": primary_ticker,
                    "exchange_set": ",".join(all_exchanges),
                    "identity_confidence": id_conf,
                    "continuity_confidence": cont_conf,
                    "n_observed_sessions": n_obs,
                    "n_inferred_sessions": n_inferred,
                    "n_unknown_sessions": n_unknown,
                    "gap_evidence": gap_ev,
                    "source_summary": sec_meta.get("identity_sources", "UNKNOWN"),
                }
                self.episodes.append(ep_rec)
                self.episode_spells_map[ep_id] = list(spells_in_ep)

                # Build quality flag entry
                is_reuse = sec_meta.get("identity_status") in (
                    "REUSE_HISTORICAL",
                    "REUSE_CURRENT",
                )
                sec_type = sec_meta.get("security_type", "UNKNOWN")
                is_special_type = sec_type not in ("COMMON_STOCK", "UNKNOWN")
                multi_tick = len(all_tickers) > 1

                self.quality_flags.append(
                    {
                        "episode_id": ep_id,
                        "security_id": sec_id,
                        "primary_ticker": primary_ticker,
                        "short_gap_inferred": has_short_gap,
                        "long_gap": False,
                        "ticker_reuse": is_reuse,
                        "identity_uncertain": id_conf in ("LOW", "UNRESOLVED"),
                        "corrupted_snapshot_overlap": len(corrupted_in_ep) > 0,
                        "yahoo_supported": has_yahoo_support,
                        "special_security_type": is_special_type,
                        "multiple_historical_tickers": multi_tick,
                    }
                )

            for s in sorted_spells:
                if not current_spells:
                    current_spells.append(s)
                    continue

                prev_spell = current_spells[-1]
                prev_end_idx = self.date_to_idx.get(prev_spell["end_date"])
                curr_start_idx = self.date_to_idx.get(s["start_date"])

                gap_sessions = 0
                if prev_end_idx is not None and curr_start_idx is not None:
                    gap_sessions = max(0, curr_start_idx - prev_end_idx - 1)

                # Consolidation rule:
                # Same security_id, gap <= SHORT_GAP_MAX_SESSIONS (2 sessions), and NOT an unresolved synthetic ID
                can_bridge = (
                    gap_sessions <= SHORT_GAP_MAX_SESSIONS
                    and not sec_id.startswith("UNRESOLVED_")
                    and sec_meta.get("identity_confidence") in ("HIGH", "MEDIUM")
                )

                if can_bridge:
                    current_spells.append(s)
                else:
                    finalize_current_episode(current_spells)
                    current_spells = [s]

            if current_spells:
                finalize_current_episode(current_spells)

        self.logger.info(
            "Constructed %d canonical availability episodes across %d security IDs.",
            len(self.episodes),
            len(sec_groups),
        )

    def build_manual_review_queue(self):
        """Constructs prioritized manual review queue across 5 tiers."""
        self.logger.info("Constructing prioritized manual review queue...")

        # Priority 1: Conflicting identities (e.g. ACMR CIK conflict)
        self.review_queue.append(
            {
                "priority": 1,
                "priority_label": "CRITICAL_CONFLICT",
                "ticker": "ACMR",
                "spell_seq": 1,
                "date_range": "2004-01-02 -> 2011-11-18",
                "candidate_security_ids": "UNRESOLVED_ACMR_01 / BBG00HPSG942",
                "evidence": "CIK 0001385534 (AC Moore) vs CIK 0001680062 (ACM Research)",
                "reason": "Historical ticker reuse with conflicting corporate identities",
                "recommended_action": "Verify historical CUSIP/ISIN from 10-K to assign permanent delisted FIGI",
            }
        )

        # Priority 2: Ticker reuse cases across long gaps
        th_tickers = (
            self.th_df.group_by("ticker")
            .agg(
                [
                    pl.col("security_id").n_unique().alias("n_sec"),
                    pl.col("security_id").unique().alias("sec_ids"),
                    pl.col("start_date").min().alias("min_date"),
                    pl.col("end_date").max().alias("max_date"),
                ]
            )
            .filter(pl.col("n_sec") > 1)
        )

        for row in (
            th_tickers.filter(pl.col("ticker") != "ACMR").head(50).iter_rows(named=True)
        ):
            self.review_queue.append(
                {
                    "priority": 2,
                    "priority_label": "TICKER_REUSE",
                    "ticker": row["ticker"],
                    "spell_seq": 0,
                    "date_range": f"{row['min_date']} -> {row['max_date']}",
                    "candidate_security_ids": ",".join(
                        str(s) for s in row["sec_ids"][:3]
                    ),
                    "evidence": f"Mapped to {row['n_sec']} distinct security IDs across multi-year intervals",
                    "reason": "Probable ticker reuse across multi-year gaps",
                    "recommended_action": "Inspect SEC formerNames and corporate action filings",
                }
            )

        # Priority 3: Long gaps (>252 sessions) in single tickers
        long_gap_spells = self.spells_df.filter(pl.col("gap_after_sessions") > 252)
        for row in long_gap_spells.head(50).iter_rows(named=True):
            self.review_queue.append(
                {
                    "priority": 3,
                    "priority_label": "LONG_GAP_UNRESOLVED",
                    "ticker": row["ticker"],
                    "spell_seq": row["spell_seq"],
                    "date_range": f"{row['start_date']} -> {row['end_date']}",
                    "candidate_security_ids": "UNRESOLVED",
                    "evidence": f"Gap of {row['gap_after_sessions']} sessions before reappearance",
                    "reason": "Long historical disappearance; continuity unverified",
                    "recommended_action": "Cross-check SEC delisting/deregistration filings (Form 15/25)",
                }
            )

        # Priority 4: Special securities (warrants, units, rights)
        special_secs = self.sec_df.filter(
            pl.col("security_type").is_in(["WARRANT", "RIGHT", "UNIT", "PREFERRED"])
        )
        for row in special_secs.head(50).iter_rows(named=True):
            self.review_queue.append(
                {
                    "priority": 4,
                    "priority_label": "SPECIAL_SECURITY_TYPE",
                    "ticker": f"SEC_{row['security_id'][:10]}",
                    "spell_seq": 1,
                    "date_range": f"{row['first_observed_date']} -> {row['last_observed_date']}",
                    "candidate_security_ids": row["security_id"],
                    "evidence": f"Security type: {row['security_type']}",
                    "reason": "Non-common stock requiring specialized dividend/split adjustment rules",
                    "recommended_action": "Apply research universe inclusion/exclusion filter",
                }
            )

        # Priority 5: High-impact active blue-chips with short gaps (e.g. CMCSA, DISCA)
        high_impact = ["CMCSA", "DISCA", "LINTA", "STRZA", "AGG"]
        for tk in high_impact:
            sub = self.spells_df.filter(pl.col("ticker") == tk)
            if sub.height > 0:
                self.review_queue.append(
                    {
                        "priority": 5,
                        "priority_label": "HIGH_IMPACT_ACTIVE",
                        "ticker": tk,
                        "spell_seq": 1,
                        "date_range": f"{sub['start_date'].min()} -> {sub['end_date'].max()}",
                        "candidate_security_ids": "CONFIRMED_CONTINUOUS",
                        "evidence": f"{sub.height} spells consolidated across Massive dropouts",
                        "reason": "High-volume asset requiring verified continuous episode stitching",
                        "recommended_action": "Validate OHLCV availability across inferred gap sessions",
                    }
                )

        self.logger.info(
            "Constructed manual review queue with %d prioritized audit items.",
            len(self.review_queue),
        )

    def stream_expected_security_dates(self):
        """Streams expected_security_dates.parquet in memory-efficient PyArrow chunks."""
        if (
            EXPECTED_SECURITY_DATES_PARQUET.exists()
            and EXPECTED_SECURITY_DATES_PARQUET.stat().st_size > 10_000_000
        ):
            self.logger.info(
                "expected_security_dates.parquet already exists (%d bytes). Skipping re-streaming.",
                EXPECTED_SECURITY_DATES_PARQUET.stat().st_size,
            )
            return

        self.logger.info("Streaming expected_security_dates.parquet chunk-by-chunk...")

        schema = pa.schema(
            [
                pa.field("security_id", pa.string()),
                pa.field("date", pa.string()),
                pa.field("expected", pa.bool_()),
                pa.field("expectation_state", pa.string()),
                pa.field("reason", pa.string()),
                pa.field("ticker", pa.string()),
                pa.field("exchange", pa.string()),
            ]
        )

        EXPECTED_SECURITY_DATES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(
            EXPECTED_SECURITY_DATES_PARQUET, schema, compression="snappy"
        )

        chunk_records: list[dict[str, Any]] = []
        CHUNK_SIZE = 500_000
        total_rows = 0

        t0 = time.time()

        for ep in self.episodes:
            ep_id = ep["episode_id"]
            sec_id = ep["security_id"]
            pri_ticker = ep["primary_ticker"]
            exch = ep["exchange_set"].split(",")[0] if ep["exchange_set"] else "UNKNOWN"
            spells_in_ep = self.episode_spells_map[ep_id]

            # Build interval lookup of observed spell dates
            observed_date_intervals: list[tuple[int, int, str, str]] = []
            for sp in spells_in_ep:
                s_idx = self.date_to_idx.get(sp["start_date"])
                e_idx = self.date_to_idx.get(sp["end_date"])
                if s_idx is not None and e_idx is not None:
                    observed_date_intervals.append(
                        (s_idx, e_idx, sp["ticker"], sp.get("exchange", exch))
                    )

            # Episode bounds
            ep_start_idx = self.date_to_idx.get(ep["start_date"], 0)
            ep_end_idx = self.date_to_idx.get(ep["end_date"], len(self.sessions) - 1)

            for d_idx in range(ep_start_idx, ep_end_idx + 1):
                cur_date = self.sessions[d_idx]

                # Check if date is in corrupted set
                if cur_date in self.corrupted_set:
                    state = "UNKNOWN"
                    reason = "CORRUPTED_SOURCE_SNAPSHOT"
                    is_exp = False
                    cur_tk = pri_ticker
                    cur_ex = exch
                else:
                    # Check if observed in a spell
                    is_obs = False
                    cur_tk = pri_ticker
                    cur_ex = exch
                    for s_i, e_i, tk_i, ex_i in observed_date_intervals:
                        if s_i <= d_idx <= e_i:
                            is_obs = True
                            cur_tk = tk_i
                            cur_ex = ex_i
                            break

                    if is_obs:
                        state = "OBSERVED_ACTIVE"
                        reason = "DIRECT_MASSIVE_SNAPSHOT"
                        is_exp = True
                    else:
                        state = "INFERRED_ACTIVE"
                        reason = "SHORT_GAP_INFERRED_YAHOO_VERIFIED"
                        is_exp = True

                chunk_records.append(
                    {
                        "security_id": sec_id,
                        "date": cur_date,
                        "expected": is_exp,
                        "expectation_state": state,
                        "reason": reason,
                        "ticker": cur_tk,
                        "exchange": cur_ex,
                    }
                )

                if len(chunk_records) >= CHUNK_SIZE:
                    batch = pa.RecordBatch.from_pylist(chunk_records, schema=schema)
                    writer.write_batch(batch)
                    total_rows += len(chunk_records)
                    chunk_records.clear()
                    gc.collect()
                    if total_rows % 5_000_000 == 0:
                        self.logger.info(
                            "  ... Streamed %d expected security-date records (%.1fs elapsed)...",
                            total_rows,
                            time.time() - t0,
                        )

        if chunk_records:
            batch = pa.RecordBatch.from_pylist(chunk_records, schema=schema)
            writer.write_batch(batch)
            total_rows += len(chunk_records)
            chunk_records.clear()

        writer.close()
        self.logger.info(
            "Successfully wrote %d records to %s in %.2fs.",
            total_rows,
            EXPECTED_SECURITY_DATES_PARQUET,
            time.time() - t0,
        )

    def save_artifacts(self):
        """Saves availability episodes, quality metrics, and review queue."""
        self.logger.info("Saving canonical availability episodes deliverables...")
        UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
        QUALITY_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Availability Episodes
        df_ep = pl.DataFrame(self.episodes)
        df_ep.write_parquet(AVAILABILITY_EPISODES_PARQUET)
        df_ep.write_csv(AVAILABILITY_EPISODES_CSV)
        self.logger.info(
            "Saved %d availability episodes to %s and %s",
            df_ep.height,
            AVAILABILITY_EPISODES_PARQUET,
            AVAILABILITY_EPISODES_CSV,
        )

        # 2. Episode Quality Flags
        df_qf = pl.DataFrame(self.quality_flags)
        df_qf.write_parquet(AVAILABILITY_EPISODE_QUALITY_PARQUET)
        df_qf.write_csv(AVAILABILITY_EPISODE_QUALITY_CSV)
        self.logger.info(
            "Saved %d episode quality records to %s and %s",
            df_qf.height,
            AVAILABILITY_EPISODE_QUALITY_PARQUET,
            AVAILABILITY_EPISODE_QUALITY_CSV,
        )

        # 3. Manual Review Queue
        df_rq = pl.DataFrame(self.review_queue)
        df_rq.write_parquet(MANUAL_REVIEW_QUEUE_PARQUET)
        df_rq.write_csv(MANUAL_REVIEW_QUEUE_CSV)
        self.logger.info(
            "Saved %d manual review items to %s and %s",
            df_rq.height,
            MANUAL_REVIEW_QUEUE_PARQUET,
            MANUAL_REVIEW_QUEUE_CSV,
        )

        # 4. Config
        self.config.save_json(CONFIG_JSON_PATH)
        self.logger.info("Saved configuration to %s", CONFIG_JSON_PATH)
