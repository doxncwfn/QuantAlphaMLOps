"""
Single Worker Channel Execution Unit.
=====================================
Encapsulates execution on a dedicated API key slot with private rate limiting.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Any

from src.identity.massive.cache import CacheManager
from src.identity.massive.client import MassiveClient
from src.identity.massive.date_strategy import RepresentativeDateStrategy
from src.identity.massive.rate_limiter import PerKeyRateLimiter
from src.identity.massive.telemetry import WorkerTelemetry


class MassiveWorker:
    """Worker channel pinned 1:1 to an API key slot."""

    def __init__(
        self,
        worker_id: str,
        api_key: str,
        cache_manager: CacheManager,
        telemetry: WorkerTelemetry,
        date_strategy: RepresentativeDateStrategy,
        min_interval_seconds: float = 12.1,
        logger: logging.Logger | None = None,
    ):
        self.worker_id = worker_id
        self._api_key = api_key
        self.cache_manager = cache_manager
        self.telemetry = telemetry
        self.date_strategy = date_strategy
        self.logger = logger or logging.getLogger(f"worker.{worker_id}")

        self.rate_limiter = PerKeyRateLimiter(
            min_interval_seconds=min_interval_seconds,
            logger=self.logger,
        )
        self.client = MassiveClient(
            api_key=api_key,
            rate_limiter=self.rate_limiter,
            logger=self.logger,
        )
        self.telemetry.init_worker(self.worker_id)

    def query_single(
        self,
        ticker: str,
        query_date: str,
        spell_id: str,
        allow_live: bool = True,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Queries single ticker:date from cache or live API."""
        clean_tk = ticker.strip().upper()
        t_req = time.time()

        # 1. Cache hit check
        matched, raw_data, hit = self.cache_manager.get(clean_tk, query_date)
        if hit:
            t_resp = time.time()
            outcome = (
                "SUCCESS"
                if (
                    matched
                    and (
                        matched.get("cik")
                        or matched.get("share_class_figi")
                        or matched.get("composite_figi")
                    )
                )
                else "MASSIVE_EMPTY"
            )
            self.telemetry.record_request(
                worker_id=self.worker_id,
                spell_id=spell_id,
                ticker=clean_tk,
                query_date=query_date,
                req_ts=t_req,
                resp_ts=t_resp,
                http_status=200,
                retry_count=0,
                is_cache=True,
                is_live=False,
                outcome_category=outcome,
            )
            return matched, {
                "source": "CACHE",
                "worker": self.worker_id,
                "cached": True,
                "outcome": outcome,
                "http_status": 200,
                "attempt_count": 1,
                "error_category": "NONE",
            }

        # 2. Live query if allowed
        if allow_live:
            matched, raw_data, outcome, http_status, retries = self.client.query_pit(
                clean_tk, query_date
            )
            t_resp = time.time()
            if raw_data:
                self.cache_manager.put_atomic(clean_tk, query_date, raw_data)

            self.telemetry.record_request(
                worker_id=self.worker_id,
                spell_id=spell_id,
                ticker=clean_tk,
                query_date=query_date,
                req_ts=t_req,
                resp_ts=t_resp,
                http_status=http_status,
                retry_count=retries,
                is_cache=False,
                is_live=True,
                outcome_category=outcome,
            )
            return matched, {
                "source": "LIVE_API",
                "worker": self.worker_id,
                "cached": False,
                "outcome": outcome,
                "http_status": http_status,
                "attempt_count": retries + 1,
                "error_category": outcome
                if outcome not in ("SUCCESS", "MASSIVE_EMPTY")
                else "NONE",
            }

        # 3. Offline pending
        return None, {
            "source": "OFFLINE",
            "worker": self.worker_id,
            "cached": False,
            "outcome": "OFFLINE_PENDING",
            "http_status": 0,
            "attempt_count": 0,
            "error_category": "OFFLINE_PENDING",
        }

    def process_spell(
        self,
        spell_row: dict[str, Any],
        allow_live: bool = True,
    ) -> dict[str, Any]:
        """Processes an individual spell using the 3-level representative date strategy."""
        ticker = spell_row["ticker"].strip()
        clean_tk = ticker.upper()
        seq = spell_row["spell_seq"]
        s_date = spell_row["start_date"]
        e_date = spell_row["end_date"]
        dur = spell_row.get("n_sessions") or spell_row.get("duration_sessions", 1)
        spell_id = f"{ticker}_{seq}"

        # Level 1: Primary Trading Session Midpoint
        mid_date = self.date_strategy.get_midpoint_session(s_date, e_date)
        rep_date = mid_date
        rep_method = "MIDPOINT_SESSION"

        m_match, telem = self.query_single(
            clean_tk, mid_date, spell_id=spell_id, allow_live=allow_live
        )
        l1_status = telem.get("outcome", "OFFLINE_PENDING")
        l2_s_status = "NOT_ATTEMPTED"
        l2_e_status = "NOT_ATTEMPTED"

        # Level 2: Boundary Fallback
        needs_level2 = l1_status in (
            "MASSIVE_EMPTY",
            "NOT_FOUND",
            "OFFLINE_PENDING",
        ) or (
            l1_status == "SUCCESS"
            and m_match
            and not m_match.get("share_class_figi")
            and not m_match.get("cik")
        )

        level2_start_match = None
        level2_end_match = None

        if needs_level2 and (s_date != mid_date or e_date != mid_date):
            if s_date != mid_date:
                level2_start_match, s_telem = self.query_single(
                    clean_tk, s_date, spell_id=spell_id, allow_live=allow_live
                )
                l2_s_status = s_telem.get("outcome", "OFFLINE_PENDING")
            if e_date != mid_date:
                level2_end_match, e_telem = self.query_single(
                    clean_tk, e_date, spell_id=spell_id, allow_live=allow_live
                )
                l2_e_status = e_telem.get("outcome", "OFFLINE_PENDING")

            if not m_match or l1_status in ("MASSIVE_EMPTY", "OFFLINE_PENDING"):
                if level2_start_match and (
                    level2_start_match.get("cik")
                    or level2_start_match.get("share_class_figi")
                ):
                    m_match = level2_start_match
                    rep_date = s_date
                    rep_method = "BOUNDARY_START_FALLBACK"
                    telem = s_telem
                elif level2_end_match and (
                    level2_end_match.get("cik")
                    or level2_end_match.get("share_class_figi")
                ):
                    m_match = level2_end_match
                    rep_date = e_date
                    rep_method = "BOUNDARY_END_FALLBACK"
                    telem = e_telem

        # Level 3: Drift Detection
        evidence_points = []
        if level2_start_match:
            evidence_points.append(("START", s_date, level2_start_match))
        if m_match and rep_date == mid_date:
            evidence_points.append(("MIDPOINT", mid_date, m_match))
        if level2_end_match:
            evidence_points.append(("END", e_date, level2_end_match))

        drift_detected, drift_details = self.date_strategy.detect_drift(evidence_points)
        if drift_detected:
            self.logger.warning("[DRIFT] %s spell %d: %s", ticker, seq, drift_details)

        m_cik = (
            str(m_match.get("cik")).zfill(10)
            if (m_match and m_match.get("cik"))
            else None
        )
        m_figi = (
            (m_match.get("share_class_figi") or m_match.get("composite_figi"))
            if m_match
            else None
        )
        m_comp = m_match.get("composite_figi") if m_match else None
        m_name = m_match.get("name") if m_match else None
        m_type = m_match.get("type") if m_match else None
        m_exch = m_match.get("primary_exchange") if m_match else None
        m_act = m_match.get("active") if m_match else None

        now_ts = datetime.datetime.now(datetime.UTC).isoformat()

        return {
            "spell_id": spell_id,
            "ticker": ticker,
            "spell_seq": seq,
            "start_date": s_date,
            "end_date": e_date,
            "duration_sessions": dur,
            "representative_date": rep_date,
            "representative_date_method": rep_method,
            "lookup_status": telem.get("outcome", "OFFLINE_PENDING"),
            "cache_status": "HIT"
            if telem.get("cached")
            else ("MISS" if telem.get("source") == "LIVE_API" else "NONE"),
            "attempt_count": telem.get("attempt_count", 0),
            "worker_slot": self.worker_id,
            "completion_timestamp": now_ts,
            "error_category": telem.get("error_category", "NONE"),
            "massive_cik": m_cik,
            "massive_figi": m_figi,
            "massive_composite_figi": m_comp,
            "massive_name": m_name,
            "massive_type": m_type,
            "massive_exchange": m_exch,
            "massive_active": m_act,
            "level1_status": l1_status,
            "level2_start_status": l2_s_status,
            "level2_end_status": l2_e_status,
            "drift_detected": drift_detected,
            "drift_details": drift_details,
        }
