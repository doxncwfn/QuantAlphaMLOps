"""
Thread-Safe Telemetry and Execution Metrics.
============================================
Tracks per-request and per-worker performance, latency percentiles,
and guarantees exact reconciliation (total = cache_hits + live_requests).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl


class WorkerTelemetry:
    """Thread-safe telemetry collector for multi-worker API execution."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._lock = threading.Lock()
        self.logger = logger or logging.getLogger("telemetry")
        self.records: List[Dict[str, Any]] = []
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        self.global_cache_hits = 0
        self.global_live_requests = 0

    def init_worker(self, worker_id: str) -> None:
        with self._lock:
            if worker_id not in self.worker_stats:
                self.worker_stats[worker_id] = {
                    "worker_id": worker_id,
                    "total_requests": 0,
                    "cache_hits": 0,
                    "live_requests": 0,
                    "successes": 0,
                    "empty_responses": 0,
                    "rate_limits_429": 0,
                    "errors": 0,
                    "retries": 0,
                    "total_latency_ms": 0.0,
                    "latencies_ms": [],
                }

    def record_request(
        self,
        worker_id: str,
        spell_id: str,
        ticker: str,
        query_date: str,
        req_ts: float,
        resp_ts: float,
        http_status: int,
        retry_count: int,
        is_cache: bool,
        is_live: bool,
        outcome_category: str,
    ) -> None:
        latency_ms = max(0.0, (resp_ts - req_ts) * 1000.0)
        with self._lock:
            if is_cache:
                self.global_cache_hits += 1
            if is_live:
                self.global_live_requests += 1

            if worker_id not in self.worker_stats:
                self.worker_stats[worker_id] = {
                    "worker_id": worker_id,
                    "total_requests": 0,
                    "cache_hits": 0,
                    "live_requests": 0,
                    "successes": 0,
                    "empty_responses": 0,
                    "rate_limits_429": 0,
                    "errors": 0,
                    "retries": 0,
                    "total_latency_ms": 0.0,
                    "latencies_ms": [],
                }

            w = self.worker_stats[worker_id]
            w["total_requests"] += 1
            if is_cache:
                w["cache_hits"] += 1
            if is_live:
                w["live_requests"] += 1
                w["latencies_ms"].append(latency_ms)
                w["total_latency_ms"] += latency_ms

            if outcome_category == "SUCCESS":
                w["successes"] += 1
            elif outcome_category == "MASSIVE_EMPTY":
                w["empty_responses"] += 1
            elif outcome_category == "RATE_LIMITED":
                w["rate_limits_429"] += 1
            else:
                w["errors"] += 1
            w["retries"] += retry_count

            self.records.append({
                "worker_id": worker_id,
                "spell_id": spell_id,
                "ticker": ticker,
                "query_date": query_date,
                "request_timestamp": req_ts,
                "response_timestamp": resp_ts,
                "latency_ms": latency_ms,
                "latency_sec": latency_ms / 1000.0,
                "http_status": http_status,
                "retry_count": retry_count,
                "is_cache": is_cache,
                "is_live": is_live,
                "outcome_category": outcome_category,
            })

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            all_live_latencies = []
            for st in self.worker_stats.values():
                all_live_latencies.extend(st["latencies_ms"])
            all_live_latencies.sort()

            n = len(all_live_latencies)
            median_lat = all_live_latencies[n // 2] if n > 0 else 0.0
            p90_lat = all_live_latencies[int(n * 0.90)] if n > 0 else 0.0
            p95_lat = all_live_latencies[int(n * 0.95)] if n > 0 else 0.0
            p99_lat = all_live_latencies[int(n * 0.99)] if n > 0 else 0.0

            return {
                "total_records": len(self.records),
                "total_requests": len(self.records),
                "global_cache_hits": self.global_cache_hits,
                "global_live_requests": self.global_live_requests,
                "workers": list(self.worker_stats.values()),
                "all_live_latencies_count": n,
                "median_latency_ms": median_lat,
                "p90_latency_ms": p90_lat,
                "p95_latency_ms": p95_lat,
                "p99_latency_ms": p99_lat,
            }

    def to_dataframe(self) -> pl.DataFrame:
        with self._lock:
            if not self.records:
                return pl.DataFrame()
            return pl.DataFrame(self.records)
