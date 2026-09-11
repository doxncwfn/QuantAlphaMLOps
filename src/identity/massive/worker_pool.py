"""
Concurrent 9-Key Worker Pool Orchestrator.
==========================================
Manages a pool of worker threads pinned 1:1 to API key slots, ensuring
key isolation, polite rate pacing, and balanced queue dispatch.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from src.common.config import (
    ENV_PATH,
    MASSIVE_V3_CACHE_DIR,
    PER_KEY_INTERVAL_SECONDS,
)
from src.identity.massive.cache import CacheManager
from src.identity.massive.client import MassiveClient
from src.identity.massive.date_strategy import RepresentativeDateStrategy
from src.identity.massive.telemetry import WorkerTelemetry
from src.identity.massive.worker import MassiveWorker


class ConcurrentKeyWorkerPool:
    """Orchestrates 9 concurrent worker channels with strict key isolation."""

    def __init__(
        self,
        min_per_key_interval: float = PER_KEY_INTERVAL_SECONDS,
        cache_dir: Path | None = None,
        api_keys: list[str] | None = None,
        sessions_path: Path | None = None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or logging.getLogger("worker_pool")
        self.primary_cache_dir = cache_dir or MASSIVE_V3_CACHE_DIR
        self.primary_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_manager = CacheManager(
            primary_cache_dir=self.primary_cache_dir, logger=self.logger
        )
        self.telemetry = WorkerTelemetry(logger=self.logger)
        self.date_strategy = RepresentativeDateStrategy(
            sessions_path=sessions_path, logger=self.logger
        )
        self.min_interval = min_per_key_interval

        # Discover API keys
        discovered_keys = api_keys if api_keys is not None else self._discover_keys()
        if not discovered_keys:
            self.logger.info(
                "Initialized ConcurrentKeyWorkerPool with 0 live workers (offline mode)."
            )
            discovered_keys = []

        # Bind 1:1 worker slots
        self.workers: list[MassiveWorker] = []
        for idx, key in enumerate(discovered_keys):
            wid = f"WORKER_{idx + 1}"
            worker = MassiveWorker(
                worker_id=wid,
                api_key=key,
                cache_manager=self.cache_manager,
                telemetry=self.telemetry,
                date_strategy=self.date_strategy,
                min_interval_seconds=self.min_interval,
                logger=self.logger,
            )
            self.workers.append(worker)

        self.logger.info(
            "Initialized ConcurrentKeyWorkerPool with %d workers (interval=%.2fs).",
            len(self.workers),
            self.min_interval,
        )

    def _discover_keys(self) -> list[str]:
        """Discovers up to 9 API keys from os.environ or .env."""
        env_vars = {}
        if ENV_PATH.exists():
            env_vars = dotenv_values(ENV_PATH)

        keys: list[str] = []
        for i in range(1, 10):
            var_name = f"MASSIVE_API_KEY_{i}"
            val = os.environ.get(var_name) or env_vars.get(var_name)
            if val and not val.strip().lower().startswith("your_"):
                keys.append(val.strip())

        if not keys:
            single = os.environ.get("MASSIVE_API_KEY") or env_vars.get(
                "MASSIVE_API_KEY"
            )
            if single and not single.strip().lower().startswith("your_"):
                keys.append(single.strip())

        return keys

    def _atomic_write_cache(self, cache_file: Path, data: Any):
        """Atomically writes data to cache_file via temp file and rename."""
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=cache_file.parent, delete=False, encoding="utf-8"
        ) as tf:
            json.dump(data, tf)
            tf.flush()
            os.fsync(tf.fileno())
            temp_path = Path(tf.name)
        os.replace(temp_path, cache_file)

    def _extract_match(self, data: Any, clean_tk: str) -> dict[str, Any] | None:
        """Extracts exact matching ticker dictionary from cached JSON data."""
        return MassiveClient.extract_match(data, clean_tk)

    def query(
        self,
        ticker: str,
        query_date: str,
        spell_id: str = "",
        preferred_worker_idx: int | None = None,
        allow_live: bool = True,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Dispatches a query to a worker slot."""
        if not self.workers:
            # Check cache directly in offline mode
            clean_tk = ticker.strip().upper()
            m, raw, hit = self.cache_manager.get(clean_tk, query_date)
            outcome = (
                "SUCCESS"
                if (m and (m.get("cik") or m.get("share_class_figi")))
                else ("MASSIVE_EMPTY" if hit else "OFFLINE_PENDING")
            )
            return m, {
                "source": "CACHE" if hit else "OFFLINE",
                "worker": "WORKER_1",
                "cached": hit,
                "outcome": outcome,
            }

        w_idx = (
            (preferred_worker_idx % len(self.workers))
            if preferred_worker_idx is not None
            else 0
        )
        worker = self.workers[w_idx]
        return worker.query_single(
            ticker, query_date, spell_id=spell_id, allow_live=allow_live
        )

    def process_spell(
        self,
        spell_row: dict[str, Any],
        worker_idx: int,
        allow_live: bool = True,
    ) -> dict[str, Any]:
        """Processes a spell using the designated worker channel."""
        if not self.workers:
            # Dummy worker in offline mode
            worker = MassiveWorker(
                "WORKER_1",
                "OFFLINE_KEY",
                self.cache_manager,
                self.telemetry,
                self.date_strategy,
            )
            return worker.process_spell(spell_row, allow_live=False)

        w_idx = worker_idx % len(self.workers)
        return self.workers[w_idx].process_spell(spell_row, allow_live=allow_live)
