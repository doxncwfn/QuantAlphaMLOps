"""
Point-in-Time Cache Management & Indexing.
==========================================
Manages multi-directory indexing of cached JSON responses and ensures
atomic writes via NamedTemporaryFile + fsync + os.replace.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from src.common.config import ALL_CACHE_DIRS, MASSIVE_V3_CACHE_DIR
from src.identity.massive.client import MassiveClient


class CacheManager:
    """Thread-safe manager for caching Massive PIT responses with atomic file replacement."""

    def __init__(
        self,
        primary_cache_dir: Path | None = None,
        search_dirs: list[Path] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.primary_cache_dir = primary_cache_dir or MASSIVE_V3_CACHE_DIR
        self.search_dirs = search_dirs or ALL_CACHE_DIRS
        self.logger = logger or logging.getLogger("cache_manager")
        self._lock = threading.Lock()
        self.cache_index: dict[str, dict[str, Any]] = {}
        self.primary_cache_dir.mkdir(parents=True, exist_ok=True)

    def index_caches(self) -> int:
        """Indexes all existing cache files into memory for O(1) retrieval."""
        with self._lock:
            count = 0
            for cdir in self.search_dirs:
                if not cdir.exists():
                    continue
                for f in cdir.rglob("*.json"):
                    parts = f.stem.split("_")
                    if len(parts) >= 2:
                        tk = parts[0].upper()
                        dt = parts[1]
                        key = f"{tk}:{dt}"
                        if key not in self.cache_index:
                            try:
                                with open(f, encoding="utf-8") as fp:
                                    data = json.load(fp)
                                matched = MassiveClient.extract_match(data, tk)
                                self.cache_index[key] = {
                                    "matched": matched,
                                    "raw": data,
                                    "source_file": str(f),
                                }
                                count += 1
                            except (
                                OSError,
                                json.JSONDecodeError,
                                ValueError,
                                KeyError,
                            ) as err:
                                self.logger.debug(
                                    "Failed to read cache file %s: %s", f, err
                                )
            self.logger.info(
                "Indexed %d unique ticker:date entries across cache directories.", count
            )
            return count

    def get(
        self, ticker: str, date_str: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
        """Looks up a ticker:date in memory index.

        Returns:
            (matched_record, raw_json, is_hit)
        """
        clean_tk = ticker.strip().upper()
        key = f"{clean_tk}:{date_str}"
        with self._lock:
            entry = self.cache_index.get(key)
            if entry is not None:
                return entry["matched"], entry["raw"], True

            # Check direct file path in primary cache dir if not indexed
            target_file = self.primary_cache_dir / f"{clean_tk}_{date_str}.json"
            if target_file.exists():
                try:
                    with open(target_file, encoding="utf-8") as fp:
                        data = json.load(fp)
                    matched = MassiveClient.extract_match(data, clean_tk)
                    self.cache_index[key] = {
                        "matched": matched,
                        "raw": data,
                        "source_file": str(target_file),
                    }
                    return matched, data, True
                except (OSError, json.JSONDecodeError, ValueError, KeyError) as err:
                    self.logger.debug(
                        "Failed to read cache file %s: %s", target_file, err
                    )

        return None, None, False

    def put_atomic(self, ticker: str, date_str: str, data: dict[str, Any]) -> Path:
        """Atomically saves data to JSON cache and updates in-memory index."""
        clean_tk = ticker.strip().upper()
        target_path = self.primary_cache_dir / f"{clean_tk}_{date_str}.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target_path.parent,
                prefix=f".{target_path.stem}_",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                tmp_name = tmp_file.name
                json.dump(data, tmp_file, indent=2, ensure_ascii=False)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_name, target_path)
        except Exception:
            if tmp_name and os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise

        matched = MassiveClient.extract_match(data, clean_tk)
        key = f"{clean_tk}:{date_str}"
        with self._lock:
            self.cache_index[key] = {
                "matched": matched,
                "raw": data,
                "source_file": str(target_path),
            }
        return target_path
