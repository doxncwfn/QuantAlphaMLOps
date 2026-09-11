"""Massive reference client for selective point-in-time and current queries."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from src.common.config import (
    MASSIVE_API_KEY,
    MASSIVE_CACHE_DIR,
    MASSIVE_RATE_DELAY_SECONDS,
    MASSIVE_REFERENCE_URL,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class MassiveReferenceClient:
    def __init__(self, cache_dir: Path = MASSIVE_CACHE_DIR, api_key: str = MASSIVE_API_KEY):
        self.cache_dir = cache_dir
        self.api_key = api_key
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def query_ticker_pit(self, ticker: str, target_date: str) -> Optional[Dict[str, Any]]:
        """Queries point-in-time ticker reference on target_date with persistent caching."""
        tk_clean = ticker.strip().upper()
        cache_file = self.cache_dir / f"{tk_clean}_{target_date}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        params = {
            "ticker": tk_clean,
            "date": target_date,
            "apiKey": self.api_key
        }

        try:
            logger.info("Querying Massive PIT reference for '%s' on %s...", tk_clean, target_date)
            time.sleep(MASSIVE_RATE_DELAY_SECONDS)
            resp = self.session.get(MASSIVE_REFERENCE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            
            if resp.status_code == 429:
                logger.warning("Massive API 429 rate limit. Waiting 30s...")
                time.sleep(30.0)
                resp = self.session.get(MASSIVE_REFERENCE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)

            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            
            record = None
            for item in results:
                if isinstance(item, dict) and item.get("ticker", "").upper() == tk_clean:
                    record = item
                    break

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(record, f)

            return record

        except Exception as exc:
            logger.warning("Error querying Massive PIT reference for '%s' on %s: %s", tk_clean, target_date, exc)
            return None

    def query_ticker_current(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Queries current reference for a ticker with caching."""
        tk_clean = ticker.strip().upper()
        cache_file = self.cache_dir / f"{tk_clean}_current.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        params = {
            "ticker": tk_clean,
            "apiKey": self.api_key
        }

        try:
            logger.info("Querying Massive current reference for '%s'...", tk_clean)
            time.sleep(MASSIVE_RATE_DELAY_SECONDS)
            resp = self.session.get(MASSIVE_REFERENCE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            
            if resp.status_code == 429:
                logger.warning("Massive API 429 rate limit. Waiting 30s...")
                time.sleep(30.0)
                resp = self.session.get(MASSIVE_REFERENCE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)

            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            
            record = None
            for item in results:
                if isinstance(item, dict) and item.get("ticker", "").upper() == tk_clean:
                    record = item
                    break

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(record, f)

            return record

        except Exception as exc:
            logger.warning("Error querying Massive current reference for '%s': %s", tk_clean, exc)
            return None
