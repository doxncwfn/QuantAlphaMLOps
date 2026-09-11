"""
Massive Reference API Client.
=============================
Pure HTTP client handling point-in-time ticker queries, response parsing,
retry pacing, and strict outcome classification.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.common.config import (
    MASSIVE_REFERENCE_URL,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
)
from src.common.rate_limiter import PerKeyRateLimiter


class MassiveClient:
    """HTTP Client for querying Massive Point-in-Time Reference API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = MASSIVE_REFERENCE_URL,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
        rate_limiter: Optional[PerKeyRateLimiter] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.rate_limiter = rate_limiter or PerKeyRateLimiter()
        self.logger = logger or logging.getLogger("massive_client")
        self.session = requests.Session()

    def query_pit(
        self,
        ticker: str,
        date_str: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str, int, int]:
        """Queries Massive for ticker on date_str.

        Returns:
            (matched_record, raw_response_json, outcome_category, http_status, retries_used)
        """
        clean_tk = ticker.strip().upper()
        params = {
            "ticker": clean_tk,
            "date": date_str,
            "market": "stocks",
            "active": "true",
            "limit": 10,
            "apiKey": self.api_key,
        }

        retries = 0
        while retries <= self.max_retries:
            self.rate_limiter.wait()
            try:
                resp = self.session.get(self.base_url, params=params, timeout=self.timeout)
                http_status = resp.status_code

                if http_status == 200:
                    self.rate_limiter.reset_backoff()
                    try:
                        data = resp.json()
                    except Exception:
                        return None, None, "INVALID_RESPONSE", 200, retries

                    matched = self.extract_match(data, clean_tk)
                    if matched and (matched.get("cik") or matched.get("share_class_figi") or matched.get("composite_figi")):
                        return matched, data, "SUCCESS", 200, retries
                    return matched, data, "MASSIVE_EMPTY", 200, retries

                elif http_status == 429:
                    retry_after = None
                    if "Retry-After" in resp.headers:
                        try:
                            retry_after = float(resp.headers["Retry-After"])
                        except ValueError:
                            pass
                    self.rate_limiter.handle_rate_limit(retry_after)
                    retries += 1
                    continue

                elif http_status == 404:
                    return None, None, "NOT_FOUND", 404, retries

                elif 500 <= http_status < 600:
                    self.logger.warning("HTTP %d from Massive API. Retrying (%d/%d)...", http_status, retries + 1, self.max_retries)
                    self.rate_limiter.handle_rate_limit()
                    retries += 1
                    continue

                else:
                    return None, None, f"HTTP_{http_status}", http_status, retries

            except requests.exceptions.Timeout:
                self.logger.warning("Request timeout for %s:%s. Retrying (%d/%d)...", clean_tk, date_str, retries + 1, self.max_retries)
                self.rate_limiter.handle_rate_limit()
                retries += 1
                if retries > self.max_retries:
                    return None, None, "TIMEOUT", 0, retries

            except requests.exceptions.RequestException as e:
                self.logger.warning("Network error for %s:%s: %s. Retrying (%d/%d)...", clean_tk, date_str, e, retries + 1, self.max_retries)
                self.rate_limiter.handle_rate_limit()
                retries += 1
                if retries > self.max_retries:
                    return None, None, "NETWORK_FAILURE", 0, retries

        return None, None, "RATE_LIMITED", 429, retries

    @staticmethod
    def extract_match(data: Optional[Dict[str, Any]], query_ticker: str) -> Optional[Dict[str, Any]]:
        """Finds the most specific matching record from Massive's results array."""
        if not data or not isinstance(data, dict):
            return None
        results = data.get("results")
        if not results or not isinstance(results, list):
            return None

        clean_q = query_ticker.strip().upper()
        # 1. Exact ticker match
        for r in results:
            if isinstance(r, dict) and r.get("ticker", "").strip().upper() == clean_q:
                return r

        # 2. Match without dot / with dot (e.g. BRK.A vs BRKA)
        alt_q = clean_q.replace(".", "") if "." in clean_q else clean_q.replace("/", "")
        for r in results:
            if isinstance(r, dict):
                rtk = r.get("ticker", "").strip().upper().replace(".", "").replace("/", "")
                if rtk == alt_q:
                    return r

        # 3. Fallback to first result
        first = results[0]
        return first if isinstance(first, dict) else None
