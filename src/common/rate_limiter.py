"""
Per-Key Rate Limiter with Backoff and Jitter.
============================================
Enforces polite request pacing per worker thread without serializing across
distinct API key slots.
"""

from __future__ import annotations

import logging
import random
import threading
import time


class PerKeyRateLimiter:
    """Thread-safe rate limiter managing the minimum interval between requests for a single key slot."""

    def __init__(
        self,
        min_interval_seconds: float = 12.1,
        initial_backoff_seconds: float = 2.0,
        max_backoff_seconds: float = 60.0,
        logger: logging.Logger | None = None,
    ):
        self.min_interval = min_interval_seconds
        self.initial_backoff = initial_backoff_seconds
        self.max_backoff = max_backoff_seconds
        self.logger = logger or logging.getLogger("rate_limiter")
        self._lock = threading.Lock()
        self._last_call_time = 0.0
        self._current_backoff = initial_backoff_seconds

    def wait(self) -> float:
        """Paces execution to ensure min_interval elapsed since previous call. Returns wait duration."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            sleep_time = max(0.0, self.min_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._last_call_time = time.time()
            return sleep_time

    def handle_rate_limit(self, retry_after: float | None = None) -> float:
        """Applies exponential backoff with full jitter when HTTP 429 occurs. Returns wait duration."""
        with self._lock:
            if retry_after is not None and retry_after > 0:
                wait_duration = retry_after
            else:
                jitter = random.uniform(0.5, 1.5)
                wait_duration = min(self.max_backoff, self._current_backoff * jitter)
                self._current_backoff = min(
                    self.max_backoff, self._current_backoff * 2.0
                )

            self.logger.warning(
                "Rate limit encountered. Backing off for %.2fs...", wait_duration
            )
            time.sleep(wait_duration)
            self._last_call_time = time.time()
            return wait_duration

    def reset_backoff(self) -> None:
        """Resets exponential backoff after a successful response."""
        with self._lock:
            self._current_backoff = self.initial_backoff
