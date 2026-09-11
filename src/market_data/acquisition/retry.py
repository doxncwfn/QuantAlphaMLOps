"""Resilience, retry policies, and circuit breaker."""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    allowed_exceptions: tuple = (Exception,)
) -> T:
    """Executes func with exponential backoff on allowed exceptions."""
    delay = initial_delay
    last_exc = None

    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except allowed_exceptions as exc:
            last_exc = exc
            if attempt == max_retries:
                logger.warning("Failed after %d attempts: %s", max_retries, exc)
                raise exc
            logger.debug("Attempt %d failed (%s). Retrying in %.2fs...", attempt, exc, delay)
            time.sleep(delay)
            delay *= backoff_factor

    raise last_exc
