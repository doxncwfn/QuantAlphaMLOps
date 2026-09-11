"""Unit tests for PerKeyRateLimiter."""

import time
import unittest

from src.common.rate_limiter import PerKeyRateLimiter


class TestRateLimiter(unittest.TestCase):
    def test_pacing_wait(self):
        limiter = PerKeyRateLimiter(min_interval_seconds=0.05)
        t0 = time.time()
        limiter.wait()
        limiter.wait()
        elapsed = time.time() - t0
        self.assertGreaterEqual(elapsed, 0.045)

    def test_backoff_handling(self):
        limiter = PerKeyRateLimiter(
            initial_backoff_seconds=0.01, max_backoff_seconds=0.1
        )
        dur1 = limiter.handle_rate_limit()
        self.assertGreater(dur1, 0.0)

        limiter.reset_backoff()
        self.assertEqual(limiter._current_backoff, 0.01)

    def test_retry_after_header(self):
        limiter = PerKeyRateLimiter()
        dur = limiter.handle_rate_limit(retry_after=0.01)
        self.assertEqual(dur, 0.01)


if __name__ == "__main__":
    unittest.main()
