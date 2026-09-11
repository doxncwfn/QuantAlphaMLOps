"""Unit tests for MassiveClient."""

import unittest
from unittest.mock import MagicMock, patch

from src.common.rate_limiter import PerKeyRateLimiter
from src.identity.massive.client import MassiveClient


class TestMassiveClient(unittest.TestCase):
    def setUp(self):
        self.rate_limiter = PerKeyRateLimiter(
            min_interval_seconds=0.001, initial_backoff_seconds=0.001
        )
        self.client = MassiveClient(
            api_key="TEST_API_KEY",
            rate_limiter=self.rate_limiter,
            timeout_seconds=1.0,
            max_retries=2,
        )

    def test_extract_match_exact(self):
        data = {
            "results": [
                {"ticker": "AAPL", "cik": "0000320193", "name": "Apple Inc."},
                {"ticker": "AAPLW", "cik": "0000320193", "name": "Apple Warrant"},
            ]
        }
        match = MassiveClient.extract_match(data, "AAPL")
        self.assertIsNotNone(match)
        self.assertEqual(match["ticker"], "AAPL")

    def test_extract_match_dot_notation(self):
        data = {
            "results": [
                {
                    "ticker": "CMCSA",
                    "cik": "0001234567",
                    "name": "Comcast Corp Class A",
                },
            ]
        }
        match = MassiveClient.extract_match(data, "CMCS.A")
        self.assertIsNotNone(match)
        self.assertEqual(match["ticker"], "CMCSA")

    def test_extract_match_empty_or_none(self):
        self.assertIsNone(MassiveClient.extract_match(None, "AAPL"))
        self.assertIsNone(MassiveClient.extract_match({}, "AAPL"))
        self.assertIsNone(MassiveClient.extract_match({"results": []}, "AAPL"))

    @patch("requests.Session.get")
    def test_query_pit_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {
                    "ticker": "AAPL",
                    "cik": "0000320193",
                    "share_class_figi": "BBG001S5N8V8",
                }
            ]
        }
        mock_get.return_value = mock_resp

        match, raw, outcome, status, retries = self.client.query_pit(
            "AAPL", "2020-01-02"
        )
        self.assertEqual(outcome, "SUCCESS")
        self.assertEqual(status, 200)
        self.assertEqual(retries, 0)
        self.assertEqual(match["cik"], "0000320193")

    @patch("requests.Session.get")
    def test_query_pit_massive_empty(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        mock_get.return_value = mock_resp

        match, raw, outcome, status, retries = self.client.query_pit(
            "EMPTY_TICKER", "2020-01-02"
        )
        self.assertEqual(outcome, "MASSIVE_EMPTY")
        self.assertEqual(status, 200)

    @patch("requests.Session.get")
    def test_query_pit_404(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        match, raw, outcome, status, retries = self.client.query_pit(
            "UNKNOWN", "2020-01-02"
        )
        self.assertEqual(outcome, "NOT_FOUND")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
