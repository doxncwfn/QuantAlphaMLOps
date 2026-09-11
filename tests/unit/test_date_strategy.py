"""Unit tests for RepresentativeDateStrategy."""

import unittest

from src.identity.massive.date_strategy import RepresentativeDateStrategy


class TestDateStrategy(unittest.TestCase):
    def setUp(self):
        self.strat = RepresentativeDateStrategy()
        # Seed dummy sessions for deterministic testing
        self.strat.all_sessions = [
            "2020-01-02",
            "2020-01-03",
            "2020-01-06",
            "2020-01-07",
            "2020-01-08",
        ]
        self.strat.session_to_idx = {
            d: i for i, d in enumerate(self.strat.all_sessions)
        }

    def test_midpoint_calculation(self):
        mid = self.strat.get_midpoint_session("2020-01-02", "2020-01-08")
        self.assertEqual(mid, "2020-01-06")

    def test_drift_detection_no_drift(self):
        evidence = [
            (
                "START",
                "2020-01-02",
                {"cik": "0000320193", "share_class_figi": "BBG001S5N8V8"},
            ),
            (
                "END",
                "2020-01-08",
                {"cik": "0000320193", "share_class_figi": "BBG001S5N8V8"},
            ),
        ]
        drift, details = self.strat.detect_drift(evidence)
        self.assertFalse(drift)
        self.assertEqual(details, "")

    def test_drift_detection_with_drift(self):
        evidence = [
            (
                "START",
                "2020-01-02",
                {"cik": "0000111111", "share_class_figi": "FIGI_A"},
            ),
            ("END", "2020-01-08", {"cik": "0000222222", "share_class_figi": "FIGI_B"}),
        ]
        drift, details = self.strat.detect_drift(evidence)
        self.assertTrue(drift)
        self.assertIn("Boundary divergence", details)


if __name__ == "__main__":
    unittest.main()
