"""Unit tests for resolver domain models and ID generators."""

import unittest

from src.identity.resolver.model import (
    SecurityType,
    UniverseStatus,
    classify_universe_status,
    make_deterministic_unresolved_id,
    make_provisional_cik_id,
)


class TestResolverModel(unittest.TestCase):
    def test_provisional_cik_id(self):
        prov_id = make_provisional_cik_id("0000320193", "AAPL", "2020-01-02")
        self.assertTrue(prov_id.startswith("PROVISIONAL_CIK_0000320193_AAPL_"))

    def test_deterministic_unresolved_id(self):
        id1 = make_deterministic_unresolved_id("AAPL", 1, "2020-01-02")
        id2 = make_deterministic_unresolved_id("AAPL", 2, "2020-05-01")
        self.assertNotEqual(id1, id2)
        self.assertTrue(id1.startswith("UNRESOLVED_AAPL_1_"))
        self.assertTrue(id2.startswith("UNRESOLVED_AAPL_2_"))

    def test_classify_universe_status(self):
        self.assertEqual(
            classify_universe_status(SecurityType.COMMON_STOCK), UniverseStatus.INCLUDE
        )
        self.assertEqual(
            classify_universe_status(SecurityType.ETF), UniverseStatus.EXCLUDE
        )
        self.assertEqual(
            classify_universe_status(SecurityType.UNKNOWN), UniverseStatus.QUARANTINE
        )


if __name__ == "__main__":
    unittest.main()
