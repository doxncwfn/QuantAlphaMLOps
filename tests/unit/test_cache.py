"""Unit tests for CacheManager."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.identity.massive.cache import CacheManager


class TestCacheManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="cache_test_"))
        self.cache_mgr = CacheManager(primary_cache_dir=self.test_dir)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_put_atomic_and_get(self):
        payload = {
            "ticker": "AAPL",
            "results": [{"ticker": "AAPL", "cik": "0000320193"}],
        }
        saved_path = self.cache_mgr.put_atomic("AAPL", "2020-01-02", payload)

        self.assertTrue(saved_path.exists())
        self.assertEqual(len(list(self.test_dir.glob("*.tmp"))), 0)

        match, raw, hit = self.cache_mgr.get("AAPL", "2020-01-02")
        self.assertTrue(hit)
        self.assertEqual(match["cik"], "0000320193")

    def test_index_caches(self):
        sub_cache = self.test_dir / "sub"
        sub_cache.mkdir()
        file_path = sub_cache / "MSFT_2021-05-10.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"results": [{"ticker": "MSFT", "cik": "0000789019"}]}, f)

        mgr = CacheManager(primary_cache_dir=self.test_dir, search_dirs=[sub_cache])
        count = mgr.index_caches()
        self.assertEqual(count, 1)

        match, raw, hit = mgr.get("MSFT", "2021-05-10")
        self.assertTrue(hit)
        self.assertEqual(match["cik"], "0000789019")


if __name__ == "__main__":
    unittest.main()
