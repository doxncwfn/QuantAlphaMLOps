"""Unit tests for CheckpointManager."""

import shutil
import tempfile
import unittest
from pathlib import Path

import polars as pl

from src.identity.massive.checkpoint import CheckpointManager


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="checkpoint_test_"))
        self.chk_dir = self.test_dir / "checkpoints"
        self.master_file = self.test_dir / "master.parquet"
        self.mgr = CheckpointManager(checkpoints_dir=self.chk_dir, master_manifest_path=self.master_file)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_save_and_load_checkpoints(self):
        records = [
            {
                "spell_id": "AAPL_1",
                "ticker": "AAPL",
                "spell_seq": 1,
                "start_date": "2020-01-02",
                "end_date": "2020-01-10",
                "duration_sessions": 6,
                "representative_date": "2020-01-06",
                "representative_date_method": "MIDPOINT_SESSION",
                "lookup_status": "SUCCESS",
                "cache_status": "HIT",
                "attempt_count": 1,
                "worker_slot": "WORKER_1",
                "completion_timestamp": "2026-09-11T00:00:00Z",
                "error_category": "NONE",
                "massive_cik": "0000320193",
                "massive_figi": "BBG001S5N8V8",
                "massive_composite_figi": "BBG000B9XRY4",
                "massive_name": "Apple Inc.",
                "massive_type": "CS",
                "massive_exchange": "XNAS",
                "massive_active": True,
                "level1_status": "SUCCESS",
                "level2_start_status": "NOT_ATTEMPTED",
                "level2_end_status": "NOT_ATTEMPTED",
                "drift_detected": False,
                "drift_details": "",
            }
        ]
        self.mgr.save_checkpoint(records, 0)
        completed = self.mgr.load_completed_spells()
        self.assertIn("AAPL_1", completed)

    def test_merge_checkpoints(self):
        def make_rec(sp_id, seq):
            return {
                "spell_id": sp_id, "ticker": "TEST", "spell_seq": seq,
                "start_date": "2020-01-02", "end_date": "2020-01-10", "duration_sessions": 6,
                "representative_date": "2020-01-06", "representative_date_method": "MIDPOINT_SESSION",
                "lookup_status": "SUCCESS", "cache_status": "HIT", "attempt_count": 1,
                "worker_slot": "WORKER_1", "completion_timestamp": "2026-09-11T00:00:00Z",
                "error_category": "NONE", "massive_cik": None, "massive_figi": None,
                "massive_composite_figi": None, "massive_name": None, "massive_type": None,
                "massive_exchange": None, "massive_active": True, "level1_status": "SUCCESS",
                "level2_start_status": "NOT_ATTEMPTED", "level2_end_status": "NOT_ATTEMPTED",
                "drift_detected": False, "drift_details": "",
            }

        self.mgr.save_checkpoint([make_rec("TEST_1", 1)], 0)
        self.mgr.save_checkpoint([make_rec("TEST_2", 2)], 1)

        merged = self.mgr.merge_all_checkpoints()
        self.assertEqual(merged.height, 2)
        self.assertTrue(self.master_file.exists())


if __name__ == "__main__":
    unittest.main()
