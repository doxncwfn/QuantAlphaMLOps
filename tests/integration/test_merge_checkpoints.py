"""Integration tests for merge_checkpoints script."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl


class TestMergeCheckpointsIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="merge_chk_test_"))
        self.chk_dir = self.test_dir / "checkpoints"
        self.chk_dir.mkdir()
        self.out_file = self.test_dir / "merged.parquet"

        # Create 2 dummy chunks
        df1 = pl.DataFrame(
            {"spell_id": ["A_1", "B_1"], "ticker": ["A", "B"], "spell_seq": [1, 1]}
        )
        df2 = pl.DataFrame(
            {"spell_id": ["B_1", "C_1"], "ticker": ["B", "C"], "spell_seq": [1, 1]}
        )
        df1.write_parquet(self.chk_dir / "checkpoint_00000.parquet")
        df2.write_parquet(self.chk_dir / "checkpoint_00001.parquet")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_merge_checkpoints_cli(self):
        cmd = [
            sys.executable,
            "scripts/merge_checkpoints.py",
            "--checkpoints-dir",
            str(self.chk_dir),
            "--output",
            str(self.out_file),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Script failed: {res.stderr}")
        self.assertTrue(self.out_file.exists())

        df_out = pl.read_parquet(self.out_file)
        self.assertEqual(df_out.height, 3)
        self.assertListEqual(
            sorted(df_out["spell_id"].to_list()), ["A_1", "B_1", "C_1"]
        )


if __name__ == "__main__":
    unittest.main()
