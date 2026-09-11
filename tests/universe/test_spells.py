"""Unit and regression tests for universe spells integrity."""

import hashlib
import unittest

import polars as pl

from src.common.config import (
    EXPECTED_SPELLS_HASH,
    EXPECTED_SPELLS_ROWS,
    EXPECTED_UNIQUE_TICKERS,
    SPELLS_CSV_PATH,
)


class TestSpellsIntegrity(unittest.TestCase):
    def test_spells_file_exists(self):
        self.assertTrue(SPELLS_CSV_PATH.exists(), f"Spells file not found: {SPELLS_CSV_PATH}")

    def test_spells_sha256_immutable(self):
        actual_hash = hashlib.sha256(SPELLS_CSV_PATH.read_bytes()).hexdigest()
        self.assertEqual(
            actual_hash,
            EXPECTED_SPELLS_HASH,
            f"Spells file hash changed! Expected {EXPECTED_SPELLS_HASH}, got {actual_hash}",
        )

    def test_spells_row_and_ticker_count(self):
        df = pl.read_csv(SPELLS_CSV_PATH)
        self.assertEqual(df.height, EXPECTED_SPELLS_ROWS)
        unique_tickers = df["ticker"].n_unique()
        self.assertEqual(unique_tickers, EXPECTED_UNIQUE_TICKERS)


if __name__ == "__main__":
    unittest.main()
