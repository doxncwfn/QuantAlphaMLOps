"""
Checkpoint and Manifest Management.
==================================
Handles periodic Parquet checkpoint serialization, resume loading,
and master manifest concatenation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from src.common.config import MANIFESTS_DIR
from src.common.io import atomic_write_parquet

MANIFEST_SCHEMA = {
    "spell_id": pl.Utf8,
    "ticker": pl.Utf8,
    "spell_seq": pl.Int64,
    "start_date": pl.Utf8,
    "end_date": pl.Utf8,
    "duration_sessions": pl.Int64,
    "representative_date": pl.Utf8,
    "representative_date_method": pl.Utf8,
    "lookup_status": pl.Utf8,
    "cache_status": pl.Utf8,
    "attempt_count": pl.Int32,
    "worker_slot": pl.Utf8,
    "completion_timestamp": pl.Utf8,
    "error_category": pl.Utf8,
    "massive_cik": pl.Utf8,
    "massive_figi": pl.Utf8,
    "massive_composite_figi": pl.Utf8,
    "massive_name": pl.Utf8,
    "massive_type": pl.Utf8,
    "massive_exchange": pl.Utf8,
    "massive_active": pl.Boolean,
    "level1_status": pl.Utf8,
    "level2_start_status": pl.Utf8,
    "level2_end_status": pl.Utf8,
    "drift_detected": pl.Boolean,
    "drift_details": pl.Utf8,
}


class CheckpointManager:
    """Manages writing and reading checkpoint chunks to support robust backfill resumption."""

    def __init__(
        self,
        checkpoints_dir: Path | None = None,
        master_manifest_path: Path | None = None,
        logger: logging.Logger | None = None,
    ):
        self.checkpoints_dir = checkpoints_dir or (MANIFESTS_DIR / "checkpoints")
        self.master_manifest_path = master_manifest_path or (
            MANIFESTS_DIR / "massive_manifest.parquet"
        )
        self.logger = logger or logging.getLogger("checkpoint_manager")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.master_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def load_completed_spells(self) -> dict[str, dict[str, Any]]:
        """Scans and reads existing checkpoint files to discover already-completed spells."""
        completed: dict[str, dict[str, Any]] = {}
        chk_files = sorted(self.checkpoints_dir.glob("checkpoint_*.parquet"))
        if not chk_files:
            return completed

        self.logger.info(
            "Found %d checkpoint chunks. Loading completed records...", len(chk_files)
        )
        for f in chk_files:
            try:
                df = pl.read_parquet(f)
                for r in df.iter_rows(named=True):
                    completed[r["spell_id"]] = r
            except (OSError, pl.exceptions.PolarsError, RuntimeError, ValueError) as e:
                self.logger.warning("Could not read checkpoint %s: %s", f.name, e)

        self.logger.info("Loaded %d completed spells from checkpoints.", len(completed))
        return completed

    def save_checkpoint(self, records: list[dict[str, Any]], chunk_index: int) -> Path:
        """Atomically saves a batch of processed spell records to a checkpoint file."""
        df = pl.DataFrame(records, schema=MANIFEST_SCHEMA)
        target = self.checkpoints_dir / f"checkpoint_{chunk_index:05d}.parquet"
        atomic_write_parquet(df, target)
        self.logger.info("Checkpoint saved: %d records -> %s", df.height, target.name)
        return target

    def merge_all_checkpoints(self) -> pl.DataFrame:
        """Concatenates, deduplicates, and saves all checkpoints into master manifest."""
        chk_files = sorted(self.checkpoints_dir.glob("checkpoint_*.parquet"))
        if not chk_files:
            raise FileNotFoundError(
                f"No checkpoint files found in {self.checkpoints_dir}"
            )

        frames = [pl.read_parquet(f) for f in chk_files]
        merged = pl.concat(frames).unique(subset=["spell_id"], keep="last")
        merged = merged.sort(["ticker", "spell_seq"])
        atomic_write_parquet(merged, self.master_manifest_path)
        self.logger.info(
            "Wrote master manifest (%d rows) -> %s",
            merged.height,
            self.master_manifest_path,
        )
        return merged
