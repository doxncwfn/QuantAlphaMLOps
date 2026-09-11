"""
Historical Point-in-Time Identity Backfill Engine.
==================================================
Coordinates spell iteration, multi-worker querying, resumable checkpointing,
and master manifest generation.
"""

from __future__ import annotations
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from src.common.config import (
    EXPECTED_SPELLS_HASH,
    LOG_DIR,
    MANIFESTS_DIR,
    SPELLS_CSV_PATH,
)
from src.common.logging import setup_logger
from src.identity.massive.checkpoint import CheckpointManager
from src.identity.massive.worker_pool import ConcurrentKeyWorkerPool


class BackfillEngine:
    """High-level backfill engine coordinating 9-worker concurrent execution."""

    def __init__(
        self,
        worker_pool: ConcurrentKeyWorkerPool,
        checkpoint_manager: Optional[CheckpointManager] = None,
        max_live_queries: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        self.worker_pool = worker_pool
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.max_live_queries = max_live_queries
        self.logger = logger or setup_logger("backfill_engine", LOG_DIR / "v3_backfill.log")
        self.live_queries_executed = 0

    def verify_spells_integrity(self) -> pl.DataFrame:
        """Verifies spells.csv exists and matches expected SHA-256 hash."""
        if not SPELLS_CSV_PATH.exists():
            raise FileNotFoundError(f"Spells file not found: {SPELLS_CSV_PATH}")

        with open(SPELLS_CSV_PATH, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()

        if actual_hash != EXPECTED_SPELLS_HASH:
            raise ValueError(f"Spells hash mismatch! Expected {EXPECTED_SPELLS_HASH}, got {actual_hash}")

        df = pl.read_csv(SPELLS_CSV_PATH)
        self.logger.info("Spells SHA-256 verified (%s). Total spells: %d", actual_hash[:16], df.height)
        return df

    def run(
        self,
        batch_size: int = 1000,
        resume: bool = True,
    ) -> pl.DataFrame:
        """Runs the backfill over all spells."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING MASSIVE POINT-IN-TIME BACKFILL")
        self.logger.info("=" * 80)

        df_spells = self.verify_spells_integrity()
        total_spells = df_spells.height

        # Pre-index local caches
        self.worker_pool.cache_manager.index_caches()

        completed: Dict[str, Dict[str, Any]] = {}
        if resume:
            completed = self.checkpoint_manager.load_completed_spells()
            self.logger.info("Resuming: %d / %d spells already completed.", len(completed), total_spells)

        records: List[Dict[str, Any]] = []
        chunk_idx = len(list(self.checkpoint_manager.checkpoints_dir.glob("checkpoint_*.parquet")))
        num_workers = len(self.worker_pool.workers)

        t_start = time.time()
        for idx, row in enumerate(df_spells.iter_rows(named=True)):
            spell_id = f"{row['ticker'].strip()}_{row['spell_seq']}"
            if spell_id in completed:
                continue

            allow_live = self.live_queries_executed < self.max_live_queries
            rec = self.worker_pool.process_spell(row, worker_idx=idx % num_workers, allow_live=allow_live)
            if rec.get("cache_status") == "MISS":
                self.live_queries_executed += 1

            records.append(rec)
            completed[spell_id] = rec

            if len(records) >= batch_size:
                self.checkpoint_manager.save_checkpoint(records, chunk_idx)
                chunk_idx += 1
                records = []
                self.logger.info("Progress: %d / %d spells (%.2f%%)", len(completed), total_spells, len(completed) / total_spells * 100)

        if records:
            self.checkpoint_manager.save_checkpoint(records, chunk_idx)
            chunk_idx += 1

        master_manifest = self.checkpoint_manager.merge_all_checkpoints()
        elapsed = time.time() - t_start
        self.logger.info("Backfill complete in %.2f seconds.", elapsed)
        return master_manifest


def main():
    parser = argparse.ArgumentParser(description="Massive Point-in-Time Identity Backfill")
    parser.add_argument("--batch-size", type=int, default=1000, help="Checkpoints batch size")
    parser.add_argument("--max-live-queries", type=int, default=0, help="Limit on live queries (0 = cache/offline only)")
    parser.add_argument("--no-resume", action="store_true", help="Do not load existing checkpoints")
    parser.add_argument("--full-live", action="store_true", help="Allow unbounded live queries")
    args = parser.parse_args()

    max_live = 999999 if args.full_live else args.max_live_queries
    pool = ConcurrentKeyWorkerPool()
    engine = BackfillEngine(worker_pool=pool, max_live_queries=max_live)
    engine.run(batch_size=args.batch_size, resume=not args.no_resume)


if __name__ == "__main__":
    main()
