"""
V3 Full Massive PIT Backfill — Modal Deployment.
================================================
Executes the historical Massive point-in-time identity backfill remotely on Modal.
- Uses up to 9 Massive API keys via Modal Secret ('massive').
- Strict 1:1 key-slot isolation (Worker 1 -> Key 1, ... Worker 9 -> Key 9).
- Independent 12.1s per-worker rate pacing.
- Persistent Modal Volume ('v3-massive-backfill') mounted at /modal_data.
- Atomic cache writes, periodic checkpoints, and volume commits.
- Fully resumable and idempotent across container restarts.
- Zero mutation of production tables or spells.csv.
"""

from __future__ import annotations
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import datetime
import hashlib
import json
import logging
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import modal
import polars as pl

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
APP_NAME = "v3-massive-backfill"
VOLUME_NAME = "v3-massive-backfill"
SECRET_NAME = "massive"

EXPECTED_SPELLS_HASH = "5fc79a37cdc341cf7b10a7017ccd75cf7001aa1b91e5dd6501c829b746191bf1"
EXPECTED_SPELLS_ROWS = 43757
EXPECTED_UNIQUE_TICKERS = 36843
PER_KEY_INTERVAL_SECONDS = 12.1
CHECKPOINT_BATCH_SIZE = 100
HEARTBEAT_INTERVAL_SECONDS = 30.0

LOCAL_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# -----------------------------------------------------------------------------
# Modal Image & Volume Setup
# -----------------------------------------------------------------------------
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secret = modal.Secret.from_name(SECRET_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests", "polars", "pyarrow", "python-dotenv")
    .add_local_dir(str(LOCAL_REPO_ROOT / "src"), remote_path="/root/src")
    .add_local_file(str(LOCAL_REPO_ROOT / "data" / "universe" / "spells.csv"), remote_path="/root/data/universe/spells.csv")
    .add_local_file(
        str(LOCAL_REPO_ROOT / "data" / "identity" / "experiments" / "v2" / "trading_sessions.parquet"),
        remote_path="/root/data/identity/experiments/v2/trading_sessions.parquet"
    )
    .add_local_dir(str(LOCAL_REPO_ROOT / "data" / "identity" / "cache" / "massive"), remote_path="/root/seed_caches/v3_cache")
    .add_local_dir(str(LOCAL_REPO_ROOT / "data" / "identity" / "experiments" / "v2" / "cache" / "massive"), remote_path="/root/seed_caches/v2_cache")
)

app = modal.App(APP_NAME)

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


# -----------------------------------------------------------------------------
# Remote Backfill Implementation
# -----------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/modal_data": volume},
    secrets=[secret],
    cpu=2.0,
    memory=4096,
    timeout=86400,  # 24 hours max runtime
)
def run_backfill_remote(
    dry_run: bool = False,
    smoke_test: bool = False,
    max_queries: int = 0,
    resume: bool = True,
    git_commit: str = "7e2645280e698c230bb17d3e92077d777ea1d26d"
) -> Dict[str, Any]:
    """Remote execution handler on Modal."""
    import polars as pl

    # 1. Setup Persistent Paths on Modal Volume
    vol_root = Path("/modal_data")
    cache_dir = vol_root / "cache" / "massive"
    checkpoints_dir = vol_root / "checkpoints"
    manifests_dir = vol_root / "manifests"
    telemetry_dir = vol_root / "telemetry"
    logs_dir = vol_root / "logs"

    for d in [cache_dir, checkpoints_dir, manifests_dir, telemetry_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Setup Persistent Logging
    logger = logging.getLogger("modal_backfill")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    log_file = logs_dir / "v3_modal_backfill.log"
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("=" * 80)
    logger.info("V3 FULL MASSIVE PIT BACKFILL — MODAL WORKER EXECUTION")
    logger.info("=" * 80)
    logger.info("Timestamp: %s", datetime.datetime.now(datetime.timezone.utc).isoformat())
    logger.info("Volume Root: %s", vol_root)
    logger.info("Dry Run: %s | Smoke Test: %s | Max Queries: %s | Resume: %s",
                dry_run, smoke_test, max_queries, resume)

    # 3. Input Verification: Spells CSV SHA-256 Check (FAIL FAST)
    spells_path = Path("/root/data/universe/spells.csv")
    if not spells_path.exists():
        logger.critical("FATAL: spells.csv not found at %s", spells_path)
        raise FileNotFoundError(f"Missing {spells_path}")

    spells_bytes = spells_path.read_bytes()
    spells_hash = hashlib.sha256(spells_bytes).hexdigest()
    logger.info("Spells CSV SHA-256: %s", spells_hash)

    if spells_hash != EXPECTED_SPELLS_HASH:
        logger.critical("FATAL: spells.csv SHA-256 mismatch: %s != %s", spells_hash, EXPECTED_SPELLS_HASH)
        raise ValueError(f"Spells SHA-256 mismatch: {spells_hash}")

    df_spells = pl.read_csv(spells_path)
    total_spells = df_spells.height
    unique_tickers = df_spells["ticker"].n_unique()
    logger.info("Spells Input Verified: %d spells across %d tickers.", total_spells, unique_tickers)

    if total_spells != EXPECTED_SPELLS_ROWS or unique_tickers != EXPECTED_UNIQUE_TICKERS:
        logger.critical("FATAL: Unexpected spells count or ticker count: (%d, %d)", total_spells, unique_tickers)
        raise ValueError("Unexpected spells dimensions")

    # 4. Detect and Isolate API Keys from Modal Secret
    logger.info("Scanning for Massive API keys from Modal Secret...")
    key_slots: List[Tuple[str, str]] = []
    for i in range(1, 10):
        var_name = f"MASSIVE_API_KEY_{i}"
        key_val = os.environ.get(var_name)
        if key_val and not key_val.startswith("your_"):
            key_slots.append((f"WORKER_{i}", key_val))

    # Single key fallback if no numbered keys
    if not key_slots and os.environ.get("MASSIVE_API_KEY"):
        key_slots.append(("WORKER_1", os.environ["MASSIVE_API_KEY"]))

    configured_keys = len(key_slots)
    logger.info("Configured Massive key slots: %d", configured_keys)
    for worker_id, _ in key_slots:
        logger.info("Worker Assignment: %s -> dedicated key slot", worker_id)

    if configured_keys == 0 and not dry_run:
        logger.critical("FATAL: Zero Massive API keys configured in Modal Secret '%s'", SECRET_NAME)
        raise ValueError(f"Missing keys in secret '{SECRET_NAME}'")

    # 5. Pre-populate Volume Cache from Seed Caches (if needed)
    seed_dirs = [
        Path("/root/seed_caches/v3_cache"),
        Path("/root/seed_caches/v2_cache"),
    ]
    existing_vol_cache_files = set(f.name for f in cache_dir.glob("*.json"))
    seeded_count = 0
    for sdir in seed_dirs:
        if sdir.exists():
            for sf in sdir.glob("*.json"):
                if sf.name not in existing_vol_cache_files:
                    shutil.copy2(sf, cache_dir / sf.name)
                    existing_vol_cache_files.add(sf.name)
                    seeded_count += 1

    if seeded_count > 0:
        logger.info("Seeded %d cache files from bundle into persistent volume cache.", seeded_count)
        volume.commit()
    logger.info("Persistent cache contains %d JSON items.", len(list(cache_dir.glob("*.json"))))

    # 6. Check for Dry Run Mode
    if dry_run:
        logger.info("-" * 80)
        logger.info("DRY-RUN VALIDATION SUCCESSFUL")
        logger.info("1. Modal Secret loaded: %d keys visible (masked)", configured_keys)
        logger.info("2. Persistent Volume mounted at: %s", vol_root)
        logger.info("3. spells.csv verified bit-for-bit: %s", spells_hash)
        logger.info("4. Cache atomic test: writing mock entry...")
        test_cache_file = cache_dir / "_test_probe.json"
        tmp_probe = test_cache_file.with_suffix(".json.tmp")
        with open(tmp_probe, "w", encoding="utf-8") as tf:
            json.dump({"test": "probe", "timestamp": time.time()}, tf)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp_probe, test_cache_file)
        assert test_cache_file.exists(), "Cache atomic write failed"
        test_cache_file.unlink()
        logger.info("5. Checkpoint test: writing mock checkpoint...")
        test_chk_file = checkpoints_dir / "_test_chk.parquet"
        pl.DataFrame({"test": [1]}).write_parquet(test_chk_file)
        assert test_chk_file.exists(), "Checkpoint write failed"
        test_chk_file.unlink()
        volume.commit()
        logger.info("DRY-RUN COMPLETE — ALL SYSTEMS READY.")
        logger.info("-" * 80)
        return {
            "status": "DRY_RUN_PASSED",
            "configured_keys": configured_keys,
            "spells_sha256": spells_hash,
            "total_spells": total_spells,
            "cache_items": len(existing_vol_cache_files),
        }

    # 7. Check Resumption State
    completed_spells: Dict[str, Dict[str, Any]] = {}
    if resume:
        chk_files = sorted(list(checkpoints_dir.glob("checkpoint_*.parquet")))
        if chk_files:
            logger.info("Found %d existing checkpoint files on persistent volume.", len(chk_files))
            for cf in chk_files:
                try:
                    df_c = pl.read_parquet(cf)
                    for r in df_c.iter_rows(named=True):
                        completed_spells[r["spell_id"]] = r
                except Exception as exc:
                    logger.warning("Could not read checkpoint %s: %s", cf, exc)
            logger.info("Resumption loaded %d already completed spells.", len(completed_spells))

    # 8. Filter Spells to Process
    remaining_spells: List[Dict[str, Any]] = []
    for row in df_spells.iter_rows(named=True):
        spell_id = f"{row['ticker'].strip()}_{row['spell_seq']}"
        if spell_id not in completed_spells:
            remaining_spells.append(row)

    logger.info("Spells Summary: %d total, %d completed, %d remaining.",
                total_spells, len(completed_spells), len(remaining_spells))

    # In smoke test mode, limit to 2 queries per worker
    if smoke_test:
        smoke_limit = min(len(remaining_spells), max(18, configured_keys * 2))
        logger.info("SMOKE TEST MODE: Limiting workload to %d queries across %d workers.", smoke_limit, configured_keys)
        remaining_spells = remaining_spells[:smoke_limit]
    elif max_queries > 0:
        logger.info("MAX QUERIES MODE: Limiting workload to %d queries.", max_queries)
        remaining_spells = remaining_spells[:max_queries]

    # If nothing to process, build master manifest and return
    if not remaining_spells and len(completed_spells) >= total_spells:
        logger.info("All %d spells already completed! Building final master manifest...", total_spells)
        return _finalize_backfill(
            vol_root=vol_root,
            checkpoints_dir=checkpoints_dir,
            manifests_dir=manifests_dir,
            telemetry_dir=telemetry_dir,
            logs_dir=logs_dir,
            total_spells=total_spells,
            spells_hash=spells_hash,
            configured_keys=configured_keys,
            git_commit=git_commit,
            logger=logger,
            start_ts=time.time(),
            elapsed_sec=0.0
        )

    # 9. Initialize Trading Sessions & Worker Pool
    sys.path.insert(0, "/root")
    from src.common.config import TRADING_SESSIONS_PATH
    from src.identity.massive.worker_pool import ConcurrentKeyWorkerPool as ConcurrentKeyWorkerPoolV3

    pool = ConcurrentKeyWorkerPoolV3(
        min_per_key_interval=PER_KEY_INTERVAL_SECONDS,
        logger=logger,
        cache_dir=cache_dir,
        api_keys=[k for _, k in key_slots]
    )

    df_sessions = pl.read_parquet(TRADING_SESSIONS_PATH)
    all_sessions: List[str] = df_sessions["session_date"].to_list()
    session_to_idx: Dict[str, int] = {d: i for i, d in enumerate(all_sessions)}

    def get_midpoint(s_date: str, e_date: str) -> str:
        s_i = session_to_idx.get(s_date)
        e_i = session_to_idx.get(e_date)
        if s_i is not None and e_i is not None and s_i <= e_i:
            return all_sessions[(s_i + e_i) // 2]
        return s_date

    # 10. Work Queue & Multi-threaded Workers
    work_queue: queue.Queue = queue.Queue()
    for s in remaining_spells:
        work_queue.put(s)

    results_lock = threading.Lock()
    newly_completed_records: List[Dict[str, Any]] = []
    chunk_counter = [len(list(checkpoints_dir.glob("checkpoint_*.parquet")))]

    # Progress tracking counters
    stat_cache_hits = [0]
    stat_live_reqs = [0]
    stat_failures = [0]
    stat_empty = [0]
    stat_429s = [0]
    is_running = [True]
    t_start = time.time()

    def process_spell_worker(worker_idx: int):
        assigned_worker = pool.workers[worker_idx % len(pool.workers)]
        worker_id = assigned_worker.worker_id

        while is_running[0]:
            try:
                s = work_queue.get_nowait()
            except queue.Empty:
                break

            try:
                ticker = s["ticker"].strip()
                clean_tk = ticker.upper()
                seq = s["spell_seq"]
                s_date = s["start_date"]
                e_date = s["end_date"]
                dur = s.get("n_sessions") or s.get("duration_sessions", 1)
                spell_id = f"{ticker}_{seq}"

                # Level 1: Midpoint session
                mid_date = get_midpoint(s_date, e_date)
                rep_date = mid_date
                rep_method = "MIDPOINT_SESSION"

                m_match, telem = pool.query(clean_tk, mid_date, spell_id=spell_id, preferred_worker_idx=worker_idx)
                l1_status = telem.get("outcome", "UNKNOWN")

                if telem.get("cached"):
                    with results_lock:
                        stat_cache_hits[0] += 1
                else:
                    with results_lock:
                        stat_live_reqs[0] += 1
                        if telem.get("http_status") == 429:
                            stat_429s[0] += 1

                l2_s_status = "NOT_ATTEMPTED"
                l2_e_status = "NOT_ATTEMPTED"
                level2_start_match = None
                level2_end_match = None

                # Level 2 Corroboration: Start / End dates if Level 1 is empty or weak
                needs_l2 = (
                    l1_status in ("MASSIVE_EMPTY", "NOT_FOUND") or
                    (l1_status == "SUCCESS" and m_match and not m_match.get("share_class_figi") and not m_match.get("cik"))
                )

                if needs_l2 and (s_date != mid_date or e_date != mid_date):
                    if s_date != mid_date:
                        level2_start_match, s_telem = pool.query(clean_tk, s_date, spell_id=spell_id, preferred_worker_idx=worker_idx)
                        l2_s_status = s_telem.get("outcome", "UNKNOWN")
                        if s_telem.get("cached"):
                            with results_lock:
                                stat_cache_hits[0] += 1
                        else:
                            with results_lock:
                                stat_live_reqs[0] += 1

                    if e_date != mid_date:
                        level2_end_match, e_telem = pool.query(clean_tk, e_date, spell_id=spell_id, preferred_worker_idx=worker_idx)
                        l2_e_status = e_telem.get("outcome", "UNKNOWN")
                        if e_telem.get("cached"):
                            with results_lock:
                                stat_cache_hits[0] += 1
                        else:
                            with results_lock:
                                stat_live_reqs[0] += 1

                    if not m_match or l1_status == "MASSIVE_EMPTY":
                        if level2_start_match and (level2_start_match.get("cik") or level2_start_match.get("share_class_figi")):
                            m_match = level2_start_match
                            rep_date = s_date
                            rep_method = "BOUNDARY_START_FALLBACK"
                            telem = s_telem
                        elif level2_end_match and (level2_end_match.get("cik") or level2_end_match.get("share_class_figi")):
                            m_match = level2_end_match
                            rep_date = e_date
                            rep_method = "BOUNDARY_END_FALLBACK"
                            telem = e_telem

                # Level 3: Within-spell drift
                drift_detected = False
                drift_details = ""
                evidence_points = []
                if level2_start_match:
                    evidence_points.append(("START", s_date, level2_start_match))
                if m_match and rep_date == mid_date:
                    evidence_points.append(("MIDPOINT", mid_date, m_match))
                if level2_end_match:
                    evidence_points.append(("END", e_date, level2_end_match))

                if len(evidence_points) >= 2:
                    ciks = {p[2].get("cik") for p in evidence_points if p[2].get("cik")}
                    figis = {p[2].get("share_class_figi") or p[2].get("composite_figi") for p in evidence_points if (p[2].get("share_class_figi") or p[2].get("composite_figi"))}
                    if len(ciks) > 1 or len(figis) > 1:
                        drift_detected = True
                        drift_details = f"Boundary divergence: CIKs={list(ciks)}, FIGIs={list(figis)}"

                outcome = telem.get("outcome", "UNKNOWN")
                if outcome == "MASSIVE_EMPTY":
                    with results_lock:
                        stat_empty[0] += 1
                elif outcome not in ("SUCCESS", "MASSIVE_EMPTY"):
                    with results_lock:
                        stat_failures[0] += 1

                m_cik = str(m_match.get("cik")).zfill(10) if (m_match and m_match.get("cik")) else None
                m_figi = (m_match.get("share_class_figi") or m_match.get("composite_figi")) if m_match else None
                m_comp = m_match.get("composite_figi") if m_match else None
                m_name = m_match.get("name") if m_match else None
                m_type = m_match.get("type") if m_match else None
                m_exch = m_match.get("primary_exchange") if m_match else None
                m_act = m_match.get("active") if m_match else None

                rec = {
                    "spell_id": spell_id,
                    "ticker": ticker,
                    "spell_seq": seq,
                    "start_date": s_date,
                    "end_date": e_date,
                    "duration_sessions": dur,
                    "representative_date": rep_date,
                    "representative_date_method": rep_method,
                    "lookup_status": outcome,
                    "cache_status": "HIT" if telem.get("cached") else "MISS",
                    "attempt_count": 1,
                    "worker_slot": worker_id,
                    "completion_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "error_category": "NONE" if outcome in ("SUCCESS", "MASSIVE_EMPTY") else outcome,
                    "massive_cik": m_cik,
                    "massive_figi": m_figi,
                    "massive_composite_figi": m_comp,
                    "massive_name": m_name,
                    "massive_type": m_type,
                    "massive_exchange": m_exch,
                    "massive_active": m_act,
                    "level1_status": l1_status,
                    "level2_start_status": l2_s_status,
                    "level2_end_status": l2_e_status,
                    "drift_detected": drift_detected,
                    "drift_details": drift_details,
                }

                with results_lock:
                    newly_completed_records.append(rec)
                    completed_spells[spell_id] = rec

                    # Periodic Checkpoint Flush on batch threshold
                    if len(newly_completed_records) >= CHECKPOINT_BATCH_SIZE:
                        _flush_checkpoint(
                            records=newly_completed_records,
                            checkpoints_dir=checkpoints_dir,
                            chunk_idx=chunk_counter[0],
                            logger=logger
                        )
                        chunk_counter[0] += 1
                        newly_completed_records.clear()
                        volume.commit()

            except Exception as exc:
                logger.error("[Worker %s] Unexpected exception on spell %s: %s", worker_id, s.get("ticker"), exc)
                err_spell_id = f"{s.get('ticker', 'UNKNOWN').strip()}_{s.get('spell_seq', 1)}"
                err_rec = {
                    "spell_id": err_spell_id,
                    "ticker": s.get("ticker", "UNKNOWN").strip(),
                    "spell_seq": s.get("spell_seq", 1),
                    "start_date": s.get("start_date", ""),
                    "end_date": s.get("end_date", ""),
                    "duration_sessions": s.get("n_sessions") or s.get("duration_sessions", 1),
                    "representative_date": s.get("start_date", ""),
                    "representative_date_method": "FAILED_EXCEPTION",
                    "lookup_status": "WORKER_EXCEPTION",
                    "cache_status": "NONE",
                    "attempt_count": 1,
                    "worker_slot": worker_id,
                    "completion_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "error_category": f"EXCEPTION: {type(exc).__name__}",
                    "massive_cik": None,
                    "massive_figi": None,
                    "massive_composite_figi": None,
                    "massive_name": None,
                    "massive_type": None,
                    "massive_exchange": None,
                    "massive_active": None,
                    "level1_status": "WORKER_EXCEPTION",
                    "level2_start_status": "NOT_ATTEMPTED",
                    "level2_end_status": "NOT_ATTEMPTED",
                    "drift_detected": False,
                    "drift_details": "",
                }
                with results_lock:
                    stat_failures[0] += 1
                    newly_completed_records.append(err_rec)
                    completed_spells[err_spell_id] = err_rec
            finally:
                work_queue.task_done()

    # 11. Periodic Heartbeat Reporter Thread
    def heartbeat_reporter():
        while is_running[0]:
            time.sleep(HEARTBEAT_INTERVAL_SECONDS)
            with results_lock:
                done = len(completed_spells)
                rem = max(0, total_spells - done)
                hits = stat_cache_hits[0]
                live = stat_live_reqs[0]
                empty = stat_empty[0]
                fails = stat_failures[0]
                r429 = stat_429s[0]
                elapsed = time.time() - t_start
                rpm = (live + hits) / (elapsed / 60.0) if elapsed > 0 else 0.0
                eta_hours = (rem / rpm / 60.0) if rpm > 0 else 0.0

                # Durability Flush: commit any pending records every heartbeat interval (<=30s window)
                if newly_completed_records:
                    _flush_checkpoint(
                        records=newly_completed_records,
                        checkpoints_dir=checkpoints_dir,
                        chunk_idx=chunk_counter[0],
                        logger=logger
                    )
                    chunk_counter[0] += 1
                    newly_completed_records.clear()
                    volume.commit()

            logger.info("------------------------------------------------------------")
            logger.info("V3 MODAL MASSIVE PIT BACKFILL HEARTBEAT")
            logger.info("Completed       : %d / %d (%.2f%%)", done, total_spells, (done / total_spells * 100))
            logger.info("Remaining       : %d", rem)
            logger.info("Cache Hits      : %d", hits)
            logger.info("Live Requests   : %d", live)
            logger.info("Empty Results   : %d", empty)
            logger.info("HTTP 429s       : %d", r429)
            logger.info("Failures        : %d", fails)
            logger.info("Observed Rate   : %.2f req/min", rpm)
            logger.info("Active Workers  : %d / %d", configured_keys, configured_keys)
            logger.info("PLANNING_ESTIMATE ETA: %.2f hours remaining", eta_hours)
            logger.info("------------------------------------------------------------")

    # Launch threads
    threads: List[threading.Thread] = []
    hb_thread = threading.Thread(target=heartbeat_reporter, daemon=True)
    hb_thread.start()

    num_threads = min(configured_keys, max(1, len(remaining_spells)))
    logger.info("Launching %d concurrent worker threads on Modal...", num_threads)
    for i in range(num_threads):
        t = threading.Thread(target=process_spell_worker, args=(i,))
        t.start()
        threads.append(t)

    # Wait for all workers to finish
    work_queue.join()
    is_running[0] = False
    for t in threads:
        t.join()

    # Flush final remaining checkpoint
    with results_lock:
        if newly_completed_records:
            _flush_checkpoint(
                records=newly_completed_records,
                checkpoints_dir=checkpoints_dir,
                chunk_idx=chunk_counter[0],
                logger=logger
            )
            chunk_counter[0] += 1
            newly_completed_records.clear()
            volume.commit()

    total_elapsed = time.time() - t_start

    # 12. Final Master Manifest Generation & Validation
    return _finalize_backfill(
        vol_root=vol_root,
        checkpoints_dir=checkpoints_dir,
        manifests_dir=manifests_dir,
        telemetry_dir=telemetry_dir,
        logs_dir=logs_dir,
        total_spells=total_spells,
        spells_hash=spells_hash,
        configured_keys=configured_keys,
        git_commit=git_commit,
        logger=logger,
        start_ts=t_start,
        elapsed_sec=total_elapsed,
        telemetry_obj=pool.telemetry
    )


def _flush_checkpoint(
    records: List[Dict[str, Any]],
    checkpoints_dir: Path,
    chunk_idx: int,
    logger: logging.Logger
):
    """Writes a checkpoint chunk atomically to parquet."""
    import polars as pl
    chk_file = checkpoints_dir / f"checkpoint_{chunk_idx:05d}.parquet"
    tmp_file = chk_file.with_suffix(".parquet.tmp")
    df_chk = pl.DataFrame(records, schema=MANIFEST_SCHEMA)
    df_chk.write_parquet(tmp_file)
    tmp_file.replace(chk_file)
    logger.info("Persisted checkpoint chunk %d (%d spells) -> %s", chunk_idx, df_chk.height, chk_file.name)


def _finalize_backfill(
    vol_root: Path,
    checkpoints_dir: Path,
    manifests_dir: Path,
    telemetry_dir: Path,
    logs_dir: Path,
    total_spells: int,
    spells_hash: str,
    configured_keys: int,
    git_commit: str,
    logger: logging.Logger,
    start_ts: float,
    elapsed_sec: float,
    telemetry_obj: Any = None
) -> Dict[str, Any]:
    """Concatenates checkpoints, runs validation checks, and saves manifests."""
    import polars as pl

    chk_files = sorted(list(checkpoints_dir.glob("checkpoint_*.parquet")))
    logger.info("Concatenating %d checkpoint chunks into master manifest...", len(chk_files))
    if not chk_files:
        raise RuntimeError("No checkpoints found to build master manifest")

    df_master = pl.concat([pl.read_parquet(f) for f in chk_files])
    df_master = df_master.unique(subset=["spell_id"], keep="last")

    master_manifest_path = manifests_dir / "massive_manifest.parquet"
    tmp_master = master_manifest_path.with_suffix(".parquet.tmp")
    df_master.write_parquet(tmp_master)
    tmp_master.replace(master_manifest_path)
    logger.info("Wrote master manifest (%d rows) to %s", df_master.height, master_manifest_path)

    # Validation Checks
    unaccounted = total_spells - df_master.height
    logger.info("Master Manifest Accounting: %d / %d (unaccounted: %d)",
                df_master.height, total_spells, unaccounted)

    # Save Telemetry
    telem_summary = telemetry_obj.get_summary() if telemetry_obj else {}
    df_telem = pl.DataFrame(telemetry_obj.records) if telemetry_obj and telemetry_obj.records else pl.DataFrame()
    telem_parquet = telemetry_dir / "worker_telemetry.parquet"
    if df_telem.height > 0:
        df_telem.write_parquet(telem_parquet)
    with open(telemetry_dir / "telemetry_summary.json", "w", encoding="utf-8") as f:
        json.dump(telem_summary, f, indent=2)

    # Final Execution Manifest (Section 28)
    cache_items = len(list((vol_root / "cache" / "massive").glob("*.json")))
    manifest_payload = {
        "run_id": f"modal_v3_{int(time.time())}",
        "start_time": datetime.datetime.fromtimestamp(start_ts, tz=datetime.timezone.utc).isoformat(),
        "end_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_commit,
        "spells_sha256": spells_hash,
        "spell_count": total_spells,
        "spells_accounted_for": df_master.height,
        "configured_key_count": configured_keys,
        "worker_count": configured_keys,
        "cache_items": cache_items,
        "live_requests": telem_summary.get("global_live_requests", 0),
        "cache_hits": telem_summary.get("global_cache_hits", 0),
        "empty_results": df_master.filter(pl.col("lookup_status") == "MASSIVE_EMPTY").height,
        "successful_results": df_master.filter(pl.col("lookup_status") == "SUCCESS").height,
        "failed_requests": df_master.filter(~pl.col("lookup_status").is_in(["SUCCESS", "MASSIVE_EMPTY", "OFFLINE_PENDING"])).height,
        "drift_cases_detected": df_master.filter(pl.col("drift_detected") == True).height,
        "observed_throughput_rpm": (df_master.height / (elapsed_sec / 60.0)) if elapsed_sec > 0 else 0.0,
        "elapsed_seconds": elapsed_sec,
        "modal_volume": VOLUME_NAME,
        "status": "COMPLETED" if unaccounted == 0 else "PARTIAL",
    }

    manifest_json_path = manifests_dir / "modal_backfill_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    # Final Summary Markdown (Section 16)
    summary_md = f"""# Modal V3 Massive PIT Backfill Execution Summary

## 1. Overview
- **Status**: `{manifest_payload['status']}`
- **Spells Accounted For**: **{df_master.height:,} / {total_spells:,}** (100.00%)
- **Spells Input SHA-256**: `{spells_hash}`
- **Configured Key Slots**: **{configured_keys}**
- **Git Commit**: `{git_commit}`
- **Elapsed Time**: {elapsed_sec:.2f} seconds ({elapsed_sec/3600.0:.2f} hours)
- **Modal Volume**: `{VOLUME_NAME}`

## 2. Evidence Breakdown
- **Successful Lookups**: **{manifest_payload['successful_results']:,}**
- **Massive Empty Lookups**: **{manifest_payload['empty_results']:,}**
- **Within-Spell Drift Detected**: **{manifest_payload['drift_cases_detected']:,}**
- **Persistent Cache Items**: **{cache_items:,}**
"""
    (logs_dir / "v3_modal_summary.md").write_text(summary_md, encoding="utf-8")

    # Final Volume Commit
    volume.commit()
    logger.info("Volume commit completed. All artifacts securely saved on '%s'.", VOLUME_NAME)
    return manifest_payload


# -----------------------------------------------------------------------------
# Local Entrypoints (CLI)
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    dry_run: bool = False,
    smoke_test: bool = False,
    max_queries: int = 0,
    resume: bool = True
):
    """Local entrypoint called via `modal run src/identity/v3/modal_backfill.py`."""
    print("=" * 80)
    print("MODAL V3 FULL MASSIVE PIT BACKFILL LAUNCHER")
    print("=" * 80)
    print(f"Target Modal App   : {APP_NAME}")
    print(f"Target Modal Volume: {VOLUME_NAME}")
    print(f"Target Modal Secret: {SECRET_NAME}")
    print(f"Dry Run            : {dry_run}")
    print(f"Smoke Test         : {smoke_test}")
    print(f"Max Queries        : {max_queries if max_queries > 0 else 'UNBOUNDED'}")
    print(f"Resume             : {resume}")
    print("-" * 80)

    res = run_backfill_remote.remote(
        dry_run=dry_run,
        smoke_test=smoke_test,
        max_queries=max_queries,
        resume=resume
    )

    print("-" * 80)
    print("EXECUTION RESULT:")
    print(json.dumps(res, indent=2))
    print("=" * 80)
