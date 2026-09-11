# Download command: modal volume get "US_market_universe" / ./data
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import modal
import pandas_market_calendars as pd_mcal
import polars as pl
import requests

NUM_WORKERS = 9
START_DATE = "2004-01-01"
END_DATE = "2026-09-01"

VOLUME_NAME = "US_market_universe"
APP_NAME = "Market Reconstruction"
MODAL_SECRET_NAME = "massive"

VOLUME_ROOT = Path("/mnt/data")

DATA_RAW_DIR = VOLUME_ROOT / "raw" / "massive" / "active_tickers"
MANIFEST_DIR = VOLUME_ROOT / "manifests"
FINAL_MANIFEST_FILE = MANIFEST_DIR / "active_tickers_manifest.parquet"

INTER_PAGE_DELAY = 12.0
INITIAL_BACKOFF = 3.0
MAX_BACKOFF = 30.0
MAX_REQUEST_RETRIES = 10

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ExtractionError(Exception):
    pass


class MassiveThrottledManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def get_active_symbols_on_date(self, target_date: str) -> list[str]:
        active_symbols = []
        seen = set()

        url = "https://api.massive.com/v3/reference/tickers"
        params = {
            "date": target_date,
            "active": "true",
            "market": "stocks",
            "limit": 1000,
            "apiKey": self.api_key,
        }

        page_num = 1
        while True:
            added_tickers, next_url = self._fetch_page(
                target_date, page_num, url, params
            )

            for t in added_tickers:
                if t not in seen:
                    seen.add(t)
                    active_symbols.append(t)

            if next_url is None:
                break

            url = next_url
            params = {}
            if "apiKey" not in url:
                params["apiKey"] = self.api_key

            page_num += 1
            time.sleep(INTER_PAGE_DELAY)

        if not active_symbols:
            raise ExtractionError(
                f"[{target_date}] Extraction yielded zero tickers. Failing date."
            )

        return active_symbols

    def _fetch_page(self, target_date: str, page_num: int, url: str, params: dict):
        retries = 0
        backoff = INITIAL_BACKOFF

        while True:
            try:
                logger.debug("[%s] Fetching page %d...", target_date, page_num)
                response = self.session.get(url, params=params, timeout=30.0)

                if response.status_code == 429:
                    retries += 1
                    if retries >= MAX_REQUEST_RETRIES:
                        raise ExtractionError(
                            f"[{target_date}] Page {page_num} failed after {MAX_REQUEST_RETRIES} retries (HTTP 429)"
                        )
                    logger.warning(
                        "[%s] HTTP 429 on page %d. Retrying in %.1f seconds.",
                        target_date,
                        page_num,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                    continue

                response.raise_for_status()
                payload = response.json()

                results = payload.get("results", [])
                if not results:
                    return [], None

                extracted_symbols = []
                for idx, item in enumerate(results):
                    if not isinstance(item, dict):
                        raise ExtractionError(
                            f"[{target_date}] Page {page_num}: result[{idx}] is not a dict. Aborting."
                        )
                    raw_ticker = item.get("ticker")
                    if not raw_ticker or not str(raw_ticker).strip():
                        raise ExtractionError(
                            f"[{target_date}] Page {page_num}: missing/empty ticker. Aborting."
                        )

                    t = str(raw_ticker).strip()
                    extracted_symbols.append(t)

                next_url = payload.get("next_url")
                if next_url is not None and not isinstance(next_url, str):
                    raise ExtractionError(
                        f"[{target_date}] Page {page_num}: malformed next_url (got {type(next_url).__name__}). Aborting."
                    )
                return extracted_symbols, next_url

            except requests.exceptions.RequestException as exc:
                retries += 1
                if retries >= MAX_REQUEST_RETRIES:
                    raise ExtractionError(
                        f"[{target_date}] Page {page_num} network error after {MAX_REQUEST_RETRIES} retries: {exc}"
                    )
                wait = 5.0 * retries
                logger.warning(
                    "[%s] Network error on page %d. Retrying in %.1f s.",
                    target_date,
                    page_num,
                    wait,
                )
                time.sleep(wait)


def ensure_directories():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest(worker_id: int) -> pl.DataFrame:
    manifest_file = MANIFEST_DIR / f"worker_{worker_id}.parquet"
    if not manifest_file.exists():
        schema = {
            "date": pl.Utf8,
            "status": pl.Utf8,
            "ticker_count": pl.Int64,
            "processed_at": pl.Utf8,
            "error": pl.Utf8,
        }
        return pl.DataFrame(schema=schema)
    try:
        return pl.read_parquet(manifest_file)
    except (OSError, pl.exceptions.PolarsError, RuntimeError, ValueError) as exc:
        logger.critical(
            "Manifest for worker %d is unreadable or corrupt: %s", worker_id, exc
        )
        sys.exit(1)


def save_manifest(worker_id: int, manifest: pl.DataFrame):
    manifest_file = MANIFEST_DIR / f"worker_{worker_id}.parquet"
    tmp_path = manifest_file.with_suffix(".parquet.tmp")
    manifest.write_parquet(tmp_path)
    tmp_path.replace(manifest_file)


def is_date_already_successful(manifest: pl.DataFrame, target_date: str) -> bool:
    if manifest.is_empty():
        return False
    row = manifest.filter(pl.col("date") == target_date)
    if row.is_empty():
        return False
    return row["status"][0] == "success"


def upsert_manifest_row(
    manifest: pl.DataFrame,
    target_date: str,
    status: str,
    ticker_count: int,
    processed_at: str,
    error_msg: str | None,
) -> pl.DataFrame:
    new_row = pl.DataFrame(
        {
            "date": [target_date],
            "status": [status],
            "ticker_count": [ticker_count],
            "processed_at": [processed_at],
            "error": [error_msg],
        },
        schema=manifest.schema,
    )
    if manifest.is_empty():
        return new_row

    manifest = manifest.filter(pl.col("date") != target_date)
    return pl.concat([manifest, new_row])


def save_raw_snapshot(target_date: str, tickers: list[str]):
    year = target_date[:4]
    year_dir = DATA_RAW_DIR / year
    year_dir.mkdir(parents=True, exist_ok=True)

    json_path = year_dir / f"{target_date}.json"
    tmp_path = json_path.with_suffix(".json.tmp")

    payload = {
        "date": target_date,
        "tickers": tickers,
        "ticker_count": len(tickers),
    }

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    tmp_path.replace(json_path)


def get_xnys_trading_dates(start_date: str, end_date: str) -> list[str]:
    calendar = pd_mcal.get_calendar("NYSE")
    schedule = calendar.schedule(
        start_date=start_date,
        end_date=end_date,
    )
    return [d.strftime("%Y-%m-%d") for d in schedule.index]


volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "requests", "polars", "pyarrow", "pandas_market_calendars"
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    cpu=1.0,
    memory=256,
)
def validate_api_keys():
    logger.info(
        "Validating all %d API keys are present in the Modal Secret...", NUM_WORKERS
    )
    missing = []
    for i in range(1, NUM_WORKERS + 1):
        env_var_name = f"MASSIVE_API_KEY_{i}"
        if not os.environ.get(env_var_name):
            missing.append(env_var_name)
    if missing:
        logger.critical(
            "Missing API keys in Secret %s: %s", MODAL_SECRET_NAME, ", ".join(missing)
        )
        sys.exit(1)
    logger.info("All %d API keys are present.", NUM_WORKERS)


@app.function(
    image=image,
    volumes={"/mnt/data": volume},
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    cpu=1.0,
    memory=1024,
    timeout=86400,
)
def extract_worker(worker_id: int, assigned_dates: list[str]):
    logger.info("=" * 70)
    logger.info("Worker %d STARTING", worker_id)
    logger.info("Assigned dates      : %d", len(assigned_dates))
    if assigned_dates:
        logger.info(
            "Date range          : %s -> %s", assigned_dates[0], assigned_dates[-1]
        )
    logger.info("Volume root         : %s", VOLUME_ROOT)
    logger.info("=" * 70)

    env_var_name = f"MASSIVE_API_KEY_{worker_id}"
    api_key = os.environ.get(env_var_name)
    if not api_key:
        logger.critical(
            "Worker %d API key missing. Ensure %s is set in the Secret.",
            worker_id,
            env_var_name,
        )
        sys.exit(1)

    ensure_directories()
    manifest = load_manifest(worker_id)

    already_done = sum(
        1 for d in assigned_dates if is_date_already_successful(manifest, d)
    )
    logger.info(
        "Worker %d: Already completed (will skip): %d / %d",
        worker_id,
        already_done,
        len(assigned_dates),
    )

    manager = MassiveThrottledManager(api_key=api_key)

    n_skipped = 0
    n_success = 0
    n_failed = 0

    for i, target_date in enumerate(assigned_dates, start=1):
        if is_date_already_successful(manifest, target_date):
            logger.debug(
                "[Worker %d | %s] Already successful -- skipping.",
                worker_id,
                target_date,
            )
            n_skipped += 1
            continue

        logger.info(
            "[Worker %d | %s] Processing (%d / %d)...",
            worker_id,
            target_date,
            i,
            len(assigned_dates),
        )
        processed_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        try:
            tickers = manager.get_active_symbols_on_date(target_date)
            tickers = list(dict.fromkeys(tickers))

            save_raw_snapshot(target_date, tickers)

            manifest = upsert_manifest_row(
                manifest, target_date, "success", len(tickers), processed_at, None
            )
            save_manifest(worker_id, manifest)

            volume.commit()
            logger.info(
                "[Worker %d | %s] SUCCESS -- %d tickers. Volume committed.",
                worker_id,
                target_date,
                len(tickers),
            )
            n_success += 1

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "[Worker %d | %s] FAILED: %s", worker_id, target_date, error_msg
            )
            manifest = upsert_manifest_row(
                manifest, target_date, "failed", 0, processed_at, error_msg
            )
            try:
                save_manifest(worker_id, manifest)
                volume.commit()
                logger.info(
                    "[Worker %d | %s] Failure state saved to manifest and volume committed.",
                    worker_id,
                    target_date,
                )
            except (OSError, pl.exceptions.PolarsError) as save_exc:
                logger.error(
                    "[Worker %d | %s] Could not save manifest after failure: %s",
                    worker_id,
                    target_date,
                    save_exc,
                )

            n_failed += 1

    logger.info("=" * 70)
    logger.info("Worker %d extraction complete.", worker_id)
    logger.info("Successfully processed : %d", n_success)
    logger.info("Failed                 : %d", n_failed)
    logger.info("Skipped                : %d", n_skipped)
    logger.info("=" * 70)
    return worker_id, n_success, n_failed


@app.function(
    image=image,
    volumes={"/mnt/data": volume},
    cpu=1.0,
    memory=1024,
    timeout=3600,
)
def merge_manifests(expected_dates: list[str]):
    logger.info("=" * 70)
    logger.info("Starting Manifest Merge and Coverage Validation")
    logger.info("=" * 70)

    manifest_dfs = []
    worker_ids_loaded = []

    for worker_id in range(1, NUM_WORKERS + 1):
        worker_file = MANIFEST_DIR / f"worker_{worker_id}.parquet"
        if worker_file.exists():
            df = pl.read_parquet(worker_file)
            manifest_dfs.append(df)
            worker_ids_loaded.append(worker_id)
            logger.info(
                "Loaded manifest for Worker %d: %d records", worker_id, df.height
            )
        else:
            logger.warning(
                "Manifest for Worker %d not found at %s", worker_id, worker_file
            )

    if not manifest_dfs:
        logger.error("No worker manifests found to merge.")
        return

    merged_df = pl.concat(manifest_dfs)

    dup_counts = merged_df.group_by("date").len().filter(pl.col("len") > 1)
    if not dup_counts.is_empty():
        duplicate_dates = dup_counts["date"].to_list()
        logger.error(
            "ERROR: %d duplicate dates found across worker manifests!",
            len(duplicate_dates),
        )

        for dup in duplicate_dates:
            workers_with_dup = []
            for wid, df in zip(worker_ids_loaded, manifest_dfs):
                if not df.filter(pl.col("date") == dup).is_empty():
                    workers_with_dup.append(f"worker_{wid}")
            logger.error(
                "Date %s is duplicated in: %s", dup, ", ".join(workers_with_dup)
            )

        logger.error(
            "Failing the merge due to duplicates. Fix the assignment bug or corrupted state."
        )
        sys.exit(1)

    merged_df = merged_df.sort("date")

    tmp_path = FINAL_MANIFEST_FILE.with_suffix(".parquet.tmp")
    merged_df.write_parquet(tmp_path)
    tmp_path.replace(FINAL_MANIFEST_FILE)
    volume.commit()
    logger.info(
        "Canonical manifest updated and volume committed. Total records: %d",
        merged_df.height,
    )

    expected_set = set(expected_dates)
    represented_set = set(merged_df["date"].to_list())

    success_df = merged_df.filter(pl.col("status") == "success")
    failed_df = merged_df.filter(pl.col("status") == "failed")

    successful_dates = set(success_df["date"].to_list())
    failed_dates = set(failed_df["date"].to_list())

    missing_dates = expected_set - represented_set

    logger.info("--- Coverage Validation ---")
    logger.info("Expected sessions : %d", len(expected_dates))
    logger.info("Successful dates  : %d", len(successful_dates))
    logger.info("Failed dates      : %d", len(failed_dates))
    logger.info("Missing dates     : %d", len(missing_dates))
    logger.info("Duplicate dates   : 0 (verified)")

    if missing_dates:
        logger.warning(
            "WARNING: There are %d expected dates missing from the manifests.",
            len(missing_dates),
        )

    if failed_dates:
        logger.warning(
            "WARNING: There are %d failed dates recorded in the manifests.",
            len(failed_dates),
        )

    logger.info("Merge and validation complete.")


@app.local_entrypoint()
def main():
    logger.info("Pre-flight: Validating Modal Secrets...")
    validate_api_keys.remote()

    logger.info("Calculating XNYS trading sessions...")
    trading_dates = get_xnys_trading_dates(START_DATE, END_DATE)
    total_sessions = len(trading_dates)

    if not trading_dates:
        logger.critical("No trading dates found! Aborting.")
        sys.exit(1)

    first_session = trading_dates[0]
    last_session = trading_dates[-1]

    if sorted(trading_dates) != trading_dates:
        logger.critical("Trading dates are not sorted. Aborting.")
        sys.exit(1)

    if len(set(trading_dates)) != total_sessions:
        logger.critical("Duplicate dates found in trading sessions. Aborting.")
        sys.exit(1)

    logger.info("First session: %s", first_session)
    logger.info("Last session: %s", last_session)
    logger.info("Total sessions found: %d", total_sessions)

    if first_session > "2004-01-02":
        logger.critical(
            "First session %s is later than 2004-01-02. Aborting.", first_session
        )
        sys.exit(1)

    if last_session != "2026-09-01":
        logger.critical(
            "Last session %s is not exactly 2026-09-01. Aborting.", last_session
        )
        sys.exit(1)

    chunk_size = total_sessions // NUM_WORKERS
    remainder = total_sessions % NUM_WORKERS

    chunks = []
    start_idx = 0
    for i in range(NUM_WORKERS):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        chunks.append(trading_dates[start_idx:end_idx])
        start_idx = end_idx

    logger.info("Worker Assignments:")
    for i in range(NUM_WORKERS):
        worker_id = i + 1
        chunk = chunks[i]
        if chunk:
            logger.info(
                "Worker %d: %s -> %s, %d sessions",
                worker_id,
                chunk[0],
                chunk[-1],
                len(chunk),
            )
        else:
            logger.info("Worker %d: No sessions assigned", worker_id)

    logger.info("Spawning %d remote concurrent workers...", NUM_WORKERS)

    calls = []
    for i in range(NUM_WORKERS):
        worker_id = i + 1
        call = extract_worker.spawn(worker_id, chunks[i])
        calls.append(call)

    for call in calls:
        call.get()

    logger.info("All %d workers have finished. Spawning manifest merge...", NUM_WORKERS)
    merge_manifests.remote(trading_dates)
    logger.info("Pipeline complete.")
