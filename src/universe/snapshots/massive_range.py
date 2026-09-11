"""
massive_test.py
===============
Restartable historical point-in-time active-ticker extraction pipeline.

Retrieves the active US stock ticker universe for every NYSE trading session in a configurable date range using the Massive API, then writes out:

  ./data/raw/massive/active_tickers/YYYY/YYYY-MM-DD.json
      Per-date raw snapshot (atomic, only after complete extraction)

  ./data/processed/universe/active_tickers/year=YYYY/data.parquet
      Year-partitioned, long-form dataset (date, ticker). Only the affected partition is updated per date processed.

  ./data/manifests/active_tickers_manifest.parquet
      Manifest of run status: date / status / ticker_count / processed_at / error.

  ./logs/massive_active_tickers.log
      Rotating log file (10 MB x 5 backups).

Restartability
--------------
  - Dates marked "success" in the manifest are skipped.
  - Dates marked "failed" or missing are (re-)attempted.
  - A date is marked successful only if:
      1. API extraction (all pages, no errors) completed.
      2. Raw JSON snapshot written atomically.
      3. Year-partition Parquet written atomically.

Dependencies (all required)
---------------------------
  exchange-calendars  -- XNYS NYSE session calendar
  polars              -- tabular I/O and Parquet
  pyarrow             -- Parquet backend for polars
  requests            -- HTTP
  python-dotenv       -- environment variables

If missing, install exchange-calendars:
  uv pip install exchange-calendars

Usage:
  python src/ingestion/massive_test.py

Configure START_DATE / END_DATE near the bottom of this file.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from datetime import date, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

try:
    import exchange_calendars as xcals
except ImportError:
    print(
        "ERROR: exchange-calendars is required for this pipeline.\n"
        "Install it with:  uv pip install exchange-calendars\n"
        "Then retry.",
        file=sys.stderr,
    )
    sys.exit(1)

START_DATE: str = "2020-01-01"
END_DATE: str = "2020-02-01"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = _REPO_ROOT / "data" / "raw" / "massive" / "active_tickers"
DATA_PROC_DIR = _REPO_ROOT / "data" / "processed" / "universe" / "active_tickers"
MANIFEST_FILE = _REPO_ROOT / "data" / "manifests" / "active_tickers_manifest.parquet"
LOG_DIR = _REPO_ROOT / "logs"
LOG_FILE = LOG_DIR / "massive_active_tickers.log"

LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5

MASSIVE_API_BASE = "https://api.massive.com/v3/reference/tickers"
PAGE_SIZE = 1000
INTER_PAGE_DELAY = 12.0  # 5 req/min rate limit
INITIAL_BACKOFF = 3.0  # Backoff for HTTP 429
MAX_BACKOFF = 30.0
MAX_REQUEST_RETRIES = 10


def setup_logging() -> logging.Logger:
    """Logger: DEBUG+ to file, INFO+ to stdout."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("massive_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_valid_api_key(cli_arg_key: str | None = None) -> str:
    api_key = (
        cli_arg_key or os.getenv("MASSIVE_API_KEY") or os.environ.get("MASSIVE_API_KEY")
    )
    if not api_key or api_key.lower().startswith("your_"):
        raise ValueError(
            "Valid Massive API Key must be supplied via argument or MASSIVE_API_KEY env/.env."
        )
    return api_key


def get_xnys_trading_dates(start: str, end: str, logger: logging.Logger) -> list[str]:
    """Return NYSE sessions using authoritative calendar, excluding weekends/holidays/ad-hoc closures."""
    logger.info("Initialising XNYS (NYSE) calendar via exchange_calendars …")
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(start, end)
    dates = [s.strftime("%Y-%m-%d") for s in sessions]
    logger.info(
        "XNYS calendar: %d trading sessions between %s and %s.", len(dates), start, end
    )
    return dates


def ensure_directories(logger: logging.Logger) -> None:
    for d in [DATA_RAW_DIR, DATA_PROC_DIR, MANIFEST_FILE.parent, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory: %s", d)


_MANIFEST_SCHEMA: dict = {
    "date": pl.Date,
    "status": pl.Utf8,
    "ticker_count": pl.Int64,
    "processed_at": pl.Utf8,
    "error": pl.Utf8,
}


def load_manifest(logger: logging.Logger) -> pl.DataFrame:
    """Load manifest, or empty DataFrame if missing. Abort if exists but unreadable."""
    if not MANIFEST_FILE.exists():
        logger.info("No existing manifest found -- starting fresh.")
        return pl.DataFrame(
            {k: pl.Series([], dtype=v) for k, v in _MANIFEST_SCHEMA.items()}
        )
    try:
        df = pl.read_parquet(MANIFEST_FILE)
        logger.debug("Manifest loaded: %d rows from %s", len(df), MANIFEST_FILE)
        return df
    except (OSError, pl.exceptions.PolarsError, RuntimeError, ValueError) as exc:
        logger.critical(
            "Manifest file exists at %s but cannot be read: %s\n"
            "Aborting to protect restartability. "
            "Repair or delete the manifest file manually, then retry.",
            MANIFEST_FILE,
            exc,
        )
        sys.exit(1)


def save_manifest(df: pl.DataFrame, logger: logging.Logger) -> None:
    """Atomic manifest write (parquet, temp file then replace)."""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_FILE.with_suffix(".parquet.tmp")
    try:
        df.write_parquet(tmp)
        tmp.replace(MANIFEST_FILE)
        logger.debug("Manifest saved (%d rows).", len(df))
    except Exception as exc:
        logger.error("Failed to save manifest: %s", exc)
        tmp.unlink(missing_ok=True)
        raise


def upsert_manifest_row(
    manifest: pl.DataFrame,
    target_date: str,
    status: str,
    ticker_count: int,
    processed_at: str,
    error: str | None,
    logger: logging.Logger,
) -> pl.DataFrame:
    date_val = date.fromisoformat(target_date)
    new_row = pl.DataFrame(
        {
            "date": [date_val],
            "status": [status],
            "ticker_count": [ticker_count],
            "processed_at": [processed_at],
            "error": [error],
        },
        schema=_MANIFEST_SCHEMA,
    )
    updated = manifest.filter(pl.col("date") != pl.lit(date_val).cast(pl.Date))
    updated = pl.concat([updated, new_row])
    logger.debug(
        "Manifest upsert: date=%s status=%s count=%d", target_date, status, ticker_count
    )
    return updated


def is_date_already_successful(manifest: pl.DataFrame, target_date: str) -> bool:
    date_val = date.fromisoformat(target_date)
    return (
        len(
            manifest.filter(
                (pl.col("date") == pl.lit(date_val).cast(pl.Date))
                & (pl.col("status") == pl.lit("success"))
            )
        )
        > 0
    )


def raw_snapshot_path(target_date: str) -> Path:
    return DATA_RAW_DIR / target_date[:4] / f"{target_date}.json"


def save_raw_snapshot(
    target_date: str, tickers: list[str], logger: logging.Logger
) -> None:
    """Atomic write of per-date raw JSON (temp file, then replace)."""
    path = raw_snapshot_path(target_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"date": target_date, "tickers": tickers, "ticker_count": len(tickers)}
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        Path(tmp_name).replace(path)
        logger.info("Raw snapshot saved: %s  (%d tickers)", path, len(tickers))
    except Exception as exc:
        logger.error("Failed to write raw snapshot for %s: %s", target_date, exc)
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def year_partition_path(year: int) -> Path:
    return DATA_PROC_DIR / f"year={year}" / "data.parquet"


def update_year_partition(
    target_date: str, tickers: list[str], logger: logging.Logger
) -> None:
    """
    Update only the year partition containing target_date. Other partitions remain untouched.
    For this date, remove existing rows, then append and deduplicate.
    """
    date_val = date.fromisoformat(target_date)
    year = date_val.year
    part_path = year_partition_path(year)
    part_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pl.DataFrame(
        {"date": [date_val] * len(tickers), "ticker": tickers},
        schema={"date": pl.Date, "ticker": pl.Utf8},
    )

    if part_path.exists():
        try:
            existing = pl.read_parquet(part_path)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot read existing year-{year} partition at {part_path}: {exc}. "
                "Aborting to protect historical data. Fix or remove the corrupt file manually."
            ) from exc
        existing = existing.filter(pl.col("date") != pl.lit(date_val).cast(pl.Date))
        combined = pl.concat([existing, new_df])
    else:
        combined = new_df

    combined = combined.unique(subset=["date", "ticker"]).sort(["date", "ticker"])

    tmp_path = part_path.with_suffix(".parquet.tmp")
    try:
        combined.write_parquet(tmp_path)
        tmp_path.replace(part_path)
        logger.info(
            "Year-%d partition updated: %s  (rows in partition: %d)",
            year,
            part_path,
            len(combined),
        )
    except Exception as exc:
        logger.error("Failed to write year-%d partition: %s", year, exc)
        tmp_path.unlink(missing_ok=True)
        raise


class ExtractionError(Exception):
    """Raised when the Massive API extraction for a date cannot be completed."""


class MassiveThrottledManager:
    """
    Throttled, paginated Massive API client. Strict fail/skip semantics:
    If any step fails, no output or manifest 'success' is produced for that date.
    """

    def __init__(self, api_key: str, logger: logging.Logger) -> None:
        self.api_key = api_key
        self.logger = logger

    def get_active_symbols_on_date(self, target_date: str) -> list[str]:
        """
        Retrieve all active US stock tickers for target_date.

        - Returns only after paginating all results.
        - Raises ExtractionError for any unrecoverable error so partial results do not propagate.
        - Respects rate limits and does exponential backoff (on 429).
        """
        active_symbols: list[str] = []
        seen: set[str] = set()

        current_url: str | None = MASSIVE_API_BASE
        base_params: dict = {
            "date": target_date,
            "active": "true",
            "market": "stocks",
            "limit": PAGE_SIZE,
            "apiKey": self.api_key,
        }
        backoff_delay = INITIAL_BACKOFF
        page_num = 0

        self.logger.info("Starting API extraction for date: %s", target_date)

        while current_url:
            page_num += 1
            request_params = base_params if page_num == 1 else {"apiKey": self.api_key}
            self.logger.debug("[%s] Requesting page %d.", target_date, page_num)

            backoff_ref = [backoff_delay]
            page_tickers, current_url = self._fetch_page(
                target_date=target_date,
                page_num=page_num,
                url=current_url,
                params=request_params,
                seen=seen,
                active_symbols=active_symbols,
                backoff_delay_ref=backoff_ref,
            )
            backoff_delay = backoff_ref[0]

            if current_url:
                self.logger.debug(
                    "[%s] Waiting %.1fs before page %d.",
                    target_date,
                    INTER_PAGE_DELAY,
                    page_num + 1,
                )
                time.sleep(INTER_PAGE_DELAY)

        self.logger.info(
            "[%s] Complete: %d unique stock tickers across %d page(s).",
            target_date,
            len(active_symbols),
            page_num,
        )

        if not active_symbols:
            raise ExtractionError(
                f"Massive returned zero active stock tickers for {target_date}. "
                "This is unexpected for a valid NYSE trading session. "
                "Check the API response or the date validity."
            )

        return active_symbols

    def _fetch_page(
        self,
        target_date: str,
        page_num: int,
        url: str,
        params: dict,
        seen: set[str],
        active_symbols: list[str],
        backoff_delay_ref: list[float],
    ) -> tuple[int, str | None]:
        """Fetch a single API page with retry/backoff logic."""
        retries = 0
        while True:
            try:
                self.logger.debug(
                    "[%s] HTTP GET page %d (attempt %d/%d).",
                    target_date,
                    page_num,
                    retries + 1,
                    MAX_REQUEST_RETRIES,
                )
                response = requests.get(url, params=params, timeout=15)

                if response.status_code == 429:
                    retries += 1
                    if retries >= MAX_REQUEST_RETRIES:
                        raise ExtractionError(
                            f"[{target_date}] Page {page_num} received HTTP 429 "
                            f"{MAX_REQUEST_RETRIES} times in a row -- giving up."
                        )
                    delay = backoff_delay_ref[0]
                    self.logger.warning(
                        "[%s] HTTP 429 on page %d (attempt %d/%d) -- "
                        "backing off %.1fs (collected %d tickers so far).",
                        target_date,
                        page_num,
                        retries,
                        MAX_REQUEST_RETRIES,
                        delay,
                        len(active_symbols),
                    )
                    time.sleep(delay)
                    backoff_delay_ref[0] = min(delay * 1.5, MAX_BACKOFF)
                    continue

                content_type = response.headers.get("Content-Type", "")
                if not response.text or not content_type.startswith("application/json"):
                    raise ExtractionError(
                        f"[{target_date}] Page {page_num} returned non-JSON "
                        f"(status={response.status_code}, "
                        f"content-type='{content_type}'). "
                        f"Snippet: {response.text[:300]!r}"
                    )

                response.raise_for_status()
                payload = response.json()

                if not isinstance(payload, dict):
                    raise ExtractionError(
                        f"[{target_date}] Page {page_num} returned unexpected "
                        f"payload type: {type(payload).__name__}"
                    )

                backoff_delay_ref[0] = INITIAL_BACKOFF

                results = payload.get("results", [])
                if not results:
                    self.logger.debug(
                        "[%s] Page %d: empty results -- pagination complete.",
                        target_date,
                        page_num,
                    )
                    return 0, None

                page_tickers = 0
                for idx, item in enumerate(results):
                    if not isinstance(item, dict):
                        raise ExtractionError(
                            f"[{target_date}] Page {page_num}: result[{idx}] is not a "
                            f"dict (got {type(item).__name__!r}). "
                            "Aborting to prevent partial universe."
                        )
                    raw_ticker = item.get("ticker")
                    if not raw_ticker or not str(raw_ticker).strip():
                        raise ExtractionError(
                            f"[{target_date}] Page {page_num}: result[{idx}] has a "
                            f"missing or empty 'ticker' field (value={raw_ticker!r}). "
                            "Aborting to prevent partial universe."
                        )
                    t = str(raw_ticker).strip()
                    if t not in seen:
                        seen.add(t)
                        active_symbols.append(t)
                        page_tickers += 1

                self.logger.debug(
                    "[%s] Page %d: +%d tickers (running total: %d).",
                    target_date,
                    page_num,
                    page_tickers,
                    len(active_symbols),
                )

                next_url = payload.get("next_url") or None
                return page_tickers, next_url

            except ExtractionError:
                raise

            except requests.exceptions.RequestException as exc:
                retries += 1
                if retries >= MAX_REQUEST_RETRIES:
                    raise ExtractionError(
                        f"[{target_date}] Page {page_num} failed after "
                        f"{MAX_REQUEST_RETRIES} retries: {exc}"
                    ) from exc
                wait = 5.0 * retries
                self.logger.warning(
                    "[%s] Network error on page %d (attempt %d/%d): %s. "
                    "Retrying in %.1fs.",
                    target_date,
                    page_num,
                    retries,
                    MAX_REQUEST_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)

            except Exception as exc:
                retries += 1
                if retries >= MAX_REQUEST_RETRIES:
                    raise ExtractionError(
                        f"[{target_date}] Page {page_num} failed after "
                        f"{MAX_REQUEST_RETRIES} retries with unexpected error: {exc}"
                    ) from exc
                wait = 5.0 * retries
                self.logger.error(
                    "[%s] Unexpected error on page %d (attempt %d/%d): %s. "
                    "Retrying in %.1fs.",
                    target_date,
                    page_num,
                    retries,
                    MAX_REQUEST_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)


def process_date(
    target_date: str,
    manager: MassiveThrottledManager,
    manifest: pl.DataFrame,
    logger: logging.Logger,
) -> tuple[pl.DataFrame, bool]:
    """
    Process all data for a single trading date:
      1. Fetch all tickers from the API (fails out on any error)
      2. Write raw snapshot (atomic)
      3. Update partitioned Parquet (atomic)
      4. Upsert manifest with SUCCESS

    On any failure, manifest is updated with FAILED status and error message.
    """
    processed_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    logger.info("[%s] Processing …", target_date)

    try:
        tickers = manager.get_active_symbols_on_date(target_date)
        tickers = list(dict.fromkeys(tickers))
        save_raw_snapshot(target_date, tickers, logger)
        update_year_partition(target_date, tickers, logger)
        manifest = upsert_manifest_row(
            manifest, target_date, "success", len(tickers), processed_at, None, logger
        )
        save_manifest(manifest, logger)
        logger.info("[%s] SUCCESS -- %d tickers.", target_date, len(tickers))
        return manifest, True

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.exception("[%s] FAILED: %s", target_date, error_msg)
        manifest = upsert_manifest_row(
            manifest, target_date, "failed", 0, processed_at, error_msg, logger
        )
        try:
            save_manifest(manifest, logger)
        except (OSError, pl.exceptions.PolarsError) as save_exc:
            logger.error(
                "Could not save manifest after failure for %s: %s",
                target_date,
                save_exc,
            )
        return manifest, False


def run_pipeline(start_date: str, end_date: str) -> None:
    """
    Main orchestration for the entire extraction flow:
        - Load API key and configuration
        - Create directories if needed
        - Load or create manifest
        - Generate all NYSE trading days in the window
        - For each date:
            * If already a manifest 'success', skip.
            * Otherwise, extract, snapshot, update parquet, and mark manifest.
        - Summarize at end.
    """
    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("Massive active-ticker pipeline  STARTING")
    logger.info("Start date          : %s", start_date)
    logger.info("End date            : %s", end_date)
    logger.info("Repo root           : %s", _REPO_ROOT)
    logger.info("Raw dir             : %s", DATA_RAW_DIR)
    logger.info("Partitioned dataset : %s", DATA_PROC_DIR)
    logger.info("Manifest            : %s", MANIFEST_FILE)
    logger.info("Log file            : %s", LOG_FILE)
    logger.info("=" * 70)

    load_dotenv()

    try:
        api_key = get_valid_api_key()
    except ValueError as exc:
        logger.critical("API key error: %s", exc)
        sys.exit(1)

    ensure_directories(logger)
    manifest = load_manifest(logger)

    trading_dates = get_xnys_trading_dates(start_date, end_date, logger)
    logger.info("Trading dates in range: %d", len(trading_dates))

    already_done = sum(
        1 for d in trading_dates if is_date_already_successful(manifest, d)
    )
    logger.info(
        "Already completed (will skip): %d / %d", already_done, len(trading_dates)
    )

    manager = MassiveThrottledManager(api_key=api_key, logger=logger)

    n_skipped = 0
    n_success = 0
    n_failed = 0
    failed_dates: list[str] = []

    for i, target_date in enumerate(trading_dates, start=1):
        logger.debug("--- [%d / %d] %s ---", i, len(trading_dates), target_date)

        if is_date_already_successful(manifest, target_date):
            logger.info("[%s] Already successful -- skipping.", target_date)
            n_skipped += 1
            continue

        manifest, success = process_date(target_date, manager, manifest, logger)
        if success:
            n_success += 1
        else:
            n_failed += 1
            failed_dates.append(target_date)

    total_rows = 0
    glob_pattern = str(DATA_PROC_DIR / "**" / "*.parquet")
    try:
        total_rows = (
            pl.scan_parquet(glob_pattern, allow_missing_columns=True)
            .select(pl.len())
            .collect()
            .item()
        )
    except (OSError, pl.exceptions.PolarsError, RuntimeError, ValueError) as exc:
        logger.warning("Could not count total rows in partitioned dataset: %s", exc)

    lines = [
        "",
        "Historical extraction complete.",
        "",
        f"  Date range             : {start_date} -> {end_date}",
        f"  Trading dates found    : {len(trading_dates)}",
        f"  Already completed      : {n_skipped}",
        f"  Successfully processed : {n_success}",
        f"  Failed                 : {n_failed}",
        f"  Total (date, ticker)   : {total_rows:,}",
    ]
    if failed_dates:
        lines += ["", "  Failed dates:"] + [f"    {d}" for d in failed_dates]
    lines.append("")

    for line in lines:
        logger.info(line)
        print(line)


if __name__ == "__main__":
    run_pipeline(start_date=START_DATE, end_date=END_DATE)
