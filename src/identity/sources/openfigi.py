"""OpenFIGI API client with batched mapping and persistent caching."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import polars as pl
import requests

from src.common.config import (
    OPENFIGI_API_KEY,
    OPENFIGI_BATCH_SIZE,
    OPENFIGI_CACHE_DIR,
    OPENFIGI_RATE_DELAY_SECONDS,
    OPENFIGI_URL,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class OpenFigiClient:
    def __init__(
        self, cache_dir: Path = OPENFIGI_CACHE_DIR, api_key: str = OPENFIGI_API_KEY
    ):
        self.cache_dir = cache_dir
        self.api_key = api_key
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "openfigi_cache.parquet"

        self.cache: dict[str, list[dict[str, Any]]] = {}
        self._load_cache()

    def _load_cache(self):
        """Loads cached mapping results if available."""
        if self.cache_file.exists():
            try:
                df = pl.read_parquet(self.cache_file)
                for row in df.iter_rows(named=True):
                    tk = row["query_ticker"]
                    raw_json = row["raw_results_json"]
                    matches = json.loads(raw_json) if raw_json else []
                    self.cache[tk] = matches
                logger.info(
                    "Loaded %d cached OpenFIGI records from %s",
                    len(self.cache),
                    self.cache_file,
                )
            except (
                OSError,
                pl.exceptions.PolarsError,
                json.JSONDecodeError,
                KeyError,
                ValueError,
            ) as e:
                logger.error("Failed loading OpenFIGI cache: %s", e)

    def save_cache(self):
        """Saves current cache to Parquet atomically."""
        if not self.cache:
            return
        records = []
        for tk, matches in self.cache.items():
            top = matches[0] if matches else {}
            records.append(
                {
                    "query_ticker": tk,
                    "n_matches": len(matches),
                    "top_share_class_figi": top.get("shareClassFIGI"),
                    "top_composite_figi": top.get("compositeFIGI"),
                    "top_name": top.get("name"),
                    "top_security_type": top.get("securityType"),
                    "top_market_sector": top.get("marketSector"),
                    "top_exch_code": top.get("exchCode"),
                    "raw_results_json": json.dumps(matches),
                }
            )
        df = pl.DataFrame(records)
        tmp_path = self.cache_file.with_suffix(".tmp")
        df.write_parquet(tmp_path)
        tmp_path.replace(self.cache_file)
        logger.info(
            "Persisted %d OpenFIGI records to %s", len(self.cache), self.cache_file
        )

    def lookup_cached(self, ticker: str) -> list[dict[str, Any]] | None:
        """Returns cached match list if ticker has already been queried."""
        tk_clean = ticker.strip().upper()
        return self.cache.get(tk_clean)

    def batch_query(
        self, tickers: list[str], max_workers: int = 1
    ) -> dict[str, list[dict[str, Any]]]:
        """Queries OpenFIGI in batches of up to 100 tickers with rate-limit handling."""
        unique_tickers = list(
            dict.fromkeys(t.strip().upper() for t in tickers if t and str(t).strip())
        )
        missing = [t for t in unique_tickers if t not in self.cache]

        if not missing:
            logger.info(
                "All %d requested tickers are already in OpenFIGI cache.",
                len(unique_tickers),
            )
            return {t: self.cache.get(t, []) for t in unique_tickers}

        logger.info(
            "Querying OpenFIGI for %d missing tickers in batches of %d...",
            len(missing),
            OPENFIGI_BATCH_SIZE,
        )

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["X-OPENFIGI-APIKEY"] = self.api_key

        session = requests.Session()
        total_batches = (len(missing) + OPENFIGI_BATCH_SIZE - 1) // OPENFIGI_BATCH_SIZE

        for b_idx in range(total_batches):
            batch_tickers = missing[
                b_idx * OPENFIGI_BATCH_SIZE : (b_idx + 1) * OPENFIGI_BATCH_SIZE
            ]
            # Primary US query with exchCode='US'
            payload = [
                {"idType": "TICKER", "idValue": tk, "exchCode": "US"}
                for tk in batch_tickers
            ]

            retries = 0
            while retries < 5:
                try:
                    time.sleep(OPENFIGI_RATE_DELAY_SECONDS)
                    resp = session.post(
                        OPENFIGI_URL,
                        json=payload,
                        headers=headers,
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )

                    if resp.status_code == 429:
                        retries += 1
                        wait_sec = 2.0 * retries
                        logger.warning(
                            "OpenFIGI 429 rate limit. Backing off %.1fs (retry %d/5)...",
                            wait_sec,
                            retries,
                        )
                        time.sleep(wait_sec)
                        continue

                    resp.raise_for_status()
                    res_json = resp.json()

                    for i, tk in enumerate(batch_tickers):
                        matches = []
                        if i < len(res_json):
                            matches = res_json[i].get("data", [])
                        self.cache[tk] = matches

                    break
                except (
                    requests.RequestException,
                    json.JSONDecodeError,
                    ValueError,
                    KeyError,
                    IndexError,
                ) as exc:
                    retries += 1
                    logger.warning(
                        "Error querying OpenFIGI batch %d/%d (retry %d/5): %s",
                        b_idx + 1,
                        total_batches,
                        retries,
                        exc,
                    )
                    time.sleep(2.0 * retries)
                    if retries >= 5:
                        for tk in batch_tickers:
                            if tk not in self.cache:
                                self.cache[tk] = []

            # Save periodically every 10 batches
            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == total_batches:
                logger.info(
                    "OpenFIGI progress: %d / %d batches completed (%.1f%%).",
                    b_idx + 1,
                    total_batches,
                    (b_idx + 1) / total_batches * 100,
                )
                self.save_cache()

        return {t: self.cache.get(t, []) for t in unique_tickers}
