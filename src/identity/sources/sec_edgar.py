"""SEC EDGAR data ingestor and lookup manager."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from src.common.config import (
    SEC_CACHE_DIR,
    SEC_RATE_DELAY_SECONDS,
    SEC_SUBMISSIONS_URL_TEMPLATE,
    SEC_TICKERS_EXCHANGE_URL,
    SEC_TICKERS_MF_URL,
    SEC_USER_AGENT,
)

logger = logging.getLogger(__name__)


class SecEdgarClient:
    def __init__(self, cache_dir: Path = SEC_CACHE_DIR):
        self.cache_dir = cache_dir
        self.submissions_cache_dir = self.cache_dir / "submissions"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": SEC_USER_AGENT})
        
        self.exchange_tickers: Dict[str, Dict[str, Any]] = {}
        self.mf_tickers: Dict[str, Dict[str, Any]] = {}
        self._load_or_fetch_bulk_tables()

    def _load_or_fetch_bulk_tables(self):
        """Loads or downloads the SEC bulk ticker tables."""
        # 1. Company tickers exchange
        exch_file = self.cache_dir / "company_tickers_exchange.json"
        if not exch_file.exists():
            logger.info("Downloading SEC company_tickers_exchange.json from %s...", SEC_TICKERS_EXCHANGE_URL)
            try:
                resp = self.session.get(SEC_TICKERS_EXCHANGE_URL, timeout=15)
                resp.raise_for_status()
                with open(exch_file, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                logger.info("Cached SEC exchange tickers table.")
            except Exception as e:
                logger.error("Failed downloading SEC exchange tickers: %s", e)

        if exch_file.exists():
            try:
                with open(exch_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                fields = data.get("fields", [])
                rows = data.get("data", [])
                for row in rows:
                    rec = dict(zip(fields, row))
                    tk = str(rec.get("ticker", "")).strip().upper()
                    if tk:
                        cik_str = str(rec.get("cik", "")).zfill(10)
                        self.exchange_tickers[tk] = {
                            "cik": cik_str,
                            "name": rec.get("name"),
                            "ticker": tk,
                            "exchange": rec.get("exchange"),
                            "source": "SEC_EXCHANGE_TICKERS"
                        }
                logger.info("Loaded %d SEC exchange tickers into memory.", len(self.exchange_tickers))
            except Exception as e:
                logger.error("Failed parsing SEC exchange tickers: %s", e)

        # 2. Mutual funds / ETFs
        mf_file = self.cache_dir / "company_tickers_mf.json"
        if not mf_file.exists():
            logger.info("Downloading SEC company_tickers_mf.json from %s...", SEC_TICKERS_MF_URL)
            try:
                resp = self.session.get(SEC_TICKERS_MF_URL, timeout=15)
                resp.raise_for_status()
                with open(mf_file, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                logger.info("Cached SEC mutual funds/ETFs table.")
            except Exception as e:
                logger.error("Failed downloading SEC mutual funds/ETFs: %s", e)

        if mf_file.exists():
            try:
                with open(mf_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                fields = data.get("fields", [])
                rows = data.get("data", [])
                for row in rows:
                    rec = dict(zip(fields, row))
                    sym = str(rec.get("symbol", "")).strip().upper()
                    if sym and sym not in self.mf_tickers:
                        cik_str = str(rec.get("cik", "")).zfill(10)
                        self.mf_tickers[sym] = {
                            "cik": cik_str,
                            "series_id": rec.get("seriesId"),
                            "class_id": rec.get("classId"),
                            "symbol": sym,
                            "source": "SEC_MF_TICKERS"
                        }
                logger.info("Loaded %d SEC mutual fund/ETF symbols into memory.", len(self.mf_tickers))
            except Exception as e:
                logger.error("Failed parsing SEC mutual funds/ETFs: %s", e)

    def lookup_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fast in-memory lookup across SEC bulk tables."""
        tk_clean = ticker.strip().upper()
        if tk_clean in self.exchange_tickers:
            return self.exchange_tickers[tk_clean]
        if tk_clean in self.mf_tickers:
            return self.mf_tickers[tk_clean]
        
        # Check alternative punctuation (e.g. '.' vs '-')
        tk_alt = tk_clean.replace(".", "-")
        if tk_alt in self.exchange_tickers:
            return self.exchange_tickers[tk_alt]
        tk_alt2 = tk_clean.replace("-", ".")
        if tk_alt2 in self.exchange_tickers:
            return self.exchange_tickers[tk_alt2]

        return None

    def get_submissions(self, cik: str) -> Optional[Dict[str, Any]]:
        """Retrieves and caches SEC EDGAR submissions JSON for a CIK."""
        cik_clean = str(cik).strip().zfill(10)
        cache_file = self.submissions_cache_dir / f"CIK{cik_clean}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        url = SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=cik_clean)
        try:
            time.sleep(SEC_RATE_DELAY_SECONDS)
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                payload = resp.json()
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                return payload
            elif resp.status_code == 404:
                return None
            else:
                logger.warning("SEC submissions query for CIK %s returned HTTP %d", cik_clean, resp.status_code)
                return None
        except Exception as exc:
            logger.warning("Error fetching SEC submissions for CIK %s: %s", cik_clean, exc)
            return None
