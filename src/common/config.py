"""
Common Central Configuration and Directory Layout.
==================================================
Defines repository paths, candidate output directories, 9-key API parameters,
rate limits, timeouts, cache paths, and expected baseline data hashes.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository Root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = REPO_ROOT / "src" / ".env"

# Inputs (Strictly Immutable)
SPELLS_CSV_PATH = REPO_ROOT / "data" / "universe" / "spells.csv"
TRADING_SESSIONS_PATH = REPO_ROOT / "data" / "identity" / "experiments" / "v2" / "trading_sessions.parquet"
INPUT_MANIFEST_JSON = REPO_ROOT / "data" / "manifests" / "v3_input_manifest.json"

# Reference Data Sources (Local Caches)
SEC_CACHE_JSON = REPO_ROOT / "data" / "identity" / "experiments" / "resolver_v1" / "api_cache" / "sec" / "company_tickers_exchange.json"
OPENFIGI_CACHE_PARQUET = REPO_ROOT / "data" / "raw" / "openfigi" / "openfigi_cache.parquet"

# Massive Caches
MASSIVE_V3_CACHE_DIR = REPO_ROOT / "data" / "identity" / "cache" / "massive"
MASSIVE_V2_CACHE_DIR = REPO_ROOT / "data" / "identity" / "experiments" / "v2" / "cache" / "massive"
MASSIVE_V1_CACHE_DIR = REPO_ROOT / "data" / "identity" / "experiments" / "resolver_v1" / "api_cache" / "massive"
MASSIVE_RAW_CACHE_DIR = REPO_ROOT / "data" / "identity" / "experiments" / "raw_massive"
MASSIVE_REF_CACHE_DIR = REPO_ROOT / "data" / "raw" / "massive" / "reference_cache"
ALL_CACHE_DIRS = [
    MASSIVE_V3_CACHE_DIR,
    MASSIVE_V2_CACHE_DIR,
    MASSIVE_V1_CACHE_DIR,
    MASSIVE_RAW_CACHE_DIR,
    MASSIVE_REF_CACHE_DIR,
]

# Candidate Output Directories (Zero overwrite of production files)
CANDIDATES_IDENTITY_DIR = REPO_ROOT / "data" / "identity" / "candidates" / "v3"
CANDIDATES_UNIVERSE_DIR = REPO_ROOT / "data" / "universe" / "candidates" / "v3"
QUALITY_DIR = REPO_ROOT / "data" / "quality" / "v3"
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests" / "v3"
LOG_DIR = REPO_ROOT / "log"

# API Settings
MASSIVE_REFERENCE_URL = "https://api.polygon.io/v3/reference/tickers"
PER_KEY_INTERVAL_SECONDS = 12.1  # 5 req/min per key
REQUEST_TIMEOUT_SECONDS = 15.0
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0

# Expected Baseline Values
EXPECTED_SPELLS_HASH = "5fc79a37cdc341cf7b10a7017ccd75cf7001aa1b91e5dd6501c829b746191bf1"
EXPECTED_SPELLS_ROWS = 43757
EXPECTED_UNIQUE_TICKERS = 36843
EXPECTED_TIMELINE_SESSIONS = 5699
EXCLUDED_COMPROMISED_DATES = ["2009-10-29", "2010-03-30", "2010-03-31"]


def ensure_directories():
    """Ensure all runtime working directories exist."""
    for p in [
        MASSIVE_V3_CACHE_DIR,
        CANDIDATES_IDENTITY_DIR,
        CANDIDATES_UNIVERSE_DIR,
        QUALITY_DIR,
        MANIFESTS_DIR,
        LOG_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


ensure_directories()
