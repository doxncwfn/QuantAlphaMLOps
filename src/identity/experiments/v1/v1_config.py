"""Configuration settings for Candidate Security Identity Resolution."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Input paths
SPELLS_CSV_PATH = DATA_DIR / "universe" / "spells.csv"
YAHOO_GAP_PARQUET_PATH = DATA_DIR / "quality" / "yahoo_gap_validation.parquet"
SPELL_QUALITY_FLAGS_PATH = DATA_DIR / "quality" / "spell_quality_flags.parquet"

# Cache paths
OPENFIGI_CACHE_DIR = DATA_DIR / "raw" / "openfigi"
SEC_CACHE_DIR = DATA_DIR / "raw" / "sec"
MASSIVE_CACHE_DIR = DATA_DIR / "raw" / "massive" / "reference_cache"

# Output paths (Deliverables)
IDENTITY_DIR = DATA_DIR / "identity"
QUALITY_DIR = DATA_DIR / "quality"

SECURITY_MASTER_PARQUET = IDENTITY_DIR / "security_master.parquet"
SECURITY_MASTER_CSV = IDENTITY_DIR / "security_master.csv"

TICKER_HISTORY_PARQUET = IDENTITY_DIR / "ticker_history.parquet"
TICKER_HISTORY_CSV = IDENTITY_DIR / "ticker_history.csv"

IDENTITY_EVIDENCE_PARQUET = IDENTITY_DIR / "identity_evidence.parquet"
IDENTITY_EVIDENCE_CSV = IDENTITY_DIR / "identity_evidence.csv"

IDENTITY_CONFLICTS_PARQUET = IDENTITY_DIR / "identity_conflicts.parquet"
IDENTITY_CONFLICTS_CSV = IDENTITY_DIR / "identity_conflicts.csv"

IDENTITY_QUALITY_PARQUET = QUALITY_DIR / "identity_quality.parquet"
IDENTITY_QUALITY_CSV = QUALITY_DIR / "identity_quality.csv"

CONFIG_JSON_PATH = QUALITY_DIR / "identity_resolution_config.json"
REPORT_MD_PATH = QUALITY_DIR / "identity_resolution_report.md"
LOG_FILE_PATH = LOGS_DIR / "identity_resolution.log"

# API credentials
OPENFIGI_API_KEY = (
    os.getenv("OPENFIGI_API_KEY") or "1034b5e4-02da-440e-8ad2-dc6da71af45b"
)
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY") or "IbC9qw1ouX7vSkiyYpGVaDk9jCrk2t_K"
SEC_USER_AGENT = (
    os.getenv("SEC_USER_AGENT") or "HCMUT QuantResearch project_admin@hcmut.edu.vn"
)

# Endpoints
OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
SEC_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_TICKERS_MF_URL = "https://www.sec.gov/files/company_tickers_mf.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
MASSIVE_REFERENCE_URL = "https://api.massive.com/v3/reference/tickers"

# Operational Constants
OPENFIGI_BATCH_SIZE = 100
OPENFIGI_RATE_DELAY_SECONDS = 0.35
SEC_RATE_DELAY_SECONDS = 0.12
MASSIVE_RATE_DELAY_SECONDS = 12.2
REQUEST_TIMEOUT_SECONDS = 15.0
MAX_RETRIES = 5


@dataclass
class IdentityResolutionConfig:
    version: str = "1.0.0"
    openfigi_batch_size: int = OPENFIGI_BATCH_SIZE
    openfigi_rate_delay_seconds: float = OPENFIGI_RATE_DELAY_SECONDS
    sec_rate_delay_seconds: float = SEC_RATE_DELAY_SECONDS
    massive_rate_delay_seconds: float = MASSIVE_RATE_DELAY_SECONDS
    request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS
    max_retries: int = MAX_RETRIES

    confidence_rules: dict = None
    reuse_rules: dict = None

    def __post_init__(self):
        if self.confidence_rules is None:
            self.confidence_rules = {
                "HIGH": "Multiple independent sources agree on security/share-class, or point-in-time Massive metadata directly matches OpenFIGI/SEC shareClassFIGI and CIK.",
                "MEDIUM": "Strong agreement between ticker, exchange, and CIK/name, but share-class FIGI is derived from single current source without historical corroboration.",
                "LOW": "Ticker/name match only, or current metadata applied without temporal consistency.",
                "UNRESOLVED": "Conflicting identities, confirmed ticker reuse without disambiguation, or zero coverage across all sources.",
                "REJECTED": "Candidate identity proven incompatible with historical observation.",
            }
        if self.reuse_rules is None:
            self.reuse_rules = {
                "CONFIRMED_REUSE": "Same ticker maps to distinct CIKs or distinct share_class_figi across spells.",
                "LIKELY_REUSE": "Multi-year gap (>252 sessions) with evidence of corporate dissolution/IPO.",
                "NO_EVIDENCE_OF_REUSE": "Identical security identity preserved across all spells.",
                "UNCERTAIN": "Multi-spell ticker with large gap where early spell identity is ambiguous.",
            }

    def save_json(self, path: Path | str = CONFIG_JSON_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
