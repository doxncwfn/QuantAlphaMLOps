"""Configuration for Identity Audit and Availability Episodes Construction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Input paths
SPELLS_CSV_PATH = DATA_DIR / "universe" / "spells.csv"
SECURITY_MASTER_PARQUET = DATA_DIR / "identity" / "security_master.parquet"
TICKER_HISTORY_PARQUET = DATA_DIR / "identity" / "ticker_history.parquet"
IDENTITY_EVIDENCE_PARQUET = DATA_DIR / "identity" / "identity_evidence.parquet"
IDENTITY_CONFLICTS_PARQUET = DATA_DIR / "identity" / "identity_conflicts.parquet"
YAHOO_GAP_PARQUET_PATH = DATA_DIR / "quality" / "yahoo_gap_validation.parquet"
MANIFEST_CSV_PATH = DATA_DIR / "universe" / "manifest.csv"

# Output paths
UNIVERSE_DIR = DATA_DIR / "universe"
QUALITY_DIR = DATA_DIR / "quality"

AVAILABILITY_EPISODES_PARQUET = UNIVERSE_DIR / "availability_episodes.parquet"
AVAILABILITY_EPISODES_CSV = UNIVERSE_DIR / "availability_episodes.csv"

EXPECTED_SECURITY_DATES_PARQUET = UNIVERSE_DIR / "expected_security_dates.parquet"

AVAILABILITY_EPISODE_QUALITY_PARQUET = (
    QUALITY_DIR / "availability_episode_quality.parquet"
)
AVAILABILITY_EPISODE_QUALITY_CSV = QUALITY_DIR / "availability_episode_quality.csv"

MANUAL_REVIEW_QUEUE_PARQUET = QUALITY_DIR / "identity_manual_review_queue.parquet"
MANUAL_REVIEW_QUEUE_CSV = QUALITY_DIR / "identity_manual_review_queue.csv"

CONFIG_JSON_PATH = QUALITY_DIR / "availability_episode_config.json"
REPORT_MD_PATH = QUALITY_DIR / "availability_episode_report.md"
LOG_FILE_PATH = LOGS_DIR / "availability_episode_audit.log"

# Rules and Thresholds
SHORT_GAP_MAX_SESSIONS = 2
LONG_GAP_MIN_SESSIONS = 252
CORRUPTED_DATES: list[str] = ["2009-10-29", "2010-03-30", "2010-03-31"]
START_DATE = "2004-01-02"
END_DATE = "2026-09-01"


@dataclass
class AvailabilityEpisodeConfig:
    version: str = "1.0.0"
    short_gap_max_sessions: int = SHORT_GAP_MAX_SESSIONS
    long_gap_min_sessions: int = LONG_GAP_MIN_SESSIONS
    corrupted_dates: list[str] = None
    start_date: str = START_DATE
    end_date: str = END_DATE

    def __post_init__(self):
        if self.corrupted_dates is None:
            self.corrupted_dates = list(CORRUPTED_DATES)

    def save_json(self, path: Path | str = CONFIG_JSON_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
