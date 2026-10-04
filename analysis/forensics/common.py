"""Common paths, directory management, and logging for forensic analysis."""

from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
WRDS_DIR = DATA_DIR / "WRDS"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = PROJECT_ROOT / "log"

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
OUTPUTS_DIR = ANALYSIS_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
EXAMPLES_DIR = OUTPUTS_DIR / "examples"


def ensure_directories() -> None:
    """Ensure all required output and log directories exist."""
    for d in [LOG_DIR, OUTPUTS_DIR, FIGURES_DIR, METRICS_DIR, EXAMPLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a standard logger with unified formatting."""
    ensure_directories()
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class LogWriter:
    """Convenience file logger that buffers lines and writes atomically."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.lines: list[str] = []

    def write(self, s: str) -> None:
        if s.endswith("\n"):
            s = s[:-1]
        self.lines.append(s)

    def flush(self) -> None:
        self.file_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")

    def close(self) -> None:
        self.flush()

