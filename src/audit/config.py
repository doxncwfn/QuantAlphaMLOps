"""Configuration loader and path resolution for the data quality audit."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Resolve project root (repository root containing src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_default_config_path() -> Path:
    """Find the default audit config YAML path across common candidate locations."""
    candidates = [
        PROJECT_ROOT / "config" / "audit.yaml",
        PROJECT_ROOT / "configs" / "audit.yaml",
        PROJECT_ROOT / "report" / "quality" / "audit_config.yaml",
        Path("config/audit.yaml"),
        Path("configs/audit.yaml"),
        Path("report/quality/audit_config.yaml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return PROJECT_ROOT / "config" / "audit.yaml"


def load_audit_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """
    Load audit configuration from YAML file using PyYAML.

    Args:
        config_path: Optional explicit path to audit YAML configuration.

    Returns:
        Structured audit configuration dictionary.
    """
    if config_path is None:
        target_path = find_default_config_path()
    else:
        target_path = Path(config_path)
        if (
            not target_path.is_absolute()
            and not target_path.exists()
            and (PROJECT_ROOT / target_path).exists()
        ):
            target_path = PROJECT_ROOT / target_path

    if not target_path.exists():
        logger.warning(
            "Audit configuration file %s not found. Using fallback defaults.", target_path
        )
        return {
            "version": "1.0.0",
            "audit_name": "Russell 1000 Historical Dataset Audit",
            "paths": {
                "data_dir": "data",
                "processed_dir": "data/processed",
                "raw_dir": "data/raw",
                "wrds_dir": "data/WRDS",
                "russell1000_dir": "data/Russell1000",
                "factors_file": "data/ff.csv",
                "output_tables_dir": "report/quality/tables",
                "output_figures_dir": "report/quality/figures",
                "log_dir": "log/audit",
                "manifest_path": "report/quality/audit_input_manifest.json",
                "report_path": "report/quality/russell1000_data_quality_audit_report.md",
                "notebook_path": "notebooks/visualizations.ipynb",
            },
            "years": {
                "start_year": 2000,
                "end_year": 2026,
                "missing_years": [],
            },
            "parameters": {
                "lookback_days": 40,
                "coverage_threshold": 0.90,
                "high_coverage_threshold": 0.95,
                "missing_streak_thresholds": [5, 10, 20, 60],
                "extreme_return_thresholds": [0.10, 0.20, 0.50, 1.00],
                "random_seed": 42,
                "sample_size_cross_validation": 20,
            },
            "snapshots": [],
            "schema_comparison": [],
        }

    with open(target_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config.setdefault("version", "1.0.0")
    config.setdefault("audit_name", "Russell 1000 Historical Dataset Audit")
    config.setdefault("snapshots", [])
    config.setdefault("schema_comparison", [])
    return config
