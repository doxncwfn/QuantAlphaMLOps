"""
3-Level Point-in-Time Date Strategy & Within-Spell Drift Detection.
==================================================================
Implements:
- Level 1: Primary trading session midpoint lookup.
- Level 2: Boundary fallback (start_date & end_date) if Level 1 is empty/weak.
- Level 3: Within-spell identity drift detection (divergent CIK/FIGI).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from src.common.config import TRADING_SESSIONS_PATH


class RepresentativeDateStrategy:
    """Computes trading calendar midpoints and identifies within-spell identity drift."""

    def __init__(
        self,
        sessions_path: Path | None = None,
        logger: logging.Logger | None = None,
    ):
        self.sessions_path = sessions_path or TRADING_SESSIONS_PATH
        self.logger = logger or logging.getLogger("date_strategy")
        self.all_sessions: list[str] = []
        self.session_to_idx: dict[str, int] = {}
        self._load_sessions()

    def _load_sessions(self) -> None:
        if self.sessions_path.exists():
            df = pl.read_parquet(self.sessions_path)
            self.all_sessions = df["session_date"].to_list()
            self.session_to_idx = {d: i for i, d in enumerate(self.all_sessions)}
            self.logger.info(
                "Loaded %d trading sessions from %s.",
                len(self.all_sessions),
                self.sessions_path,
            )
        else:
            self.logger.warning(
                "Trading sessions file not found at %s. Midpoints will use fallback.",
                self.sessions_path,
            )

    def get_midpoint_session(self, start_date: str, end_date: str) -> str:
        """Finds trading calendar midpoint date between start_date and end_date."""
        s_idx = self.session_to_idx.get(start_date)
        e_idx = self.session_to_idx.get(end_date)
        if s_idx is not None and e_idx is not None and s_idx <= e_idx:
            mid_idx = (s_idx + e_idx) // 2
            return self.all_sessions[mid_idx]
        return start_date

    def detect_drift(
        self,
        evidence_points: list[tuple[str, str, dict[str, Any]]],
    ) -> tuple[bool, str]:
        """Detects if multiple sample points within a spell produce divergent CIKs or FIGIs.

        Args:
            evidence_points: List of (label, date, matched_record)
        """
        if len(evidence_points) < 2:
            return False, ""

        ciks = {p[2].get("cik") for p in evidence_points if p[2] and p[2].get("cik")}
        figis = {
            (p[2].get("share_class_figi") or p[2].get("composite_figi"))
            for p in evidence_points
            if p[2] and (p[2].get("share_class_figi") or p[2].get("composite_figi"))
        }

        if len(ciks) > 1 or len(figis) > 1:
            details = f"Boundary divergence: CIKs={list(ciks)}, FIGIs={list(figis)}"
            return True, details
        return False, ""
