"""Centralized Exception Registry for recording and exporting data quality findings."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class QualityException:
    """Structured audit exception record."""

    exception_id: str
    category: str
    severity: str
    year: int | None
    ticker: str | None
    security_id: str | int | None
    date_start: str | None
    date_end: str | None
    description: str
    evidence: str
    status: str
    resolution: str | None = None
    notes: str | None = None


class ExceptionRegistry:
    """Thread-safe, deterministic exception registry for data quality findings."""

    VALID_CATEGORIES = {
        "MEMBERSHIP",
        "SOURCE_CONFLICT",
        "DUPLICATE",
        "TICKER",
        "IDENTITY",
        "COVERAGE",
        "MISSINGNESS",
        "PRICE",
        "CORPORATE_ACTION",
        "DELISTING",
        "IPO",
        "MODEL_ELIGIBILITY",
        "LEAKAGE",
    }

    VALID_SEVERITIES = {"INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def __init__(self) -> None:
        self._exceptions: list[QualityException] = []
        self._counter = 0

    def register(
        self,
        category: str,
        severity: str,
        description: str,
        evidence: str,
        year: int | None = None,
        ticker: str | None = None,
        security_id: str | int | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
        status: str = "FLAGGED",
        resolution: str | None = None,
        notes: str | None = None,
    ) -> QualityException:
        """Register a single audit finding."""
        if category not in self.VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category: {category}. Must be one of {self.VALID_CATEGORIES}"
            )
        if severity not in self.VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity: {severity}. Must be one of {self.VALID_SEVERITIES}"
            )

        self._counter += 1
        exc_id = f"EXC-{category[:4]}-{self._counter:05d}"
        exc = QualityException(
            exception_id=exc_id,
            category=category,
            severity=severity,
            year=year,
            ticker=ticker,
            security_id=str(security_id) if security_id is not None else None,
            date_start=date_start,
            date_end=date_end,
            description=description,
            evidence=evidence,
            status=status,
            resolution=resolution,
            notes=notes,
        )
        self._exceptions.append(exc)
        return exc

    def to_polars(self) -> pl.DataFrame:
        """Export registry as a sorted, deterministic Polars DataFrame."""
        schema = {
            "exception_id": pl.String,
            "category": pl.String,
            "severity": pl.String,
            "year": pl.Int32,
            "ticker": pl.String,
            "security_id": pl.String,
            "date_start": pl.String,
            "date_end": pl.String,
            "description": pl.String,
            "evidence": pl.String,
            "status": pl.String,
            "resolution": pl.String,
            "notes": pl.String,
        }
        if not self._exceptions:
            return pl.DataFrame(schema=schema)

        data = [asdict(e) for e in self._exceptions]
        return pl.DataFrame(data, schema=schema).sort(
            ["category", "severity", "year", "ticker", "exception_id"]
        )

    def save(self, output_dir: Path | str) -> tuple[Path, Path]:
        """Save registry to both Parquet and CSV formats atomically."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = out_dir / "exception_registry.parquet"
        csv_path = out_dir / "exception_registry.csv"

        df = self.to_polars()

        # Atomic write parquet
        tmp_parquet = parquet_path.with_suffix(".tmp")
        df.write_parquet(tmp_parquet)
        tmp_parquet.replace(parquet_path)

        # Atomic write csv
        tmp_csv = csv_path.with_suffix(".tmp")
        df.write_csv(tmp_csv)
        tmp_csv.replace(csv_path)

        logger.info(
            "Saved %d exceptions to %s and %s",
            len(self._exceptions),
            parquet_path.name,
            csv_path.name,
        )
        return parquet_path, csv_path

    @property
    def total_count(self) -> int:
        return len(self._exceptions)

    def summary_by_category(self) -> dict[str, int]:
        df = self.to_polars()
        if len(df) == 0:
            return {}
        counts = df.group_by("category").agg(pl.len().alias("count")).to_dicts()
        return {r["category"]: r["count"] for r in counts}

    def summary_by_severity(self) -> dict[str, int]:
        df = self.to_polars()
        if len(df) == 0:
            return {}
        counts = df.group_by("severity").agg(pl.len().alias("count")).to_dicts()
        return {r["severity"]: r["count"] for r in counts}
