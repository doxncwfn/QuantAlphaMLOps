"""Input provenance manifest generator computing SHA-256 hashes and metadata."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hash of a file efficiently in 64KB chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_input_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """
    Generate cryptographic and metadata manifest for all audit inputs.

    Returns the manifest dictionary and writes audit_input_manifest.json atomically.
    """
    logger.info("Generating input provenance manifest...")
    paths = config.get("paths", {})
    manifest_path = Path(paths.get("manifest_path", "report/quality/audit_input_manifest.json"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    input_files: list[dict[str, Any]] = []

    # 1. Processed files
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    if processed_dir.exists():
        for p in sorted(processed_dir.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                stat = p.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
                size = stat.st_size
                sha256 = compute_sha256(p)

                row_count = None
                columns = None
                if p.suffix == ".csv":
                    try:
                        df_info = pl.scan_csv(p).collect_schema()
                        columns = df_info.names()
                        with open(p, encoding="utf-8", errors="ignore") as f:
                            row_count = sum(1 for _ in f) - 1
                    except (OSError, pl.exceptions.PolarsError):
                        pass
                elif p.suffix == ".parquet":
                    try:
                        df_info = pl.scan_parquet(p)
                        columns = df_info.collect_schema().names()
                        row_count = df_info.select(pl.len()).collect().item()
                    except (OSError, pl.exceptions.PolarsError):
                        pass
                elif p.suffix == ".txt":
                    with open(p, encoding="utf-8", errors="ignore") as f:
                        row_count = sum(1 for line in f if line.strip())

                input_files.append(
                    {
                        "path": str(p),
                        "filename": p.name,
                        "category": "processed_constituents",
                        "extension": p.suffix,
                        "size_bytes": size,
                        "mtime_utc": mtime,
                        "row_count": row_count,
                        "columns": columns,
                        "sha256": sha256,
                    }
                )

    # 2. Raw files
    raw_dir = Path(paths.get("raw_dir", "data/raw"))
    if raw_dir.exists():
        for p in sorted(raw_dir.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                stat = p.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
                size = stat.st_size
                sha256 = compute_sha256(p)

                input_files.append(
                    {
                        "path": str(p),
                        "filename": p.name,
                        "category": "raw_holdings_source",
                        "extension": p.suffix,
                        "size_bytes": size,
                        "mtime_utc": mtime,
                        "sha256": sha256,
                    }
                )

    # 3. WRDS Parquet files
    wrds_dir = Path(paths.get("wrds_dir", "data/WRDS"))
    if wrds_dir.exists():
        for p in sorted(wrds_dir.iterdir()):
            if p.is_file() and p.suffix == ".parquet" and not p.name.startswith("."):
                stat = p.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
                size = stat.st_size
                sha256 = compute_sha256(p)

                columns = None
                row_count = None
                try:
                    df_scan = pl.scan_parquet(p)
                    columns = df_scan.collect_schema().names()
                    row_count = df_scan.select(pl.len()).collect().item()
                except (OSError, pl.exceptions.PolarsError) as e:
                    logger.warning("Could not read parquet schema for %s: %s", p.name, e)

                input_files.append(
                    {
                        "path": str(p),
                        "filename": p.name,
                        "category": "wrds_market_data",
                        "extension": p.suffix,
                        "size_bytes": size,
                        "mtime_utc": mtime,
                        "row_count": row_count,
                        "column_count": len(columns) if columns else None,
                        "columns": columns,
                        "sha256": sha256,
                    }
                )

    # 3b. Unified Russell 1000 full history files
    russell1000_dir = Path(paths.get("russell1000_dir", "data/Russell1000"))
    if russell1000_dir.exists():
        for p in sorted(russell1000_dir.iterdir()):
            if p.is_file() and p.suffix == ".parquet" and not p.name.startswith("."):
                stat = p.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
                size = stat.st_size
                sha256 = compute_sha256(p)

                columns = None
                row_count = None
                try:
                    df_scan = pl.scan_parquet(p)
                    columns = df_scan.collect_schema().names()
                    row_count = df_scan.select(pl.len()).collect().item()
                except (OSError, pl.exceptions.PolarsError) as e:
                    logger.warning("Could not read parquet schema for %s: %s", p.name, e)

                input_files.append(
                    {
                        "path": str(p),
                        "filename": p.name,
                        "category": "unified_russell1000_history",
                        "extension": p.suffix,
                        "size_bytes": size,
                        "mtime_utc": mtime,
                        "row_count": row_count,
                        "column_count": len(columns) if columns else None,
                        "columns": columns,
                        "sha256": sha256,
                    }
                )

    # 4. Factors file
    factors_file = Path(paths.get("factors_file", "data/ff.csv"))
    if factors_file.exists():
        stat = factors_file.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
        size = stat.st_size
        sha256 = compute_sha256(factors_file)
        row_count = None
        columns = None
        try:
            df_info = pl.scan_csv(factors_file).collect_schema()
            columns = df_info.names()
            with open(factors_file, encoding="utf-8", errors="ignore") as f:
                row_count = sum(1 for _ in f) - 1
        except (OSError, pl.exceptions.PolarsError):
            pass

        input_files.append(
            {
                "path": str(factors_file),
                "filename": factors_file.name,
                "category": "fama_french_factors",
                "extension": factors_file.suffix,
                "size_bytes": size,
                "mtime_utc": mtime,
                "row_count": row_count,
                "columns": columns,
                "sha256": sha256,
            }
        )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_files": len(input_files),
        "manifest_version": "1.0.0",
        "inputs": input_files,
    }

    temp_path = manifest_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    temp_path.replace(manifest_path)

    logger.info("Saved input manifest with %d files to %s", len(input_files), manifest_path)
    return manifest
