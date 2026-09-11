"""
Atomic File I/O Utilities.
==========================
Ensures torn-write protection across parquet, json, and csv files using
temporary files and atomic filesystem replacement (os.replace).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import polars as pl


def atomic_write_parquet(df: pl.DataFrame, target_path: str | Path) -> None:
    """Atomically write a Polars DataFrame to Parquet via temporary file rename."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(f".{os.getpid()}_{os.urandom(4).hex()}.tmp")
    try:
        df.write_parquet(tmp_path)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def atomic_write_json(data: Any, target_path: str | Path, indent: int = 2) -> None:
    """Atomically write Python data to JSON via temporary file rename with fsync."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.stem}_",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_name = tmp_file.name
            json.dump(data, tmp_file, indent=indent, ensure_ascii=False)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_name, target)
    except Exception:
        if tmp_name and os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def atomic_write_csv(df: pl.DataFrame, target_path: str | Path) -> None:
    """Atomically write a Polars DataFrame to CSV via temporary file rename."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(f".{os.getpid()}_{os.urandom(4).hex()}.tmp")
    try:
        df.write_csv(tmp_path)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
