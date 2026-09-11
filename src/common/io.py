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
from typing import Any, Dict, List, Union

import polars as pl


def atomic_write_parquet(df: pl.DataFrame, target_path: Union[str, Path]) -> None:
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


def atomic_write_json(data: Any, target_path: Union[str, Path], indent: int = 2) -> None:
    """Atomically write Python data to JSON via temporary file rename with fsync."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.stem}_",
        suffix=".tmp",
        delete=False,
    )
    try:
        json.dump(data, tmp_file, indent=indent, ensure_ascii=False)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_file.close()
        os.replace(tmp_file.name, target)
    except Exception:
        if os.path.exists(tmp_file.name):
            os.remove(tmp_file.name)
        raise


def atomic_write_csv(df: pl.DataFrame, target_path: Union[str, Path]) -> None:
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
