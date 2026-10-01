#!/usr/bin/env python3
"""
High-performance CSV to Parquet conversion utility.

Supports single files, directories, glob patterns, multi-worker parallel conversion,
custom compression (snappy, zstd, gzip, lz4), and streaming conversion for large files.
Available engines: 'polars' (default), 'pyarrow', 'pandas'.

Handles financial data quirks (e.g. CRSP/WRDS non-numeric codes, mixed-type columns)
by inferring schema across the full file by default and offering graceful fallbacks.

Usage Examples:
    # Convert a single file
    python tools/csv_to_parquet.py "data/WRDS/2000.csv"

    # Convert all CSVs in a directory to Parquet using ZSTD compression
    python tools/csv_to_parquet.py "data/processed/" -c zstd

    # Convert with custom output directory and 4 workers
    python tools/csv_to_parquet.py "data/processed/" -o "data/parquet/" -w 4

    # Convert with explicit null values or ignore errors
    python tools/csv_to_parquet.py "data/WRDS/2000.csv" --ignore-errors
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("csv_to_parquet")


@dataclass
class ConversionResult:
    input_path: Path
    output_path: Path
    success: bool
    num_rows: int = 0
    input_size_bytes: int = 0
    output_size_bytes: int = 0
    duration_sec: float = 0.0
    error_message: str | None = None

    @property
    def compression_ratio(self) -> float:
        if self.input_size_bytes == 0:
            return 0.0
        return (1.0 - (self.output_size_bytes / self.input_size_bytes)) * 100.0


def _format_size(size_bytes: int) -> str:
    """Format byte sizes in human-readable units."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:3.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def convert_file_polars(
    csv_path: Path,
    output_path: Path,
    compression: str = "snappy",
    streaming: bool = True,
    infer_schema_length: int | None = None,
    null_values: list[str] | None = None,
    ignore_errors: bool = False,
    all_strings: bool = False,
) -> int:
    """
    Convert a CSV file to Parquet using Polars.

    Default infer_schema_length=None ensures full-file schema scanning,
    preventing mid-file type mismatches (such as 'T' in WRDS/CRSP float columns).
    """
    comp_map = {
        "snappy": "snappy",
        "zstd": "zstd",
        "gzip": "gzip",
        "lz4": "lz4",
        "uncompressed": "uncompressed",
    }
    comp = comp_map.get(compression.lower(), "snappy")

    # If all_strings requested, set infer_schema_length to 0
    effective_infer_len = 0 if all_strings else infer_schema_length

    # 1. Attempt streaming sink with full schema inference
    if streaming and not all_strings:
        try:
            pl.scan_csv(
                csv_path,
                infer_schema_length=effective_infer_len,
                null_values=null_values,
                ignore_errors=ignore_errors,
            ).sink_parquet(output_path, compression=comp)
            return pl.scan_parquet(output_path).select(pl.len()).collect().item()
        except (pl.exceptions.PolarsError, OSError, ValueError) as scan_err:
            logger.debug(
                "Polars streaming failed for %s (%s). Retrying with eager read...",
                csv_path.name,
                scan_err,
            )
            output_path.unlink(missing_ok=True)

    # 2. Eager read fallback
    try:
        df = pl.read_csv(
            csv_path,
            infer_schema_length=effective_infer_len,
            null_values=null_values,
            ignore_errors=ignore_errors,
        )
        df.write_parquet(output_path, compression=comp)
        return len(df)
    except Exception:
        # 3. Fallback: if infer_schema_length was partial, retry with None
        if effective_infer_len is not None:
            logger.warning(
                "Polars read failed for %s with partial schema. Retrying with full schema scan...",
                csv_path.name,
            )
            df = pl.read_csv(
                csv_path,
                infer_schema_length=None,
                null_values=null_values,
                ignore_errors=ignore_errors,
            )
            df.write_parquet(output_path, compression=comp)
            return len(df)
        raise


def convert_file_pyarrow(
    csv_path: Path,
    output_path: Path,
    compression: str = "snappy",
    null_values: list[str] | None = None,
    ignore_errors: bool = False,
) -> int:
    """Convert a CSV file to Parquet using PyArrow."""
    import pyarrow.csv as pc
    import pyarrow.parquet as pq

    read_options = pc.ReadOptions(autogenerate_column_names=False)
    parse_options = pc.ParseOptions(
        ignore_empty_lines=True,
        null_values=null_values,
    )
    convert_options = pc.ConvertOptions()

    table = pc.read_csv(
        csv_path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )
    pq.write_table(table, output_path, compression=compression)
    return table.num_rows


def convert_file_pandas(
    csv_path: Path,
    output_path: Path,
    compression: str = "snappy",
    null_values: list[str] | None = None,
    all_strings: bool = False,
) -> int:
    """Convert a CSV file to Parquet using Pandas."""
    import pandas as pd

    dtype = str if all_strings else None
    df = pd.read_csv(
        csv_path,
        na_values=null_values,
        dtype=dtype,
        low_memory=False,
    )
    df.to_parquet(output_path, compression=compression, index=False)
    return len(df)


def convert_csv_to_parquet(
    csv_path: Path,
    output_path: Path | None = None,
    engine: str = "polars",
    compression: str = "snappy",
    streaming: bool = True,
    overwrite: bool = True,
    infer_schema_length: int | None = None,
    null_values: list[str] | None = None,
    ignore_errors: bool = False,
    all_strings: bool = False,
) -> ConversionResult:
    """
    Convert a single CSV file to Parquet format with automatic error recovery.
    """
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.with_suffix(".parquet")
    else:
        output_path = Path(output_path)
        if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / f"{csv_path.stem}.parquet"

    input_size = csv_path.stat().st_size if csv_path.exists() else 0

    if not csv_path.exists():
        return ConversionResult(
            input_path=csv_path,
            output_path=output_path,
            success=False,
            error_message=f"Source file does not exist: {csv_path}",
        )

    if output_path.exists() and not overwrite:
        logger.info("Skipping existing file: %s", output_path)
        return ConversionResult(
            input_path=csv_path,
            output_path=output_path,
            success=True,
            output_size_bytes=output_path.stat().st_size,
            input_size_bytes=input_size,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    try:
        if engine == "polars":
            try:
                num_rows = convert_file_polars(
                    csv_path,
                    output_path,
                    compression=compression,
                    streaming=streaming,
                    infer_schema_length=infer_schema_length,
                    null_values=null_values,
                    ignore_errors=ignore_errors,
                    all_strings=all_strings,
                )
            except (
                pl.exceptions.PolarsError,
                OSError,
                ValueError,
                RuntimeError,
            ) as polars_err:
                # Automatic resilient fallback to PyArrow if Polars encounters unresolvable schema quirks
                logger.warning(
                    "Polars encountered schema error on %s: %s. Falling back to PyArrow...",
                    csv_path.name,
                    polars_err,
                )
                num_rows = convert_file_pyarrow(
                    csv_path,
                    output_path,
                    compression=compression,
                    null_values=null_values,
                    ignore_errors=ignore_errors,
                )
        elif engine == "pyarrow":
            num_rows = convert_file_pyarrow(
                csv_path,
                output_path,
                compression=compression,
                null_values=null_values,
                ignore_errors=ignore_errors,
            )
        elif engine == "pandas":
            num_rows = convert_file_pandas(
                csv_path,
                output_path,
                compression=compression,
                null_values=null_values,
                all_strings=all_strings,
            )
        else:
            raise ValueError(f"Unsupported engine: '{engine}'. Choose polars, pyarrow, or pandas.")

        elapsed = time.perf_counter() - t0
        output_size = output_path.stat().st_size

        return ConversionResult(
            input_path=csv_path,
            output_path=output_path,
            success=True,
            num_rows=num_rows,
            input_size_bytes=input_size,
            output_size_bytes=output_size,
            duration_sec=elapsed,
        )

    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        # Clean up partial/corrupted output file on failure
        output_path.unlink(missing_ok=True)

        logger.error("Failed to convert %s: %s", csv_path, e)
        return ConversionResult(
            input_path=csv_path,
            output_path=output_path,
            success=False,
            input_size_bytes=input_size,
            duration_sec=elapsed,
            error_message=str(e),
        )


def collect_csv_files(
    inputs: Sequence[str],
    recursive: bool = False,
    pattern: str = "*.csv",
) -> list[Path]:
    """Collect unique CSV file paths from arguments, directory paths, or glob expressions."""
    found: set[Path] = set()

    for item in inputs:
        p = Path(item)
        if p.is_file() and p.suffix.lower() == ".csv":
            found.add(p.resolve())
        elif p.is_dir():
            glob_fn = p.rglob if recursive else p.glob
            for f in glob_fn(pattern):
                if f.is_file() and f.suffix.lower() == ".csv":
                    found.add(f.resolve())
        else:
            matches = glob.glob(item, recursive=recursive)
            if matches:
                for m in matches:
                    mp = Path(m)
                    if mp.is_file() and mp.suffix.lower() == ".csv":
                        found.add(mp.resolve())
            else:
                logger.warning("No matches found for input: %s", item)

    return sorted(found)


def batch_convert(
    csv_files: Sequence[Path],
    output_dir: Path | None = None,
    engine: str = "polars",
    compression: str = "snappy",
    streaming: bool = True,
    overwrite: bool = True,
    infer_schema_length: int | None = None,
    null_values: list[str] | None = None,
    ignore_errors: bool = False,
    all_strings: bool = False,
    max_workers: int = 4,
) -> list[ConversionResult]:
    """Convert multiple CSV files to Parquet concurrently."""
    results: list[ConversionResult] = []
    if not csv_files:
        logger.warning("No CSV files to convert.")
        return results

    logger.info(
        "Starting conversion of %d file(s) [Engine: %s, Compression: %s, Workers: %d]",
        len(csv_files),
        engine,
        compression,
        max_workers,
    )

    def _task(file_path: Path) -> ConversionResult:
        if output_dir:
            dest = output_dir / f"{file_path.stem}.parquet"
        else:
            dest = file_path.with_suffix(".parquet")
        return convert_csv_to_parquet(
            csv_path=file_path,
            output_path=dest,
            engine=engine,
            compression=compression,
            streaming=streaming,
            overwrite=overwrite,
            infer_schema_length=infer_schema_length,
            null_values=null_values,
            ignore_errors=ignore_errors,
            all_strings=all_strings,
        )

    if max_workers <= 1 or len(csv_files) == 1:
        for f in csv_files:
            res = _task(f)
            results.append(res)
            _log_result(res)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_task, f): f for f in csv_files}
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                results.append(res)
                _log_result(res)

    results.sort(key=lambda r: str(r.input_path))
    return results


def _log_result(res: ConversionResult):
    if res.success:
        logger.info(
            "✓ %s -> %s | %d rows | %s -> %s (-%.1f%%) in %.2fs",
            res.input_path.name,
            res.output_path.name,
            res.num_rows,
            _format_size(res.input_size_bytes),
            _format_size(res.output_size_bytes),
            res.compression_ratio,
            res.duration_sec,
        )
    else:
        logger.error("✗ %s failed: %s", res.input_path.name, res.error_message)


def print_summary_report(results: Sequence[ConversionResult]):
    """Print an aggregated summary report of the conversion operation."""
    total = len(results)
    succeeded = sum(1 for r in results if r.success)
    failed = total - succeeded

    total_in_bytes = sum(r.input_size_bytes for r in results if r.success)
    total_out_bytes = sum(r.output_size_bytes for r in results if r.success)
    total_rows = sum(r.num_rows for r in results if r.success)
    total_time = sum(r.duration_sec for r in results)

    savings = (1.0 - (total_out_bytes / total_in_bytes)) * 100.0 if total_in_bytes > 0 else 0.0

    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY REPORT")
    print("=" * 60)
    print(f"Files Processed   : {total} ({succeeded} succeeded, {failed} failed)")
    print(f"Total Rows Saved  : {total_rows:,}")
    print(f"Input Data Size   : {_format_size(total_in_bytes)}")
    print(f"Output Data Size  : {_format_size(total_out_bytes)}")
    print(f"Disk Space Saved  : {savings:.1f}% reduction")
    print(f"Total Engine Time : {total_time:.2f}s")
    print("=" * 60 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert CSV files to compressed Parquet format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input CSV file(s), directory, or glob pattern(s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output destination path (file path if single input, directory if multiple).",
    )
    parser.add_argument(
        "-c",
        "--compression",
        choices=["snappy", "zstd", "gzip", "lz4", "uncompressed"],
        default="snappy",
        help="Compression codec for Parquet output.",
    )
    parser.add_argument(
        "-e",
        "--engine",
        choices=["polars", "pyarrow", "pandas"],
        default="polars",
        help="Underlying data processing engine.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively scan directories for CSV files.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads for batch conversion.",
    )
    parser.add_argument(
        "--infer-schema-length",
        type=str,
        default="all",
        help="Number of rows to infer schema ('all' or 'none' for full scan, 0 for all strings, or an integer).",
    )
    parser.add_argument(
        "--null-values",
        type=str,
        default=None,
        help="Comma-separated strings to treat as null (e.g. 'NA,null,T').",
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Ignore unparseable CSV rows or cells instead of failing.",
    )
    parser.add_argument(
        "--all-strings",
        action="store_true",
        help="Load all columns as strings (guarantees schema compatibility).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode for Polars engine.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip files if destination Parquet already exists.",
    )
    return parser


def parse_infer_schema_length(val: str) -> int | None:
    """Parse infer_schema_length argument from CLI string."""
    v = val.strip().lower()
    if v in ("all", "none", "-1"):
        return None
    try:
        return int(v)
    except ValueError:
        logger.warning("Unrecognized infer-schema-length '%s', defaulting to full scan.", val)
        return None


def main():
    parser = build_parser()
    args = parser.parse_args()

    infer_schema_len = parse_infer_schema_length(args.infer_schema_length)
    null_vals = [s.strip() for s in args.null_values.split(",")] if args.null_values else None

    csv_files = collect_csv_files(args.inputs, recursive=args.recursive)
    if not csv_files:
        logger.error("No CSV files located matching the provided inputs: %s", args.inputs)
        sys.exit(1)

    # Single file with explicit .parquet output target
    output_dir = None
    if len(csv_files) == 1 and args.output and args.output.suffix.lower() == ".parquet":
        res = convert_csv_to_parquet(
            csv_path=csv_files[0],
            output_path=args.output,
            engine=args.engine,
            compression=args.compression,
            streaming=not args.no_streaming,
            overwrite=not args.no_overwrite,
            infer_schema_length=infer_schema_len,
            null_values=null_vals,
            ignore_errors=args.ignore_errors,
            all_strings=args.all_strings,
        )
        _log_result(res)
        print_summary_report([res])
        sys.exit(0 if res.success else 1)

    if args.output:
        output_dir = args.output
        output_dir.mkdir(parents=True, exist_ok=True)

    results = batch_convert(
        csv_files=csv_files,
        output_dir=output_dir,
        engine=args.engine,
        compression=args.compression,
        streaming=not args.no_streaming,
        overwrite=not args.no_overwrite,
        infer_schema_length=infer_schema_len,
        null_values=null_vals,
        ignore_errors=args.ignore_errors,
        all_strings=args.all_strings,
        max_workers=args.workers,
    )

    print_summary_report(results)
    if any(not r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
