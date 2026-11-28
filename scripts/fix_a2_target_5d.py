"""Recompute silver.features_daily.target_5d (and derived y_star) from ret_total + log1p(dlret).

Addresses audit finding A2:
1. Uses log total return (`ret_total`, including dividends) across [t+1, t+5] instead
   of ex-dividend price return (`ret_price`).
2. Compounds terminal delisting returns using `log1p(apply_delisting(...))` instead of
   arithmetic addition, stopping accumulation at the CRSP delisting session.
3. Updates `target_5d` and recomputes cross-sectional rank `y_star` in `silver.features_daily`
   while preserving all 30 features and metadata columns bit-identically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from src.data.compute_target import apply_delisting
from src.data.normalize_schema import load_warehouse_data

DEFAULT_WAREHOUSE = Path("data/warehouse/russell1000.duckdb")
DEFAULT_MANIFEST = Path("data/warehouse/fix_a2_manifest.json")

ORDERED_COLUMNS: list[str] = [
    "permno",
    "date",
    "r_cc",
    "r_on",
    "r_id",
    "range",
    "close_loc",
    "rel_vol",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "vol_20d",
    "vol_60d",
    "vol_ratio",
    "price_z_20",
    "drawdown_20",
    "eff_ratio_20",
    "vol_shock_5",
    "resid_ret",
    "resid_mom_20",
    "resid_vol_20",
    "ctx_mktrf",
    "ctx_smb",
    "ctx_hml",
    "ctx_rmw",
    "ctx_cma",
    "ctx_umd",
    "ctx_mkt_vol_20",
    "ctx_mkt_ret_20",
    "ctx_breadth",
    "ctx_dispersion",
    "ctx_factor_disp",
    "target_5d",
    "y_star",
    "in_universe",
    "features_ready",
    "era",
]

PRESERVED_COLUMNS: list[str] = [
    c for c in ORDERED_COLUMNS if c not in ("target_5d", "y_star")
]


def _series_sha256(arr: np.ndarray) -> str:
    """Compute a deterministic SHA-256 hex digest for a numeric numpy array."""
    h = hashlib.sha256()
    if arr.dtype == object or arr.dtype.kind in ("U", "S"):
        for val in arr:
            h.update(("" if val is None else str(val)).encode("utf-8"))
            h.update(b"\x00")
    else:
        contiguous = np.ascontiguousarray(arr)
        h.update(contiguous.tobytes())
    return h.hexdigest()


def _column_stats(arr: np.ndarray) -> dict[str, float | int | str]:
    """Compute summary statistics and digest for a floating-point target column."""
    arr_f = np.asarray(arr, dtype=np.float64)
    null_mask = np.isnan(arr_f)
    valid = arr_f[~null_mask]
    return {
        "row_count": len(arr_f),
        "null_count": int(null_mask.sum()),
        "null_pct": float(null_mask.mean() * 100.0),
        "min": float(valid.min()) if len(valid) else float("nan"),
        "max": float(valid.max()) if len(valid) else float("nan"),
        "mean": float(valid.mean()) if len(valid) else float("nan"),
        "std": float(valid.std(ddof=1)) if len(valid) > 1 else float("nan"),
        "sha256": _series_sha256(arr_f),
    }


def _preserved_digest(df: pd.DataFrame) -> dict[str, str]:
    """Hash every preserved non-target column in deterministic (permno, date) order."""
    digests: dict[str, str] = {}
    for col in PRESERVED_COLUMNS:
        vals = df[col].to_numpy()
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            vals = vals.astype("datetime64[ns]").view("int64")
        elif col == "era":
            vals = vals.astype(object)
        digests[col] = _series_sha256(vals)
    return digests


def compute_target_5d_v2(df: pd.DataFrame) -> np.ndarray:
    """Compute forward 5-session log total return with log1p terminal delistings.

    Parameters
    ----------
    df : pd.DataFrame
        Full warehouse panel sorted by `(permno, date)` with columns `permno`, `date`,
        `ret_total`, `dlstcd`, `dlret`, `dlretx`, `era`.

    Returns
    -------
    np.ndarray
        Float64 array aligned with `df` containing `target_5d_v2`.
    """
    p_key = df["permno"]
    rt = df["ret_total"]
    target_5d = (
        rt.groupby(p_key, sort=False).shift(-1)
        + rt.groupby(p_key, sort=False).shift(-2)
        + rt.groupby(p_key, sort=False).shift(-3)
        + rt.groupby(p_key, sort=False).shift(-4)
        + rt.groupby(p_key, sort=False).shift(-5)
    ).to_numpy(dtype=np.float64, copy=True)

    permnos = p_key.to_numpy()
    split_points = np.flatnonzero(permnos[1:] != permnos[:-1]) + 1
    starts = np.concatenate([[0], split_points])
    ends = np.concatenate([split_points, [len(df)]])

    dlstcd_arr = df["dlstcd"].to_numpy(dtype=np.float64)
    dlret_arr = df["dlret"].to_numpy(dtype=np.float64)
    dlretx_arr = df["dlretx"].to_numpy(dtype=np.float64)
    rt_arr = rt.to_numpy(dtype=np.float64)
    era_arr = df["era"].to_numpy()

    for st_idx, end_idx in zip(starts, ends):
        for l_idx in range(st_idx, end_idx):
            if era_arr[l_idx] != "CRSP":
                continue
            code_val = dlstcd_arr[l_idx]
            if np.isnan(code_val) or int(code_val) == 100:
                continue
            is_series_end = l_idx == end_idx - 1
            is_era_end = l_idx < end_idx - 1 and era_arr[l_idx + 1] != "CRSP"
            if not (is_series_end or is_era_end):
                continue

            term_ret = apply_delisting(
                {
                    "dlstcd": code_val,
                    "dlret": dlret_arr[l_idx],
                    "dlretx": dlretx_arr[l_idx],
                }
            )
            if term_ret is None:
                continue

            win_start = max(st_idx, l_idx - 4)
            for t_idx in range(win_start, l_idx + 1):
                if np.isnan(term_ret):
                    target_5d[t_idx] = np.nan
                    continue
                inter = rt_arr[t_idx + 1 : l_idx + 1]
                if len(inter) > 0 and np.isnan(inter).any():
                    target_5d[t_idx] = np.nan
                else:
                    prior_sum = float(np.sum(inter)) if len(inter) > 0 else 0.0
                    target_5d[t_idx] = prior_sum + float(np.log1p(float(term_ret)))

    return target_5d


def compute_y_star(
    dates: pd.Series,
    in_universe: np.ndarray,
    target_5d: np.ndarray,
) -> np.ndarray:
    """Recompute cross-sectional uniform rank y_star in [-0.5, +0.5]."""
    valid_mask = np.asarray(in_universe, dtype=bool) & np.isfinite(target_5d)
    y_star = np.full(len(target_5d), np.nan, dtype=np.float64)
    if not valid_mask.any():
        return y_star

    valid_df = pd.DataFrame(
        {"date": dates.to_numpy()[valid_mask], "target_5d": target_5d[valid_mask]}
    )
    ranks = valid_df.groupby("date")["target_5d"].rank(method="average", ascending=True)
    counts = valid_df.groupby("date")["target_5d"].transform("count")
    has_pairs = counts >= 2
    vals = np.full(len(valid_df), np.nan, dtype=np.float64)
    vals[has_pairs.to_numpy()] = (
        (ranks.loc[has_pairs] - 1.0) / (counts.loc[has_pairs] - 1.0) - 0.5
    ).to_numpy(dtype=np.float64)
    y_star[valid_mask] = vals
    return y_star


def apply_fix(
    warehouse_path: Path = DEFAULT_WAREHOUSE,
    manifest_path: Path = DEFAULT_MANIFEST,
    dry_run: bool = False,
) -> dict:
    """Recompute target_5d + y_star and update silver.features_daily atomically."""
    print(f"[1/5] Loading normalized daily panel from {warehouse_path}...")
    panel_df, _, _ = load_warehouse_data(warehouse_path, include_yahoo=True)
    panel_df = panel_df.sort_values(["permno", "date"], kind="mergesort").reset_index(
        drop=True
    )

    print("[2/5] Computing target_5d_v2 (ret_total + log1p(dlret))...")
    t5_full = compute_target_5d_v2(panel_df)
    in_window = (panel_df["date"] >= "2000-06-30") & (panel_df["date"] <= "2026-07-31")
    updates_df = pd.DataFrame(
        {
            "permno": panel_df.loc[in_window, "permno"].to_numpy(dtype=np.int64),
            "date": panel_df.loc[in_window, "date"].to_numpy(),
            "target_5d_v2": t5_full[in_window.to_numpy()],
        }
    )

    print(
        "[3/5] Reading existing silver.features_daily for alignment & pre-state audit..."
    )
    con = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        fd = con.execute(
            "SELECT * FROM silver.features_daily ORDER BY permno, date"
        ).df()
        orig_schema = con.execute("DESCRIBE silver.features_daily").fetchall()
    finally:
        con.close()

    fd["date"] = pd.to_datetime(fd["date"])
    if len(fd) != len(updates_df):
        raise RuntimeError(
            f"Row count mismatch: silver.features_daily={len(fd)} vs panel={len(updates_df)}"
        )
    if not (
        (fd["permno"].to_numpy() == updates_df["permno"].to_numpy()).all()
        and (fd["date"].to_numpy() == updates_df["date"].to_numpy()).all()
    ):
        raise RuntimeError("Key mismatch between silver.features_daily and panel_df")

    before_t5 = _column_stats(fd["target_5d"].to_numpy())
    before_ys = _column_stats(fd["y_star"].to_numpy())
    pre_preserved_hashes = _preserved_digest(fd)

    new_t5 = updates_df["target_5d_v2"].to_numpy(dtype=np.float64)
    new_ys = compute_y_star(fd["date"], fd["in_universe"].to_numpy(dtype=bool), new_t5)
    after_t5 = _column_stats(new_t5)
    after_ys = _column_stats(new_ys)

    if dry_run:
        print("[DRY RUN] Skipping DuckDB write.")
        return {
            "dry_run": True,
            "before_target_5d": before_t5,
            "after_target_5d": after_t5,
            "before_y_star": before_ys,
            "after_y_star": after_ys,
        }

    print("[4/5] Applying atomic table replacement on silver.features_daily...")
    upd_arrow = pa.table(
        {
            "permno": pa.array(
                updates_df["permno"].to_numpy(dtype=np.int64), type=pa.int64()
            ),
            "date": pa.array(
                pd.to_datetime(updates_df["date"]).dt.to_pydatetime(),
                type=pa.timestamp("us"),
            ),
            "new_target_5d": pa.array(new_t5, type=pa.float64(), mask=np.isnan(new_t5)),
            "new_y_star": pa.array(new_ys, type=pa.float64(), mask=np.isnan(new_ys)),
        }
    )

    select_cols = ",\n            ".join(
        [
            "u.new_target_5d AS target_5d"
            if c == "target_5d"
            else "u.new_y_star AS y_star"
            if c == "y_star"
            else f'f."{c}"'
            for c in ORDERED_COLUMNS
        ]
    )

    con_rw = duckdb.connect(str(warehouse_path), read_only=False)
    try:
        con_rw.register("a2_updates", upd_arrow)
        con_rw.execute("BEGIN TRANSACTION")
        con_rw.execute(
            f"""
            CREATE TABLE silver.features_daily_a2_tmp AS
            SELECT
                {select_cols}
            FROM silver.features_daily f
            JOIN a2_updates u
              ON f.permno = u.permno AND f.date = u.date
            ORDER BY f.date, f.permno
            """
        )
        new_cnt = con_rw.execute(
            "SELECT count(*) FROM silver.features_daily_a2_tmp"
        ).fetchone()[0]
        if new_cnt != len(fd):
            con_rw.execute("ROLLBACK")
            raise RuntimeError(f"Join row count {new_cnt} != expected {len(fd)}")

        con_rw.execute("DROP TABLE silver.features_daily")
        con_rw.execute(
            "ALTER TABLE silver.features_daily_a2_tmp RENAME TO features_daily"
        )
        con_rw.execute("COMMIT")
        con_rw.unregister("a2_updates")
    except Exception:
        try:
            con_rw.execute("ROLLBACK")
        except Exception:  # noqa: BLE001, S110
            pass
        raise
    finally:
        con_rw.close()

    print("[5/5] Verifying post-write hashes & writing manifest...")
    con_ro = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        fd_after = con_ro.execute(
            "SELECT * FROM silver.features_daily ORDER BY permno, date"
        ).df()
        post_schema = con_ro.execute("DESCRIBE silver.features_daily").fetchall()
    finally:
        con_ro.close()

    if [(r[0], r[1]) for r in orig_schema] != [(r[0], r[1]) for r in post_schema]:
        raise RuntimeError("Schema changed unexpectedly after update!")

    fd_after["date"] = pd.to_datetime(fd_after["date"])
    post_preserved_hashes = _preserved_digest(fd_after)
    if post_preserved_hashes != pre_preserved_hashes:
        mismatched = [
            k
            for k in PRESERVED_COLUMNS
            if pre_preserved_hashes[k] != post_preserved_hashes[k]
        ]
        raise RuntimeError(f"Preserved columns modified unexpectedly: {mismatched}")

    manifest = {
        "warehouse_path": str(warehouse_path),
        "row_count": len(fd_after),
        "schema_preserved": True,
        "preserved_columns_sha256_match": True,
        "preserved_columns_count": len(PRESERVED_COLUMNS),
        "before_target_5d": before_t5,
        "after_target_5d": _column_stats(fd_after["target_5d"].to_numpy()),
        "before_y_star": before_ys,
        "after_y_star": _column_stats(fd_after["y_star"].to_numpy()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix A2 target_5d in DuckDB warehouse")
    parser.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    res = apply_fix(args.warehouse, args.manifest, dry_run=args.dry_run)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
