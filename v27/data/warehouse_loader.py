"""DuckDB OHLCV ingestion, dynamic top-1000 hysteresis universe selection, and cache I/O."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import numpy as np
import torch

from v27.config import DataConfig, PathConfig
from v27.features.stock_features import compute_34_stock_features


def _scatter_panel_col(
    col_np: np.ndarray, r_idx: np.ndarray, c_idx: np.ndarray, d_all: int, n_all: int
) -> np.ndarray:
    arr = np.full((d_all, n_all), np.nan, dtype=np.float32)
    arr[r_idx, c_idx] = col_np
    return arr


def build_or_load_dataset(
    paths: PathConfig | None = None,
    data_cfg: DataConfig | None = None,
    device: torch.device | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Load prebuilt 34-feature tensor cache or reconstruct cleanly from DuckDB warehouse."""
    paths = paths or PathConfig()
    data_cfg = data_cfg or DataConfig()
    cfg_tag = f"{data_cfg.start_date}_{data_cfg.test_end}_K{data_cfg.top_k}_TTL{data_cfg.ttl_days}_L{data_cfg.lookback}_F34_v6"
    if not force_rebuild:
        for cand in (paths.cache_file, paths.output_dir / paths.cache_filename):
            if cand.exists():
                t0 = time.time()
                cache = torch.load(cand, map_location="cpu", weights_only=False)
                if cache.get("cfg_tag") == cfg_tag:
                    print(
                        f"[Cache] Loaded {cand} in {time.time() - t0:.1f}s ({len(cache['FEATURES'])} features)"
                    )
                    return cache

    import duckdb

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_threads = os.cpu_count() or 4
    t0 = time.time()
    with duckdb.connect(str(paths.warehouse_path), read_only=True) as con:
        con.execute(f"PRAGMA threads={n_threads};")
        panel = con.sql(
            f"""
            SELECT security_id::BIGINT AS security_id, date::DATE AS date,
                   open::FLOAT AS open, high::FLOAT AS high, low::FLOAT AS low,
                   close::FLOAT AS close, volume::FLOAT AS volume,
                   adj_close::FLOAT AS adj_close, ret_total::FLOAT AS ret_total
            FROM silver.v_prices
            WHERE date BETWEEN DATE '{data_cfg.start_date}' - INTERVAL 550 DAY AND DATE '{data_cfg.test_end}'
            ORDER BY date, security_id
            """
        ).pl()

    calendar = np.sort(panel["date"].unique().to_numpy()).astype("datetime64[D]")
    all_ids = np.sort(panel["security_id"].unique().to_numpy()).astype(np.int64)
    d_all, n_all = len(calendar), len(all_ids)
    r_all = np.searchsorted(calendar, panel["date"].to_numpy().astype("datetime64[D]"))
    c_all = np.searchsorted(all_ids, panel["security_id"].to_numpy())
    raw_cols = ["open", "high", "low", "close", "volume", "adj_close", "ret_total"]
    col_arrays = {col: panel[col].fill_null(np.nan).to_numpy() for col in raw_cols}
    del panel

    with ThreadPoolExecutor(max_workers=min(len(raw_cols), n_threads)) as pool:
        fn = partial(
            _scatter_panel_col, r_idx=r_all, c_idx=c_all, d_all=d_all, n_all=n_all
        )
        mats = dict(zip(raw_cols, pool.map(fn, [col_arrays[c] for c in raw_cols])))
    del col_arrays, r_all, c_all

    close_np, vol_np = mats["close"], mats["volume"]
    has_mat = ~np.isnan(close_np)
    dv_mat = np.nan_to_num(close_np * vol_np, nan=0.0).astype(np.float64)
    has_cs = np.vstack(
        [np.zeros((1, n_all), dtype=np.int32), has_mat.astype(np.int32).cumsum(axis=0)]
    )
    dv_cs = np.vstack([np.zeros((1, n_all), dtype=np.float64), dv_mat.cumsum(axis=0)])
    del dv_mat

    first_idx = int(np.searchsorted(calendar, np.datetime64(data_cfg.start_date, "D")))
    last_idx = (
        int(
            np.searchsorted(
                calendar, np.datetime64(data_cfg.test_end, "D"), side="right"
            )
        )
        - 1
    )
    refreshes = list(range(first_idx, last_idx + 1, data_cfg.ttl_days))
    buf_size = int(data_cfg.top_k * data_cfg.buffer_ratio)
    u_rows, prev_set = [], set()
    for j, idx in enumerate(refreshes):
        elig = has_mat[idx - 1] & (
            (has_cs[idx] - has_cs[idx - data_cfg.y_win]) >= data_cfg.x_win
        )
        cand = all_ids[elig]
        score = dv_cs[idx, elig] - dv_cs[idx - data_cfg.x_win, elig]
        order = np.lexsort((cand, -score))
        top = cand[order][: data_cfg.top_k + buf_size]
        kept = [s for s in top if s in prev_set]
        kept_s = set(kept)
        fill = [s for s in top if s not in kept_s][: data_cfg.top_k - len(kept)]
        chosen = kept_s | set(fill)
        selected = [int(s) for s in top if s in chosen]
        r_date = calendar[idx]
        until = (
            calendar[refreshes[j + 1]]
            if j + 1 < len(refreshes)
            else calendar[last_idx] + np.timedelta64(1, "D")
        )
        for slot in range(data_cfg.top_k):
            sid = selected[slot] if slot < len(selected) else -1
            u_rows.append((r_date, until, slot, sid, sid == -1))
        prev_set = set(selected)
    del has_mat, has_cs, dv_cs

    active_ids = np.sort(np.unique([r[3] for r in u_rows if not r[4]])).astype(np.int64)
    act_cols = np.searchsorted(all_ids, active_ids)

    with torch.no_grad():
        feat_dict, x_gpu, y_gpu = compute_34_stock_features(
            torch.from_numpy(mats["open"][:, act_cols]).to(dev),
            torch.from_numpy(mats["high"][:, act_cols]).to(dev),
            torch.from_numpy(mats["low"][:, act_cols]).to(dev),
            torch.from_numpy(mats["close"][:, act_cols]).to(dev),
            torch.from_numpy(mats["volume"][:, act_cols]).to(dev),
            torch.from_numpy(mats["adj_close"][:, act_cols]).to(dev),
            torch.from_numpy(mats["ret_total"][:, act_cols]).to(dev),
        )
        del mats
        cut = max(0, first_idx - data_cfg.lookback)
        x_dense = x_gpu[cut:].cpu().numpy()
        y_dense = y_gpu[cut:].cpu().numpy()
        calendar = calendar[cut:]
        features = list(feat_dict.keys())
        del x_gpu, y_gpu

    d_len, n_len, _ = x_dense.shape
    slots = np.full((d_len, data_cfg.top_k), -1, dtype=np.int32)
    for r_date, until, slot, sid, is_pad in u_rows:
        if not is_pad:
            lo_i = np.searchsorted(calendar, r_date)
            hi_i = np.searchsorted(calendar, until)
            slots[lo_i:hi_i, slot] = np.searchsorted(active_ids, sid)
    bad_day = np.isnan(x_dense).any(axis=2).astype(np.int32)
    bad_cs = np.vstack(
        [np.zeros((1, n_len), dtype=np.int32), np.cumsum(bad_day, axis=0)]
    )
    decision_indices = np.where(
        (slots[:, 0] >= 0) & (np.arange(d_len) >= data_cfg.lookback)
    )[0]

    cache = {
        "cfg_tag": cfg_tag,
        "calendar": calendar,
        "active_ids": active_ids,
        "FEATURES": features,
        "X_dense": x_dense,
        "Y_dense": y_dense,
        "slots": slots,
        "bad_cs": bad_cs,
        "decision_indices": decision_indices,
    }
    out_cache = paths.output_dir / paths.cache_filename
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_cache)
    print(
        f"[Build] {len(features)} features built in {time.time() - t0:.0f}s -> {out_cache}"
    )
    return cache
