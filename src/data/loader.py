"""Data loader for v1.3 cross-sectional ranking model with Blom transforms and padding.

Bug fixes applied:
- BUG-04: Fix Blom normal score denominator to use per-feature non-null count.
- BUG-05: Enforce 332-session warmup, extend context history from silver.market_factor,
          require full 60-session window validity, and eliminate zero-filling.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
import scipy.special as sp
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Dataset

from src.data.compute_features import CONTEXT_FEATURE_COLS, STOCK_FEATURE_COLS


def _rolling_mean(expr: pl.Expr, window_size: int, min_samples: int) -> pl.Expr:
    try:
        return expr.rolling_mean(window_size=window_size, min_samples=min_samples)
    except TypeError:
        return expr.rolling_mean(window_size=window_size, min_periods=min_samples)


def _rolling_std(
    expr: pl.Expr, window_size: int, min_samples: int, ddof: int = 1
) -> pl.Expr:
    try:
        return expr.rolling_std(
            window_size=window_size, min_samples=min_samples, ddof=ddof
        )
    except TypeError:
        return expr.rolling_std(
            window_size=window_size, min_periods=min_samples, ddof=ddof
        )


class FeatureDataStore:
    """Pre-processed in-memory feature tensor store for the Russell 1000 universe.

    Enforces:
    1. Cross-sectional Blom normal scores computed per date across active in-universe stocks.
    2. Rolling 252-session context z-scores strictly ending at s-1 (min 60 obs) from market factor history.
    3. Memory-mapped 3D grid [permno, date, 19] for O(1) temporal window extraction.
    4. Fixed padding to N_max = 1,200 stocks per date.
    5. Target validity mask: full 60-session lookback window valid and target_5d / y_star non-NaN.
    6. 332-session warmup period before first prediction date (>= 2001-10-01).
    """

    def __init__(
        self,
        warehouse_path: str | Path = "data/warehouse/russell1000.duckdb",
        table_name: str = "silver.features_daily",
        n_max_stocks: int = 1200,
        seq_len: int = 60,
        cache_dir: str | Path | None = "data/cache",
    ):
        self.warehouse_path = str(warehouse_path)
        self.table_name = table_name
        self.n_max_stocks = n_max_stocks
        self.seq_len = seq_len
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        self._load_and_preprocess()

    def _compute_and_validate_ynorm(self) -> None:
        """Compute per-date rank-Gaussian y_norm = Phi^{-1}(rank(y_star) / (N_valid + 1))
        over mask == True stocks only, and assert std(y_norm[mask]) in [0.95, 1.05]
        over 10 sampled evaluation dates.
        """
        self.date_ynorm: dict[int, np.ndarray] = {}
        for d_i, ys in self.date_ystar.items():
            mask = self.date_masks[d_i]
            yn = np.zeros_like(ys, dtype=np.float32)
            n_valid = int(mask.sum())
            if n_valid > 0:
                ranks = rankdata(ys[mask], method="average")
                u = ranks / (float(n_valid) + 1.0)
                yn[mask] = sp.ndtri(u).astype(np.float32)
            self.date_ynorm[d_i] = yn

        if len(self.eval_date_indices) > 0:
            rng = np.random.default_rng(42)
            n_sample = min(10, len(self.eval_date_indices))
            sampled_dates = rng.choice(
                self.eval_date_indices, size=n_sample, replace=False
            )
            for d_i in sampled_dates:
                d_int = int(d_i)
                mask = self.date_masks[d_int]
                if int(mask.sum()) >= 50:
                    std_val = float(np.std(self.date_ynorm[d_int][mask], ddof=1))
                    assert 0.95 <= std_val <= 1.05, (
                        f"y_norm std {std_val:.4f} outside [0.95, 1.05] on date index {d_int}"
                    )

    def _load_and_preprocess(self) -> None:
        """Load data from warehouse, apply Blom transforms, and build tensor lookup structures."""
        cache_file = (
            self.cache_dir / f"{self.table_name.replace('.', '_')}_store.npz"
            if self.cache_dir
            else None
        )

        if cache_file and cache_file.exists():
            print(f"Loading cached feature tensors from {cache_file}...")
            cached = np.load(cache_file, allow_pickle=True)
            self.unique_dates = cached["unique_dates"]
            self.unique_permnos = cached["unique_permnos"]
            self.grid = cached["grid"]
            self.context_norm = cached["context_norm"]
            self.date_permno_indices = cached["date_permno_indices"].item()
            self.date_permnos = cached["date_permnos"].item()
            self.date_targets = cached["date_targets"].item()
            self.date_ystar = cached["date_ystar"].item()
            self.date_masks = cached["date_masks"].item()
            self.eval_date_indices = cached["eval_date_indices"]
            self.date_to_idx = {d: i for i, d in enumerate(self.unique_dates)}
            self.permno_to_idx = {p: i for i, p in enumerate(self.unique_permnos)}
            self._compute_and_validate_ynorm()
            print(f"Loaded {len(self.eval_date_indices)} evaluation dates from cache.")
            return

        print(f"Loading {self.table_name} from DuckDB {self.warehouse_path}...")
        con = duckdb.connect(self.warehouse_path, read_only=True)
        try:
            if "." in self.table_name:
                schema_name, tbl_name = self.table_name.split(".", 1)
                sql_table = f'{schema_name}."{tbl_name.strip(chr(34))}"'
            else:
                sql_table = f'"{self.table_name.strip(chr(34))}"'
            arrow_table = con.sql(
                f"""
                SELECT permno, date,
                       r_cc, r_on, r_id, range, close_loc, rel_vol,
                       ret_5d, ret_20d, ret_60d, vol_20d, vol_60d, vol_ratio,
                       price_z_20, drawdown_20, eff_ratio_20, vol_shock_5,
                       resid_ret, resid_mom_20, resid_vol_20,
                       target_5d, y_star, features_ready
                FROM {sql_table}
                WHERE in_universe
                ORDER BY date, permno
                """
            ).to_arrow_table()

            # BUG-05 Step 5.1: Extend market-factor context source from silver.market_factor (starts 1963)
            ctx_mf_arrow = con.sql(
                """
                SELECT CAST(date AS TIMESTAMP) AS date,
                       mktrf AS ctx_mktrf,
                       smb AS ctx_smb,
                       hml AS ctx_hml,
                       rmw AS ctx_rmw,
                       cma AS ctx_cma,
                       umd AS ctx_umd
                FROM silver.market_factor
                ORDER BY date
                """
            ).to_arrow_table()

            # Cross-sectional breadth & dispersion from features_daily
            ctx_stock_arrow = con.sql(
                f"""
                SELECT date,
                       ctx_breadth, ctx_dispersion
                FROM {sql_table}
                WHERE in_universe
                GROUP BY ALL
                ORDER BY date
                """
            ).to_arrow_table()
        finally:
            con.close()

        pldf = pl.from_arrow(arrow_table)

        # 1. BUG-04: Cross-sectional Blom normal scores with per-feature non-null denominator
        print(
            "Computing cross-sectional Blom normal scores across active universe (BUG-04 fix)..."
        )
        rank_exprs = [
            pl.col(c).rank("average").over("date").alias(f"{c}_r")
            for c in STOCK_FEATURE_COLS
        ]
        pldf = pldf.with_columns(rank_exprs)

        u_exprs = [
            pl.when(pl.col(c).count().over("date") >= 100)
            .then((pl.col(f"{c}_r") - 0.375) / (pl.col(c).count().over("date") + 0.25))
            .otherwise(None)
            .alias(f"{c}_u")
            for c in STOCK_FEATURE_COLS
        ]
        pldf = pldf.with_columns(u_exprs)

        blom_arrays = []
        for c in STOCK_FEATURE_COLS:
            u = pldf[f"{c}_u"].to_numpy()
            b = sp.ndtri(u)
            blom_arrays.append(b)
        stock_blom_mat = np.column_stack(blom_arrays).astype(np.float32)

        # 2. BUG-05: Causal rolling 252d context normalization ending at s-1 (no zero filling)
        print(
            "Computing causal rolling 252-day context z-scores from full factor history (BUG-05 fix)..."
        )
        style_cols = ["ctx_smb", "ctx_hml", "ctx_rmw", "ctx_cma", "ctx_umd"]
        mf_pl = (
            pl.from_arrow(ctx_mf_arrow)
            .sort("date")
            .with_columns(pl.col("date").cast(pl.Datetime("ns")))
        )
        mf_pl = mf_pl.with_columns(
            [
                _rolling_std(
                    pl.col("ctx_mktrf"), window_size=20, min_samples=10, ddof=1
                ).alias("ctx_mkt_vol_20"),
                pl.col("ctx_mktrf")
                .rolling_sum(window_size=20, min_samples=10)
                .alias("ctx_mkt_ret_20"),
                pl.concat_list(style_cols)
                .list.eval(pl.element().std(ddof=1))
                .list.first()
                .alias("ctx_factor_disp"),
            ]
        )
        factor_cols = [
            "ctx_mktrf",
            "ctx_smb",
            "ctx_hml",
            "ctx_rmw",
            "ctx_cma",
            "ctx_umd",
            "ctx_mkt_vol_20",
            "ctx_mkt_ret_20",
            "ctx_factor_disp",
        ]
        mf_stats = mf_pl.with_columns(
            [
                _rolling_mean(
                    pl.col(c).shift(1), window_size=252, min_samples=60
                ).alias(f"{c}_mean")
                for c in factor_cols
            ]
            + [
                _rolling_std(
                    pl.col(c).shift(1), window_size=252, min_samples=60, ddof=1
                ).alias(f"{c}_std")
                for c in factor_cols
            ]
        )
        mf_norm = mf_stats.select(
            ["date"]
            + [
                pl.when(pl.col(f"{c}_std") > 0)
                .then((pl.col(c) - pl.col(f"{c}_mean")) / pl.col(f"{c}_std"))
                .otherwise(None)
                .alias(c)
                for c in factor_cols
            ]
        )

        cs_pl = (
            pl.from_arrow(ctx_stock_arrow)
            .sort("date")
            .with_columns(pl.col("date").cast(pl.Datetime("ns")))
        )
        cs_cols = ["ctx_breadth", "ctx_dispersion"]
        cs_stats = cs_pl.with_columns(
            [
                _rolling_mean(
                    pl.col(c).shift(1), window_size=252, min_samples=60
                ).alias(f"{c}_mean")
                for c in cs_cols
            ]
            + [
                _rolling_std(
                    pl.col(c).shift(1), window_size=252, min_samples=60, ddof=1
                ).alias(f"{c}_std")
                for c in cs_cols
            ]
        )
        cs_norm = cs_stats.select(
            ["date"]
            + [
                pl.when(pl.col(f"{c}_std") > 0)
                .then((pl.col(c) - pl.col(f"{c}_mean")) / pl.col(f"{c}_std"))
                .otherwise(None)
                .alias(c)
                for c in cs_cols
            ]
        )

        # 3. Build unique date and permno indices
        unique_dates = np.sort(
            np.asarray(pldf["date"].unique().to_numpy(), dtype="datetime64[ns]")
        )
        self.unique_dates = unique_dates
        self.date_to_idx = {d: i for i, d in enumerate(unique_dates)}

        unique_permnos = np.sort(pldf["permno"].unique().to_numpy()).astype(np.int64)
        self.unique_permnos = unique_permnos
        self.permno_to_idx = {p: i for i, p in enumerate(unique_permnos)}

        date_df = pl.DataFrame({"date": unique_dates})
        ctx_joined = date_df.join(mf_norm, on="date", how="left").join(
            cs_norm, on="date", how="left"
        )
        context_norm = (
            ctx_joined.select(CONTEXT_FEATURE_COLS).to_numpy().astype(np.float32)
        )

        # 4. Map stock features into 3D grid [n_permnos, n_dates, 19]
        print(
            "Assembling 3D historical feature tensor and full window validity mask (BUG-05 fix)..."
        )
        row_p_idx = np.array(
            [self.permno_to_idx[p] for p in pldf["permno"].to_numpy()], dtype=np.int32
        )
        row_d = np.asarray(pldf["date"].to_numpy(), dtype="datetime64[ns]")
        row_d_idx = np.array([self.date_to_idx[d] for d in row_d], dtype=np.int32)

        # Per-stock-day validity: present in universe, features_ready is True, and all 19 features non-NaN
        ready_raw = pldf["features_ready"].to_numpy()
        finite_features = np.all(np.isfinite(stock_blom_mat), axis=1)
        daily_valid_row = ready_raw & finite_features

        valid_daily_grid = np.zeros(
            (len(unique_permnos), len(unique_dates)), dtype=bool
        )
        valid_daily_grid[row_p_idx, row_d_idx] = daily_valid_row

        # Cumulative sum along time dimension for exact O(1) 60-session window validity
        cs_valid = np.pad(
            np.cumsum(valid_daily_grid.astype(np.int32), axis=1), ((0, 0), (1, 0))
        )

        grid = np.zeros(
            (len(unique_permnos), len(unique_dates), len(STOCK_FEATURE_COLS)),
            dtype=np.float32,
        )
        grid[row_p_idx, row_d_idx] = np.nan_to_num(stock_blom_mat, nan=0.0)
        self.grid = grid
        self.context_norm = context_norm

        # 5. Extract per-date target arrays, permnos, and validity masks
        print("Indexing evaluation dates and validity masks...")
        target_5d_raw = pldf["target_5d"].to_numpy()
        ystar_raw = pldf["y_star"].to_numpy()
        permno_raw = pldf["permno"].to_numpy().astype(np.int64)

        # Group rows by date index
        split_points = np.flatnonzero(row_d_idx[1:] != row_d_idx[:-1]) + 1
        starts = np.concatenate([[0], split_points])
        ends = np.concatenate([split_points, [len(pldf)]])

        date_permno_indices = {}
        date_permnos = {}
        date_targets = {}
        date_ystar = {}
        date_masks = {}

        for st, en in zip(starts, ends):
            d_i = int(row_d_idx[st])
            p_slice = row_p_idx[st:en]
            p_ids = permno_raw[st:en]
            t5 = target_5d_raw[st:en]
            ys = ystar_raw[st:en]

            # BUG-05 Step 5.3: Require full 60-session window validity [d_i - 59, d_i]
            if d_i >= self.seq_len - 1:
                w_count = (
                    cs_valid[p_slice, d_i + 1]
                    - cs_valid[p_slice, d_i - self.seq_len + 1]
                )
                window_valid = w_count == self.seq_len
            else:
                window_valid = np.zeros(len(p_slice), dtype=bool)

            # Valid target mask: full 60-session window valid and targets not NaN
            mask = window_valid & np.isfinite(t5) & np.isfinite(ys)

            date_permno_indices[d_i] = p_slice.astype(np.int32)
            date_permnos[d_i] = p_ids
            date_targets[d_i] = np.nan_to_num(t5.astype(np.float32), nan=0.0)
            date_ystar[d_i] = np.nan_to_num(ys.astype(np.float32), nan=0.0)
            date_masks[d_i] = mask.astype(bool)

        self.date_permno_indices = date_permno_indices
        self.date_permnos = date_permnos
        self.date_targets = date_targets
        self.date_ystar = date_ystar
        self.date_masks = date_masks

        # BUG-05 Step 5.2: Enforce 332-session warmup (d_i >= 331, first prediction date ~2001-10)
        warmup_cutoff = 331  # 332nd trading session (0-indexed 331)
        self.eval_date_indices = np.array(
            [d_i for d_i in sorted(date_permno_indices.keys()) if d_i >= warmup_cutoff],
            dtype=np.int32,
        )
        self._compute_and_validate_ynorm()

        if cache_file:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving pre-processed tensors to {cache_file}...")
            np.savez_compressed(
                cache_file,
                unique_dates=self.unique_dates,
                unique_permnos=self.unique_permnos,
                grid=self.grid,
                context_norm=self.context_norm,
                date_permno_indices=self.date_permno_indices,
                date_permnos=self.date_permnos,
                date_targets=self.date_targets,
                date_ystar=self.date_ystar,
                date_masks=self.date_masks,
                eval_date_indices=self.eval_date_indices,
            )
            print("Cache saved successfully.")


class Russell1000Dataset(Dataset):
    """PyTorch Dataset yielding 60-session stock and context tensors per decision date."""

    def __init__(
        self,
        store: FeatureDataStore,
        target_dates: np.ndarray | Sequence,
    ):
        self.store = store
        target_set = set(np.asarray(target_dates, dtype="datetime64[D]"))

        # Filter store date indices to matching target dates
        self.date_indices = np.array(
            [
                d_i
                for d_i in self.store.eval_date_indices
                if np.datetime64(self.store.unique_dates[d_i], "D") in target_set
            ],
            dtype=np.int32,
        )

    def __len__(self) -> int:
        return len(self.date_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        d_i = int(self.date_indices[idx])
        st_i = d_i - self.store.seq_len + 1  # 60 sessions: [d_i - 59, d_i]

        p_indices = self.store.date_permno_indices[d_i]
        n_stocks = len(p_indices)
        k_max = self.store.n_max_stocks

        # 1. Padded stock tensor [N_max, 60, 19]
        x_stock = np.zeros((k_max, self.store.seq_len, 19), dtype=np.float32)
        if n_stocks > 0:
            actual_n = min(n_stocks, k_max)
            x_stock[:actual_n] = self.store.grid[
                p_indices[:actual_n], st_i : d_i + 1, :
            ]

        # 2. Context tensor [60, 11]
        x_ctx = self.store.context_norm[st_i : d_i + 1, :]

        # 3. Padded target and mask tensors [N_max]
        target_5d = np.zeros(k_max, dtype=np.float32)
        y_star = np.zeros(k_max, dtype=np.float32)
        y_norm = np.zeros(k_max, dtype=np.float32)
        mask = np.zeros(k_max, dtype=bool)
        permno = np.zeros(k_max, dtype=np.int64)

        if n_stocks > 0:
            actual_n = min(n_stocks, k_max)
            target_5d[:actual_n] = self.store.date_targets[d_i][:actual_n]
            y_star[:actual_n] = self.store.date_ystar[d_i][:actual_n]
            mask[:actual_n] = self.store.date_masks[d_i][:actual_n]
            permno[:actual_n] = self.store.date_permnos[d_i][:actual_n]
            if hasattr(self.store, "date_ynorm") and d_i in self.store.date_ynorm:
                y_norm[:actual_n] = self.store.date_ynorm[d_i][:actual_n]
            else:
                valid_m = mask[:actual_n]
                n_v = int(valid_m.sum())
                if n_v > 0:
                    ranks = rankdata(y_star[:actual_n][valid_m], method="average")
                    y_norm[:actual_n][valid_m] = sp.ndtri(
                        ranks / (float(n_v) + 1.0)
                    ).astype(np.float32)

        return {
            "x_stock": torch.from_numpy(x_stock),
            "x_ctx": torch.from_numpy(x_ctx),
            "target_5d": torch.from_numpy(target_5d),
            "y_star": torch.from_numpy(y_star),
            "y_norm": torch.from_numpy(y_norm),
            "mask": torch.from_numpy(mask),
            "permno": torch.from_numpy(permno),
            "date": str(np.datetime64(self.store.unique_dates[d_i], "D")),
        }


def collate_dates(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate date samples into batched tensors."""
    return {
        "x_stock": torch.stack([b["x_stock"] for b in batch]),  # [B, K, 60, 19]
        "x_ctx": torch.stack([b["x_ctx"] for b in batch]),  # [B, 60, 11]
        "target_5d": torch.stack([b["target_5d"] for b in batch]),  # [B, K]
        "y_star": torch.stack([b["y_star"] for b in batch]),  # [B, K]
        "y_norm": torch.stack([b["y_norm"] for b in batch]),  # [B, K]
        "mask": torch.stack([b["mask"] for b in batch]),  # [B, K]
        "permno": torch.stack([b["permno"] for b in batch]),  # [B, K]
        "date": [b["date"] for b in batch],
    }


def create_dataloader(
    dataset: Russell1000Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Create a deterministic PyTorch DataLoader."""
    g = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        collate_fn=collate_dates,
        drop_last=False,
    )
