import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import yaml
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

CONFIG = {
    "data_dir": "data/WRDS",
    "fama_french_dir": "data",
    "force_start_date": "2020-01-01",
    "seq_len": 60,
    "pred_horizon": 5,
    "beta_window": 126,
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def _load_yaml_config() -> dict:
    candidates = [
        Path(__file__).resolve().parents[2] / "config" / "config.yaml",
        Path("config/config.yaml"),
        Path("config.yaml"),
    ]
    for c in candidates:
        if c.exists():
            with open(c, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


_loaded_cfg = _load_yaml_config()
if _loaded_cfg:
    CONFIG.update(_loaded_cfg)



class UniversalDataProcessor:
    def __init__(self, data_dir: str | Path, ff_dir: str | Path, tickers: list[str]):
        self.data_dir = Path(data_dir)
        self.ff_dir = Path(ff_dir)
        self.tickers = sorted(tickers)
        self.stock_to_id: dict[str, int] = {}
        self.feature_cols: list[str] = []
        self.valid_tickers: list[str] = []

    def _rolling_ols_residuals(
        self, returns: np.ndarray, factors: np.ndarray, window: int = 126
    ) -> np.ndarray:
        """Computes trailing 126-day rolling OLS and residualized 5-day forward return."""
        T = len(returns)
        if window + 5 > T:
            return np.full(T, np.nan)
        residuals = np.full(T, np.nan)
        X = np.hstack([np.ones((T, 1)), factors])
        for t in range(window, T - 5):
            X_train = X[t - window : t]
            y_train = returns[t - window : t]
            try:
                beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
                f_forward = np.prod(1 + factors[t + 1 : t + 6], axis=0) - 1
                f_forward_with_alpha = np.insert(f_forward, 0, 1.0)
                r_forward = np.prod(1 + returns[t + 1 : t + 6]) - 1
                residuals[t] = r_forward - (beta @ f_forward_with_alpha)
            except np.linalg.LinAlgError:
                pass
        return residuals

    def load_and_process(self) -> pd.DataFrame:
        print(f"Loading stocks {self.data_dir} using Polars...")
        feature_start = pl.datetime(2018, 1, 1)
        model_start = pl.datetime(2020, 1, 1)

        # Build Master Calendar using all tickers that exist in universe for calendar merge
        date_series = []
        for ticker in self.tickers:
            fpath = self.data_dir / f"{ticker}.csv"
            if fpath.exists():
                date_series.append(pl.scan_csv(fpath).select("Date").collect())

        master_calendar = (
            pl.concat(date_series)
            .unique()
            .with_columns(pl.col("Date").str.to_datetime())
            .filter(pl.col("Date") >= feature_start)
            .sort("Date")
        )

        dfs = []
        for ticker in self.tickers:
            fpath = self.data_dir / f"{ticker}.csv"
            if not fpath.exists():
                continue
            try:
                df = pl.read_csv(fpath)
                df.columns = [c.strip().title() for c in df.columns]
                df = df.with_columns(pl.col("Date").str.to_datetime()).filter(
                    pl.col("Date") >= feature_start
                )
                df = (
                    master_calendar.join(df, on="Date", how="left")
                    .sort("Date")
                    .with_columns(pl.all().fill_null(strategy="forward"))
                    .with_columns(pl.lit(ticker).alias("Ticker"))
                    .drop_nulls(subset=["Close"])
                )
                if len(df) < CONFIG["seq_len"] * 2:
                    continue
                dfs.append(df)
            except (pl.exceptions.PolarsError, OSError, ValueError, KeyError):
                continue

        big_df = pl.concat(dfs)

        print("Computing Point-in-Time Features...")
        big_df = self._engineer_features(big_df)

        print("Merging Fama-French & Computing Residuals...")
        big_df = self._compute_residualized_target(big_df)

        print("Applying Cross-Sectional Z-Scoring...")
        exclude_cols = [
            "Date",
            "Ticker",
            "Target_Residual",
            "Log_Ret",
            "Vol_Quintile",
            "Factor_VOL",
            "Mkt-RF",
            "SMB",
            "HML",
        ]
        self.feature_cols = [c for c in big_df.columns if c not in exclude_cols]

        big_df = big_df.with_columns(
            [
                (
                    (pl.col(c) - pl.col(c).mean().over("Date"))
                    / (pl.col(c).std().over("Date") + 1e-8)
                )
                .clip(-3, 3)
                .alias(c)
                for c in self.feature_cols
            ]
        )

        big_df = big_df.with_columns(
            (
                (pl.col("Target_Residual") - pl.col("Target_Residual").mean().over("Date"))
                / (pl.col("Target_Residual").std().over("Date") + 1e-8)
            ).alias("Target_5D_Z")
        ).drop_nulls()

        # Slice to model_start only
        big_df = big_df.filter(pl.col("Date") >= model_start)

        # Strict Intersection Universe: Keep only stocks with perfect, uninterrupted history
        ticker_day_counts = big_df.group_by("Ticker").agg(pl.len().alias("n_days"))
        max_valid_days = ticker_day_counts["n_days"].max()
        valid_tickers_df = ticker_day_counts.filter(pl.col("n_days") == max_valid_days)
        self.valid_tickers = sorted(valid_tickers_df["Ticker"].to_list())
        self.stock_to_id = {t: i for i, t in enumerate(self.valid_tickers)}

        big_df = big_df.filter(pl.col("Ticker").is_in(self.valid_tickers))

        # Use replace_strict to assign Stock_ID
        big_df = big_df.with_columns(
            pl.col("Ticker")
            .replace_strict(self.stock_to_id, return_dtype=pl.Int32)
            .alias("Stock_ID")
        )

        # Filter to only keep dates where every single stock in the universe has valid data
        target_universe_size = len(self.valid_tickers)

        valid_dates = (
            big_df.group_by("Date")
            .agg(pl.len().alias("stock_count"))
            .filter(pl.col("stock_count") == target_universe_size)
            .select("Date")
        )
        big_df = big_df.join(valid_dates, on="Date", how="inner")

        print(
            f"Fixed Universe (Strict Intersection): {target_universe_size} stocks across {len(valid_dates)} complete trading dates."
        )
        return big_df.to_pandas()

    def _engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                (pl.col("Close").log() - pl.col("Close").shift(1).log().over("Ticker"))
                .clip(lower_bound=np.log(0.80), upper_bound=np.log(1.20))
                .alias("Log_Ret"),
                (pl.col("Close").log() - pl.col("Close").shift(1).log().over("Ticker"))
                .rolling_std(window_size=60)
                .over("Ticker")
                .alias("Vol_60"),
                (
                    (pl.col("Close") / pl.col("Close").rolling_mean(window_size=20).over("Ticker"))
                    - 1
                ).alias("Dist_SMA_20"),
                (
                    (pl.col("Close") / pl.col("Close").rolling_mean(window_size=60).over("Ticker"))
                    - 1
                ).alias("Dist_SMA_60"),
            ]
        )

    def _compute_residualized_target(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.col("Vol_60")
            .qcut(5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
            .over("Date")
            .alias("Vol_Quintile")
        )
        vol_factor_df = (
            df.group_by("Date")
            .agg(
                [
                    pl.col("Log_Ret")
                    .filter(pl.col("Vol_Quintile") == "Q5")
                    .mean()
                    .alias("High_Vol_Ret"),
                    pl.col("Log_Ret")
                    .filter(pl.col("Vol_Quintile") == "Q1")
                    .mean()
                    .alias("Low_Vol_Ret"),
                ]
            )
            .with_columns((pl.col("Low_Vol_Ret") - pl.col("High_Vol_Ret")).alias("Factor_VOL"))
            .select(["Date", "Factor_VOL"])
        )

        # Load local Fama-French factors (WRDS or pre-cached)
        ff_cache_path = self.ff_dir / "F-F_Research_Data_Factors_daily.csv"
        ff_local_csv = Path("data/Russell 1000/ff.csv")
        ff_fallback_csv = self.ff_dir / "ff.csv"

        if ff_cache_path.exists():
            ff_df_api = pl.read_csv(ff_cache_path).with_columns(pl.col("Date").str.to_datetime())
        elif ff_fallback_csv.exists():
            ff_df_api = (
                pl.read_csv(ff_fallback_csv)
                .rename({"date": "Date", "mktrf": "Mkt-RF", "smb": "SMB", "hml": "HML"})
                .with_columns(pl.col("Date").str.to_datetime())
            )
        elif ff_local_csv.exists():
            ff_df_api = (
                pl.read_csv(ff_local_csv)
                .rename({"date": "Date", "mktrf": "Mkt-RF", "smb": "SMB", "hml": "HML"})
                .with_columns(pl.col("Date").str.to_datetime())
            )
        else:
            raise FileNotFoundError(
                f"Fama-French factors not found at '{ff_cache_path}' or '{ff_local_csv}'."
            )

        master_factors = vol_factor_df.join(ff_df_api, on="Date", how="left").drop_nulls()
        df = df.join(master_factors, on="Date", how="inner")

        pandas_df = df.to_pandas()

        def calculate_residuals(group):
            factors = group[["Mkt-RF", "SMB", "HML", "Factor_VOL"]].values
            returns = group["Log_Ret"].values
            residuals = self._rolling_ols_residuals(returns, factors, window=CONFIG["beta_window"])
            return pd.Series(residuals, index=group.index)

        print("   -> Running 126-day Rolling OLS Beta exposures (This will take 60-90 seconds)...")
        pandas_df["Target_Residual"] = pandas_df.groupby("Ticker", group_keys=False).apply(
            calculate_residuals
        )

        return pl.from_pandas(pandas_df)


class UniversalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], seq_len: int, num_stocks: int):
        self.df = df.sort_values(["Date", "Stock_ID"])
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.num_stocks = num_stocks

        self.dates = sorted(df["Date"].unique())
        self.date_to_idx = {date: i for i, date in enumerate(self.dates)}
        self.valid_dates = self.dates[self.seq_len - 1 :]

    def __len__(self) -> int:
        return len(self.valid_dates)

    def __getitem__(self, idx: int):
        date = self.valid_dates[idx]
        window_start = self.date_to_idx[date] - self.seq_len + 1
        window_dates = self.dates[window_start : self.date_to_idx[date] + 1]

        window_df = self.df[self.df["Date"].isin(window_dates)]

        if len(window_df) != self.seq_len * self.num_stocks:
            raise ValueError(f"Missing data in window ending at {date}")

        flat = window_df[self.feature_cols].values.astype(np.float32)
        x = flat.reshape(self.seq_len, self.num_stocks, -1)
        x = np.transpose(x, (1, 0, 2))
        x = torch.FloatTensor(x)

        stock_ids = torch.arange(self.num_stocks)
        y = window_df[window_df["Date"] == date]["Target_5D_Z"].values.astype(np.float32)

        return x, stock_ids, torch.FloatTensor(y)


if __name__ == "__main__":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    cfg_override = _load_yaml_config()
    if cfg_override:
        CONFIG.update(cfg_override)

    processor = UniversalDataProcessor(
        data_dir=CONFIG["data_dir"],
        ff_dir=CONFIG["fama_french_dir"],
        tickers=CONFIG.get("tickers", []),
    )
    df = processor.load_and_process()
    print(f"Final Dataset Shape: {df.shape}")
