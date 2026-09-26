import pandas as pd


class PurgedWalkForwardSplitter:
    """Expanding window walk-forward validation with purging and embargo to prevent leakage in financial time series."""

    def __init__(self, n_folds: int = 5, purge_days: int = 5, embargo_days: int = 20):
        self.n_folds = n_folds
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(
        self, df: pd.DataFrame, date_col: str = "Date"
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Returns n_folds of (train_df, test_df) expanding window splits."""
        unique_dates = sorted(df[date_col].unique())
        total_days = len(unique_dates)
        test_size = total_days // (self.n_folds + 1)

        folds = []
        for i in range(self.n_folds):
            test_start_idx = total_days - (self.n_folds - i) * test_size
            test_end_idx = test_start_idx + test_size
            burn_in_start = max(0, test_start_idx - 59)
            test_dates = unique_dates[burn_in_start:test_end_idx]

            train_end_idx = test_start_idx - self.purge_days - self.embargo_days
            train_dates = unique_dates[:train_end_idx]

            train_df = df[df[date_col].isin(train_dates)].copy()
            test_df = df[df[date_col].isin(test_dates)].copy()

            folds.append((train_df, test_df))

            print(f"Fold {i + 1}:")
            print(
                f"  Train: {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')} ({len(train_dates)} days)"
            )
            print(f"  [PURGE]: {self.purge_days} days (Blackout)")
            print(
                f"  Test : {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')} ({len(test_dates)} days)"
            )

        return folds


if __name__ == "__main__":
    import numpy as np

    dates = pd.date_range(start="2020-01-01", periods=1601, freq="B")
    dummy_df = pd.DataFrame({"Date": dates, "Val": np.random.randn(1601)})
    splitter = PurgedWalkForwardSplitter(n_folds=5, purge_days=5)
    folds = splitter.split(dummy_df)
