"""Evaluation metrics.

Contract: fn(df: pd.DataFrame[date, security_id, score, label], **kwargs) -> MetricResult.
df holds out-of-sample predictions joined with labels; rows without a label are already dropped.
Compute within each date first, then aggregate over dates.

Labels over h days overlap from one date to the next, so per-date series are autocorrelated.
Annualisation is therefore a parameter, not an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qam.registry import register


@dataclass
class MetricResult:
    value: float
    per_date: pd.Series | None = None


def daily_ic(df: pd.DataFrame, min_stocks: int = 20) -> pd.Series:
    """Spearman rank correlation between score and label, one value per date."""

    def one(group: pd.DataFrame) -> float:
        if len(group) < min_stocks or group["score"].std() < 1e-12:
            return np.nan
        return group["score"].rank().corr(group["label"].rank())

    return df.groupby("date")[["score", "label"]].apply(one).dropna().rename("ic")


def _buckets(df: pd.DataFrame, n: int) -> pd.Series:
    """Equal-count score buckets 1..n per date (integer arithmetic avoids float edge errors)."""
    rank = df.groupby("date")["score"].rank(method="first").astype(int)
    count = df.groupby("date")["score"].transform("size")
    return (rank - 1) * n // count + 1


@register("metric", "mean_ic")
def mean_ic(df: pd.DataFrame, min_stocks: int = 20) -> MetricResult:
    ic = daily_ic(df, min_stocks)
    return MetricResult(float(ic.mean()), ic)


@register("metric", "icir")
def icir(df: pd.DataFrame, min_stocks: int = 20, periods_per_year: float | None = None) -> MetricResult:
    """mean(IC) / std(IC), times sqrt(periods_per_year) when given."""
    ic = daily_ic(df, min_stocks)
    value = ic.mean() / ic.std() if len(ic) > 1 and ic.std() > 0 else np.nan
    if periods_per_year:
        value *= np.sqrt(periods_per_year)
    return MetricResult(float(value))


@register("metric", "ic_hit_rate")
def ic_hit_rate(df: pd.DataFrame, min_stocks: int = 20) -> MetricResult:
    """Share of dates with a positive IC."""
    ic = daily_ic(df, min_stocks)
    return MetricResult(float((ic > 0).mean()))


@register("metric", "decile_spread")
def decile_spread(df: pd.DataFrame, n: int = 10, cost_bps: float = 0.0, min_stocks: int = 20) -> MetricResult:
    """Mean label of the top score bucket minus the bottom bucket, per date, net of cost_bps.

    Uses the forward label (the legacy version used the same-day return). This is a ranking
    diagnostic, not portfolio P&L; P&L belongs to the backtest stage.
    """
    counts = df.groupby("date")["score"].transform("size")
    data = df[counts >= min_stocks].assign(bucket=lambda d: _buckets(d, n))
    means = data.groupby(["date", "bucket"])["label"].mean().unstack()
    spread = (means[n] - means[1] - cost_bps / 1e4).dropna().rename("spread")
    return MetricResult(float(spread.mean()), spread)


@register("metric", "decile_monotonicity")
def decile_monotonicity(df: pd.DataFrame, n: int = 10, min_stocks: int = 20) -> MetricResult:
    """Spearman correlation between bucket number and the average per-date bucket label."""
    counts = df.groupby("date")["score"].transform("size")
    data = df[counts >= min_stocks].assign(bucket=lambda d: _buckets(d, n))
    profile = data.groupby(["date", "bucket"])["label"].mean().groupby("bucket").mean()
    value = pd.Series(profile.index.to_numpy(dtype=float)).corr(pd.Series(profile.to_numpy()), method="spearman")
    return MetricResult(float(value))
