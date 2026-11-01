"""Stock features (raw and derived time series).

Contract: fn(**kwargs) -> pl.Expr aliased to a unique column name. The framework
evaluates it per security, on rows sorted by date: panel.with_columns(expr.over("security_id")).
Only past and current rows may be used (rolling windows, shifts >= 0); never look forward.
Values stay raw here; cross-sectional normalisation happens at load time (transforms.py).
"""

from __future__ import annotations

import polars as pl

from qam.registry import register


def _log_ret() -> pl.Expr:
    return pl.col("ret_total").log1p()


@register("feature", "log_ret")
def log_ret(clip: list[float] | None = None) -> pl.Expr:
    """Daily log total return. Optional clip, e.g. [-0.2231, 0.1823] = [ln 0.8, ln 1.2]."""
    expr = _log_ret()
    if clip is not None:
        expr = expr.clip(clip[0], clip[1])
    return expr.alias("log_ret")


@register("feature", "vol")
def vol(n: int = 20) -> pl.Expr:
    """Rolling standard deviation of daily log returns over n rows."""
    return _log_ret().rolling_std(window_size=n, min_samples=n).alias(f"vol_{n}")


@register("feature", "dist_sma")
def dist_sma(n: int = 20) -> pl.Expr:
    """Distance of the split-adjusted close to its n-day simple moving average."""
    sma = pl.col("adj_close").rolling_mean(window_size=n, min_samples=n)
    return (pl.col("adj_close") / sma - 1.0).alias(f"dist_sma_{n}")


@register("feature", "log_dollar_volume")
def log_dollar_volume(n: int = 1) -> pl.Expr:
    """log(1 + mean of close * volume over n rows); raw close is correct for traded value."""
    dollar = pl.col("close") * pl.col("volume")
    if n > 1:
        dollar = dollar.rolling_mean(window_size=n, min_samples=n)
    return dollar.log1p().alias("log_dollar_volume" if n == 1 else f"log_dollar_volume_{n}")


@register("feature", "raw")
def raw(column: str, alias: str | None = None) -> pl.Expr:
    """Pass a panel column through unchanged (e.g. volume)."""
    return pl.col(column).cast(pl.Float64).alias(alias or column)
