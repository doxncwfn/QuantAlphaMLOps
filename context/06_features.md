# 06 · Feature selection and engineering

**Status:** in progress. **Owners:** Cường (domain knowledge, spec), Hào (implementation)

Spec to be agreed, then implemented.

## Feature groups (from the model design)

| Group | Shape | Goes to |
|---|---|---|
| Stock time series | k series × L steps | LSTM, then stock tokens |
| Economic indicators, time series | m' series × L steps, fixed order | LSTM, then indicator tokens |
| Economic indicators, not time series | m − m' values | TFM directly, as separate tokens |
| Stock features that are not time series | per stock | open (see [05_model.md](05_model.md)) |

## Requirements for any feature

- **Point-in-time:** computed only from data before t.
- **Comparable across stocks:** stock order and identity carry no meaning, so features should be scale-free (returns, ratios, z-scores within the date) rather than raw price levels.
- **Available for delisted stocks too:** features must be computable from the warehouse data for the whole history, not only for stocks that still trade.

## Current state in the repository

The legacy `legacy/src/data/residualizer.py` uses Open, High, Low, Close, Volume, 60-day volatility, and the distance to the 20-day and 60-day moving averages. Each is z-scored within the date and clipped to ±3. This is a starting point, not the agreed spec. These features are ported to `src/qam/components/features.py`, and `configs/experiments/legacy_parity.yaml` reproduces the legacy feature set.

## Data available for features

From the warehouse (`../../warehouse/SPEC.md`), daily per stock: OHLCV, shares outstanding, the adjustment factor for splits, total return, delisting return, and point-in-time ticker, exchange and industry codes (SIC, NAICS). Market returns (value-weighted, equal-weighted, S&P 500) are kept in the bronze layer.
