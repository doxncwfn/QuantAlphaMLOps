# 05 · Model

**Status:** architecture decided at design level. **Owner:** Cường

Two components: an **LSTM** that embeds each time series independently, and a **Transformer (TFM)** that lets the embeddings interact and produces the scores.

## 1. LSTM: embed each time series

Inputs are **k + m'** time series over [t − L, t):
- **k stock series**, one per selected stock.
  - k is fixed.
  - The **order of stocks need not be the same** between inferences. The model is expected to learn general patterns, not memorize individual stocks.
- **m' economic indicators that are time series**, the time-series part of the m economic indicators.
  - The count and the **order are fixed** between inferences.

Each series is embedded **independently** into a vector.

## 2. TFM: combine and score

The TFM receives one token per item:

| Token | Source | How many |
|---|---|---|
| Stock token | LSTM embedding of a stock series | k |
| Economic indicator token (time series) | LSTM embedding of an indicator series | m' |
| Economic indicator token (not a time series) | the value itself, as its own token | m − m' |

**Output:** k scores, one per stock token, in the **same order as the input stocks**. Indicator tokens give context only and produce no score.

## Design implications

- **Stocks are permutation-equivariant.** Reordering the input stocks must reorder the scores the same way. So stock tokens get **no positional encoding**, and nothing identifies a specific stock. The repository's earlier stock-ID embedding was dropped because it led to memorization (`legacy/src/models/hybrid_embedded.py`).
- **Indicator tokens have fixed identities.** Their order is fixed, so each one can carry a learned type or position embedding that tells the TFM which indicator it is.
- **Fixed sizes.** k and m are fixed, so the TFM sees the same number of tokens at every date.
- **Masked slots.** When the Filter finds fewer than k eligible stocks, the empty slots are padding ([03_filter.md](03_filter.md)). The TFM must take a **padding mask** so real tokens never attend to padded ones, and padded slots produce no usable score.

## Open

- Whether stocks and indicators share one LSTM or use separate LSTMs.
- Where stock-level features that are not time series go (for example sector): appended to the stock token, or something else.
- Embedding size, number of layers and heads. The repository currently uses a 2-layer LSTM and a 2-layer TFM with 4 heads.

## Relation to the current repository

`legacy/src/models/hybrid.py` had the stock part: LSTM per stock, then TFM across stocks with no positional encoding, then a scorer. It is ported as `hybrid_lstm_tfm` in `src/qam/components/models.py`, which adds the padding mask and optional indicator tokens (a separate LSTM per time-series indicator, one projected token per non-time-series indicator, each with a learned type embedding). Which indicators to use is still open ([06_features.md](06_features.md)).
