# 02 · Formulation and inference flow

**Status:** decided. Notation is defined in [README.md](README.md).

## Inference at date t

```
U_t ──► Filter (cached for TTL) ──► k stocks
                                      │
         stock history over [t − L, t) ┤
   m economic indicators over [t − L, t) ┤──► Model ──► scores s_0 … s_{k−1} ──► sort ──► ranking ──► long–short decision (planned)
               non-time-series features ┘
```

1. **Universe → Filter.** Take the universe U_t. The Filter returns k stocks. The list is cached and reused until its TTL expires ([03_filter.md](03_filter.md)).
2. **Inputs.**
   - For each of the k stocks: its time series over [t − L, t).
   - The m economic indicators over the same window.
   - Features that are not time series.
3. **Model.** Outputs one score per stock: s_i for i = 0 … k − 1, in the same order as the input stocks ([05_model.md](05_model.md)).
4. **Ranking.** Sorting s gives the predicted ranking for the horizon t → t + h.
5. **Decision.** Long–short selection from the ranking. Planned, not yet specified.

## Training signal

At training time the realized returns r_i over t → t + h are known. Where the return starts is set by `offset` (see the notation in [README.md](README.md)); the default 0 matches the legacy code. The score vector is judged by the rank correlation between s and r for that date ([04_objective.md](04_objective.md)).

## Rules that follow from the formulation

- **Point-in-time.** Everything used at date t (the universe, the Filter's choice, features) must use only information available before t. Anything else is look-ahead bias.
- **Cross-sectional.** Scores are compared only among the k stocks of the **same date**. Scores from different dates are not comparable.
- **The order of stocks does not carry meaning.** The model must not depend on which position a stock occupies in the input ([05_model.md](05_model.md)).
