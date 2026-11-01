# 04 · Ranking criterion and objective

**Status:** criterion and missing-stock handling decided; return definition open. **Owner:** Hiếu

## Criterion

At date t the model outputs scores s = [s_0 … s_{k−1}] for horizon h. With realized returns r = [r_0 … r_{k−1}] from t to t + h:

```
IC_t = Spearman( s, r )        # rank correlation computed within date t only
```

**Objective: maximize IC_t**, averaged over dates. Equivalently, minimize

```
loss = − mean over t of IC_t
```

IC_t is computed over **real stocks only**: masked padding slots ([03_filter.md](03_filter.md)) are excluded from the scores, the returns and the loss.

Spearman is used because only the **order** of the scores matters ([01_problem.md](01_problem.md)). It ignores the scale of the scores and is not dominated by a few extreme returns.

## Stocks with no value at t + h

A stock selected at t can have no value at t + h. Two cases:

| Case | What happened | Can it still be traded? | Is its ranking meaningful? |
|---|---|---|---|
| 1. Temporary null | The stock still trades; the data collection is missing values | Yes | Yes |
| 2. Permanent null | The stock was delisted at or before t + h | No | — (rare) |

**Training: score only the stocks that are present.** If k stocks were selected at t and only j of them still have a value at t + h, IC_t is computed on those **j** stocks only, both scores and returns. This works because a correlation does not depend on the number of elements. The other k − j outputs are ignored, like masked slots.

**Inference: assume all k stocks will be present.** No special handling. The model ranks all k stocks; the risk that some of them get delisted before t + h is **accepted**.

### Consequences to keep in mind

- **The model is never trained on delisting outcomes.** Stocks that delist before t + h drop out of the training target, so a stock that is about to be acquired or go bankrupt is never rewarded or penalized for its delisting. The data stays survivorship-free, but within each horizon the objective only sees survivors. The effect is small when h is short and delistings are rare.
- **Backtests must still use delisting returns.** The model's ranking is scored without them, but the P&L of a long–short portfolio must include the delisting return (`silver.v_prices.ret_total`) of any held stock that delists. Otherwise performance is overstated.
- Case 1 rarely leaves a real gap in CRSP: when a stock does not trade for a few days, the return over the gap is usually still recorded. A stock is only excluded when its return from t to t + h cannot be computed.

## Implementation note

Spearman is computed on ranks, and ranks are not differentiable. Training therefore needs a **differentiable surrogate**, for example:
- a soft-rank approximation of Spearman;
- Pearson correlation between the scores and the ranks of the returns;
- a pairwise or listwise ranking loss.

Evaluation still uses the exact Spearman IC. Which surrogate to use is open.

The legacy code trains with a different loss: a hinge on top vs bottom decile plus MSE (`legacy/src/models/losses.py`). It is ported as `hybrid_hinge` in `src/qam/components/losses.py`, next to `neg_pearson_rank`, a first Spearman surrogate.

## Open

- **Which return r_i:** raw, excess over the market, or residual after factor exposures (the current repository uses a residual on Fama-French 3 + a volatility factor). Delisting returns must be included in any choice.
- **Which surrogate loss** to use.
- **Horizon h:** example value 5 days; not fixed yet.
