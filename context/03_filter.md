# 03 · Filter

**Status:** rule and definitions decided; all values are parameters. **Owner:** Hào

## Purpose

Select **k stocks** from the universe that **represent** it. The selection should be both **liquid**, so the stocks can actually be traded, and **stable**, so the list does not churn at every refresh. k is fixed for a run.

## Parameters

Every value is a parameter of the Filter; nothing is hard-coded.

| Parameter | Meaning | Example |
|---|---|---|
| `k` | Number of stocks to select | 500 |
| `buffer_ratio` | Retention buffer as a fraction of k: b = ⌊k × buffer_ratio⌋ | 0.1 (b = 50) |
| `x` | Liquidity window: number of trading days of dollar volume aggregated | — |
| `y` | Recency window in trading days; a stock needs data on **at least x** of the last y days. Requires y ≥ x | — |
| `ttl` | Time between refreshes: t_{j+1} = t_j + ttl | — |
| `agg` | How dollar volume is aggregated over the x days. Pluggable; **default: sum** | sum |

The original notes call the target count *n* and the refresh index *k*. Here **k** is the count, matching the rest of the context, and **j** is the refresh index.

## Definitions

- **Starting universe at t_j:** every stock in the **master list** that is **trading at t_j**, whether or not it is a Russell 1000 member at that date.
- **Dollar volume** of a stock on day d: price × shares traded. With the warehouse data: `close × volume` from `silver.prices_daily`. Days without trading contribute no volume.
- **Liquidity score:** `agg` over the stock's daily dollar volume in the x trading days before t_j. Default: the sum.
- **Eligible:** the stock has data on at least x of the last y trading days before t_j.

## Rule

At each refresh date t_j:

1. **Eligibility.** From the starting universe, keep the eligible stocks. Call this set E_j.
2. **Liquidity score** for each stock in E_j.
3. **Rank** E_j by liquidity, highest first. Let T_j be the **top k + b**.
4. **Retain incumbents.** Every stock of the previous list S_{j−1} that is still in T_j stays: R_j = S_{j−1} ∩ T_j.
5. **Fill the rest by rank.** Add the highest-ranked stocks of T_j not already in R_j until the list has k stocks. This gives S_j.
6. **First refresh (j = 0):** there is no previous list, so S_0 is simply the top k.
7. **Fewer than k eligible stocks** (|E_j| < k): select **all** eligible stocks and fill the remaining slots with **masked padding** so the list still has k slots. The same applies when fewer than k + b are eligible: T_j is then all of E_j.

```
S_j = R_j  ∪  top (k − |R_j|) of ( T_j \ R_j )      where R_j = S_{j−1} ∩ T_j
```

Between refreshes, every inference uses the cached S_j.

## Why the buffer

Without a buffer, a stock ranked around position k would enter and leave the list at every refresh as its volume moves slightly. With the buffer, an incumbent only leaves when it falls **below rank k + b**. A newcomer still has to be good enough to fill a free slot. The trade-off is **liquidity vs stability**, and `buffer_ratio` sets the tolerance. Russell uses the same idea ("banding") at its own reconstitutions.

Properties:
- The output always has **exactly k slots**. When at least k stocks are eligible, all slots are real stocks; T_j has k + b ≥ k members, so there are always enough candidates. Otherwise the missing slots are masked padding.
- An incumbent ranked between k + 1 and k + b **stays**. A non-incumbent at the same rank does not get in.
- Turnover per refresh is limited to stocks that fell out of the top k + b, plus slots freed by stocks that became ineligible.

## Constraints

- **Point-in-time:** prices, volumes, data availability and universe membership use only information before t_j.
- **Fixed size:** k slots, because the model takes k stock inputs ([05_model.md](05_model.md)). Each slot carries a mask flag: real stock or padding.

## Notes

- **Prices are not adjusted for splits when computing dollar volume.** Using the actual price and actual share count of each day gives the true traded value of that day. No adjustment factor is needed.
- **Starting from the master list:** this includes stocks in years before they joined or after they left the Russell 1000. The liquidity rank decides whether they are selected.

## Open

- **A selected stock stops trading inside the TTL.** For the objective this is decided: training scores only the stocks still present at t + h, and inference assumes all k are present ([04_objective.md](04_objective.md)). Still open: whether the slot is kept until the next refresh or refilled from the ranking.
- Values of the parameters for the experiments, and whether `ttl` should line up with the Russell reconstitution dates.
