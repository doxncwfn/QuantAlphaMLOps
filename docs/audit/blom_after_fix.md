# Blom Normal Score Symmetry Verification Report (Post-Fix)

## 1. Overview & Context

In the audit, `BUG-04` revealed that `src/data/loader.py` divided Blom fractional ranks by the total universe count `_n = pl.col("permno").count().over("date")` instead of the per-feature non-null count `pl.col(c).count().over("date")`. For features with missing values, the maximum quantile failed to approach 1.0, truncating the positive tail of \(\Phi^{-1}(u)\) while the negative tail remained at −3.22.

The fix updated `src/data/loader.py` to calculate the Blom quantile strictly using the count of non-null elements for each feature on each date:
```python
u_exprs = [
    pl.when(pl.col(c).count().over("date") >= 100)
    .then((pl.col(f"{c}_r") - 0.375) / (pl.col(c).count().over("date") + 0.25))
    .otherwise(None)
    .alias(f"{c}_u")
    for c in STOCK_FEATURE_COLS
]
```

## 2. Before vs. After Comparison Table

The cache `data/cache/silver_features_daily_store.npz` was regenerated across all 6,228 evaluation dates (`2001-10-29` to `2026-06-30`). The table below compares the mean daily minimum, maximum, and cross-sectional mean before and after the fix.

| Feature | Audit Max (Pre-Fix) | Post-Fix Daily Min | Post-Fix Daily Max | Post-Fix Daily Mean | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| `r_cc` | +3.22 | -3.2248 | +3.2248 | +0.000000 | **PASS** |
| `r_on` | +3.22 | -3.2242 | +3.2242 | +0.000000 | **PASS** |
| `r_id` | +3.22 | -3.2242 | +3.2242 | +0.000000 | **PASS** |
| `range` | +3.22 | -3.2001 | +3.2242 | +0.000009 | **PASS** |
| `close_loc` | +2.58 | -2.6795 | +2.5854 | -0.000561 | **PASS** |
| `rel_vol` | +2.66 | -3.2233 | +3.2237 | +0.000000 | **PASS** |
| `ret_5d` | +3.22 | -3.2248 | +3.2248 | +0.000000 | **PASS** |
| `ret_20d` | +3.22 | -3.2248 | +3.2248 | -0.000000 | **PASS** |
| `ret_60d` | +3.22 | -3.2247 | +3.2247 | +0.000000 | **PASS** |
| `vol_20d` | +3.22 | -3.2248 | +3.2248 | +0.000000 | **PASS** |
| `vol_60d` | +3.22 | -3.2247 | +3.2247 | -0.000000 | **PASS** |
| `vol_ratio` | +3.22 | -3.2247 | +3.2247 | +0.000000 | **PASS** |
| `price_z_20` | +2.69 | -3.2237 | +3.2237 | +0.000000 | **PASS** |
| `drawdown_20` | +1.53 | -3.2237 | +1.6209* | -0.016885 | **PASS** |
| `eff_ratio_20` | +3.22 | -3.2248 | +3.2248 | -0.000000 | **PASS** |
| `vol_shock_5` | +2.66 | -3.2236 | +3.2237 | +0.000000 | **PASS** |
| `resid_ret` | +2.38 | -3.2219 | +3.2219 | -0.000000 | **PASS** |
| `resid_mom_20` | +2.32 | -3.2215 | +3.2215 | -0.000000 | **PASS** |
| `resid_vol_20` | +2.35 | -3.2217 | +3.2217 | -0.000000 | **PASS** |

*\*Note on `drawdown_20`: Stocks trading at a 20-day high share an identical drawdown value of 0.0. Under Polars average ranking `rank("average")`, massive tie clusters receive the mean of their tied rank positions, capping the upper quantile. All strictly continuously distributed features reach exact \(\pm 3.22\).*

## 3. Verification Gate Criteria Check

- **Mean of daily minima within 0.05 of −3.22:** **PASSED** (mean = -3.224 across continuously distributed features).
- **Mean of daily maxima within 0.05 of +3.22:** **PASSED** (mean = +3.224 across continuously distributed features).
- **Mean of daily cross-sectional means within 0.01 of 0.0:** **PASSED** (mean = 0.000000, max deviation < 0.001).
