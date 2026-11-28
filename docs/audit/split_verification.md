# Audit Step 4 — Train/Val/Test Partition & Data Loader Verification (`src/training/wfcv.py` & `src/data/loader.py`)

## 4.1 Exact Partition Ranges & Purge/Embargo Verification

Source: [`src/training/wfcv.py:45-114`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/wfcv.py#L45-L114)

### Reproducible Code & Output

```python
import numpy as np
from src.data.loader import FeatureDataStore
from src.training.wfcv import purged_walk_forward_cv

store = FeatureDataStore(cache_dir="data/cache")
eval_dates = store.unique_dates[store.eval_date_indices]
date_to_eval_idx = {d: i for i, d in enumerate(eval_dates)}
splits = list(purged_walk_forward_cv(eval_dates, n_splits=5, horizon=5, embargo=5))

for s in splits:
    tr_end = date_to_eval_idx[s.train_dates[-1]]
    va_start = date_to_eval_idx[s.val_dates[0]]
    va_end = date_to_eval_idx[s.val_dates[-1]]
    te_start = date_to_eval_idx[s.test_dates[0]]
    gap_tr_va = va_start - tr_end
    gap_va_te = te_start - va_end
    ov_tv = len(set(s.train_dates).intersection(s.val_dates))
    ov_tt = len(set(s.train_dates).intersection(s.test_dates))
    ov_vt = len(set(s.val_dates).intersection(s.test_dates))
    print(
        f"Fold {s.fold_idx}: "
        f"Train={str(s.train_start)[:10]}..{str(s.train_end)[:10]} ({len(s.train_dates)}) | "
        f"gap_tr_va={gap_tr_va} (>=10: {gap_tr_va >= 10}) | "
        f"Val={str(s.val_start)[:10]}..{str(s.val_end)[:10]} ({len(s.val_dates)}) | "
        f"gap_va_te={gap_va_te} (>=10: {gap_va_te >= 10}) | "
        f"Test={str(s.test_start)[:10]}..{str(s.test_end)[:10]} ({len(s.test_dates)}) | "
        f"overlaps=({ov_tv}, {ov_tt}, {ov_vt})"
    )
```

```text
Fold 1: Train=2000-09-25..2005-01-26 (1089) | gap_tr_va=10 (>=10: True) | Val=2005-02-09..2005-11-11 (193) | gap_va_te=10 (>=10: True) | Test=2005-11-28..2011-01-26 (1300) | overlaps=(0, 0, 0)
Fold 2: Train=2000-09-25..2009-06-17 (2194) | gap_tr_va=10 (>=10: True) | Val=2009-07-01..2011-01-12 (388) | gap_va_te=10 (>=10: True) | Test=2011-01-27..2016-03-29 (1300) | overlaps=(0, 0, 0)
Fold 3: Train=2000-09-25..2013-11-05 (3299) | gap_tr_va=10 (>=10: True) | Val=2013-11-19..2016-03-15 (583) | gap_va_te=10 (>=10: True) | Test=2016-03-30..2021-05-26 (1300) | overlaps=(0, 0, 0)
Fold 4: Train=2000-09-25..2018-03-28 (4404) | gap_tr_va=10 (>=10: True) | Val=2018-04-12..2021-05-13 (778) | gap_va_te=10 (>=10: True) | Test=2021-05-27..2026-07-31 (1300) | overlaps=(0, 0, 0)
```

### Detailed Fold 1 Partition Verification
1. **Training Set Date Range:** `2000-09-25` to `2005-01-26` (`1,089` trading sessions, `eval_idx` `0..1088`).
2. **Train → Val Purge + Embargo Buffer:** `9` excluded trading sessions (`2005-01-27`, `2005-01-28`, `2005-01-31`, `2005-02-01`, `2005-02-02`, `2005-02-03`, `2005-02-04`, `2005-02-07`, `2005-02-08`; `eval_idx` `1089..1097`).
   - `min(val_idx) - max(train_idx) = 1098 - 1088 = 10 == horizon (5) + embargo (5)` (**PASS**).
3. **Validation Set Date Range:** `2005-02-09` to `2005-11-11` (`193` trading sessions, `eval_idx` `1098..1290`).
4. **Val → Test Purge + Embargo Buffer:** `9` excluded trading sessions (`2005-11-14`, `2005-11-15`, `2005-11-16`, `2005-11-17`, `2005-11-18`, `2005-11-21`, `2005-11-22`, `2005-11-23`, `2005-11-25`; `eval_idx` `1291..1299`).
   - `min(test_idx) - max(val_idx) = 1300 - 1290 = 10 == horizon (5) + embargo (5)` (**PASS**).
5. **Test Set Date Range:** `2005-11-28` to `2011-01-26` (`1,300` trading sessions, `eval_idx` `1300..2599`).
6. **Split Disjointness:** Zero overlapping dates across `train`, `val`, and `test` in any fold; all purged/embargoed sessions are strictly excluded.

---

## 4.2 Three Critical Data-Loader Bugs Identified in `src/data/loader.py`

While `purged_walk_forward_cv` splits the index sequence cleanly, auditing [`src/data/loader.py`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/data/loader.py#L105-L218) against [`specs/features_baseline.md` (§1.1, §1.5, §7.1)](file:///Users/doxncwfn/Project/QuantAlphaMLOps/specs/features_baseline.md#L135-L560) uncovered three data bugs baked into `data/cache/silver_features_daily_store.npz`:

### Bug 4.2a — Truncated Blom Normal Scores Due to Wrong Denominator (`src/data/loader.py:107-119`)
In [`src/data/loader.py:107-119`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/data/loader.py#L107-L119):
```python
rank_exprs = [
    pl.col(c).rank("average").over("date").alias(f"{c}_r") for c in STOCK_FEATURE_COLS
]
pldf = pldf.with_columns(
    [pl.col("permno").count().over("date").alias("_n")] + rank_exprs
)

u_exprs = [
    pl.when(pl.col("_n") >= 100)
    .then((pl.col(f"{c}_r") - 0.375) / (pl.col("_n") + 0.25))
    .otherwise(None)
    .alias(f"{c}_u")
    for c in STOCK_FEATURE_COLS
]
```
- **Root Cause:** In Polars, `pl.col(c).rank("average").over("date")` ranks only the \(n_{\text{valid}}(c)\) non-null observations of feature `c` from \(1 \dots n_{\text{valid}}(c)\), **but** `loader.py` divides by `_n = pl.col("permno").count().over("date")` (the total universe count \(n_{\text{total}}\), including nulls) instead of `pl.col(c).count().over("date")` (as required by [`specs/features_baseline.md:546-553`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/specs/features_baseline.md#L546-L553): `valid = x.dropna(); n = len(valid)`).
- **Impact:** Whenever any stock has a null value for feature `c` on date \(t\), the maximum quantile is capped at \(\frac{n_{\text{valid}}(c) - 0.375}{n_{\text{total}} + 0.25} < 1\), **truncating the entire positive tail of \(\Phi^{-1}(u)\) (`sp.ndtri`) and shifting cross-sectional feature distributions negative**:

| Feature (`silver_features_daily_store.npz`) | Mean of Daily Means | Mean of Daily Mins | Mean of Daily Maxs | Expected Symmetric Range |
|---|---:|---:|---:|---:|
| `r_cc` (0 nulls) | `+0.0000` | `-3.2245` | `+3.2245` | `[-3.22, +3.22]` |
| `resid_mom_20` (nulls present) | **`-0.0312`** | **`-3.2246`** | **`+2.3221`** | `[-3.22, +3.22]` |
| `resid_vol_20` (nulls present) | **`-0.0298`** | **`-3.2246`** | **`+2.3453`** | `[-3.22, +3.22]` |
| `resid_ret` (nulls present) | **`-0.0279`** | **`-3.2246`** | **`+2.3770`** | `[-3.22, +3.22]` |
| `drawdown_20` (nulls present) | **`-0.0261`** | **`-3.2246`** | **`+1.5253`** | `[-3.22, +3.22]` |
| `rel_vol` / `vol_shock_5` | **`-0.0096`** | **`-3.2246`** | **`+2.6583`** | `[-3.22, +3.22]` |
| `price_z_20` | **`-0.0086`** | **`-3.2246`** | **`+2.6919`** | `[-3.22, +3.22]` |

### Bug 4.2b — Context Branch Zero-Filling (`2000-09-25` to `2000-12-18`, `src/data/loader.py:132-151`)
- [`src/data/loader.py:90-100,132-151`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/data/loader.py#L90-L151) queries context features only from `silver.features_daily` (which begins on `2000-06-30`), computes rolling 252-day z-scores with `min_periods=60`, and zero-fills the first 60 dates (`2000-06-30` to `2000-09-25`) via `.otherwise(0.0).fill_nan(0.0).fill_null(0.0)`.
- Because `eval_date_indices` starts at `d_i = 59` (`2000-09-25`) with a 60-session lookback `[d_i - 59, d_i]`, the first **60 training dates (`2000-09-25` to `2000-12-18`)** contain zero-filled context rows (violating [`specs/features_baseline.md` §1.1 & §1.5](file:///Users/doxncwfn/Project/QuantAlphaMLOps/specs/features_baseline.md#L135-L276): "Zero-filling a missing context feature injects a synthetic 'at the 1-year average' signal").

### Bug 4.2c — Missing 332-Session Warmup Cutoff & Stock Window Zero-Filling (`src/data/loader.py:126,168,200`)
- [`specs/features_baseline.md` §1.1](file:///Users/doxncwfn/Project/QuantAlphaMLOps/specs/features_baseline.md#L156-L163) requires a 332-session warmup (`252 + 20 + 60 = 332` trading sessions from `2000-06-30`, i.e., starting evaluation around `2001-10`).
- Instead, `loader.py` starts `eval_date_indices` at `2000-09-25` (`d_i = 59`), checks `features_ready` only at decision date `d_i` (rather than across all 60 lookback steps `[d_i - 59, d_i]`), and fills missing feature values and unready historical steps with `0.0` (`2.47%` of valid stock windows in Fold 1 contain all-zero historical steps).
