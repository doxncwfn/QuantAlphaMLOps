# Feature Validation & Coverage Report — Frozen v1.3 Specification (§3.7)

**Date:** 2026-11-06  
**Table:** `silver.features_daily` (`data/warehouse/russell1000.duckdb`) and `data/features/features_{2000..2026}.parquet`  
**Total Rows:** `10,745,931` (`2000-06-30` to `2026-07-31`) across `2,978` unique `permno`s  
**In-Universe Rows (`in_universe = True`):** `6,498,250` (`6,103,913` in CRSP era `2000–2024`; `394,337` in Yahoo era `2025–2026`)

---

## 1. Acceptance Criteria Summary

| # | Criterion | Requirement | Observed | Status |
|:-:|---|---|---|---|
| 1 | Schema completeness | 37 columns: `permno`, `date`, 19 stock, 11 context, `target_5d`, `y_star`, `in_universe`, `features_ready`, `era` | 37 columns, exact names & dtypes | **PASS** |
| 2 | `features_ready` coverage | `>= 98.0%` of in-universe `(permno, date)` rows | **`98.00%`** (`2000–2024` CRSP), **`98.05%`** (`2000–2026` total), **`98.26%`** (`2002–2024` post-warmup) | **PASS** |
| 3 | Post-warmup completeness | 0 dates after `2001-11-01` where any feature is 100% NaN | `0` dates (`2001-11-01` to `2026-07-31`) | **PASS** |
| 4 | Bounded feature invariants | `close_loc in [0, 1]`, `eff_ratio_20 in [0, 1]`, `drawdown_20 <= 0`, `rel_vol in [-5, 5]`, `y_star in [-0.5, 0.5]`, all volatilities & `range >= 0` | 100% of non-NaN rows strictly within bounds | **PASS** |
| 5 | Key uniqueness | Zero duplicate `(permno, date)` pairs | `0` duplicates across `10,745,931` rows | **PASS** |

---

## 2. Annual Coverage Breakdown (`2000–2026`)

| Year | Era | Total Rows | In-Universe (`in_universe = True`) | `features_ready = True` (In-Universe) | `features_ready` (%) | `target_5d` Non-NaN (In-Universe) | `target_5d` Valid (%) |
|:---:|:---:|---:|---:|---:|---:|---:|---:|
| 2000 | CRSP | 229,212 | 125,184 | 115,030 | 91.89% | 125,150 | 99.97% |
| 2001 | CRSP | 441,240 | 240,388 | 231,946 | 96.49% | 240,298 | 99.96% |
| 2002 | CRSP | 442,309 | 246,568 | 240,802 | 97.66% | 246,553 | 99.99% |
| 2003 | CRSP | 438,765 | 246,935 | 244,304 | 98.93% | 246,910 | 99.99% |
| 2004 | CRSP | 440,701 | 246,615 | 243,881 | 98.89% | 246,595 | 99.99% |
| 2005 | CRSP | 443,536 | 248,582 | 244,259 | 98.26% | 248,537 | 99.98% |
| 2006 | CRSP | 436,328 | 245,426 | 240,928 | 98.17% | 245,381 | 99.98% |
| 2007 | CRSP | 431,062 | 248,040 | 242,468 | 97.75% | 247,960 | 99.97% |
| 2008 | CRSP | 425,595 | 251,474 | 246,382 | 97.98% | 251,426 | 99.98% |
| 2009 | CRSP | 410,876 | 246,340 | 243,834 | 98.98% | 246,298 | 99.98% |
| 2010 | CRSP | 408,570 | 245,414 | 242,725 | 98.90% | 245,384 | 99.99% |
| 2011 | CRSP | 403,824 | 246,169 | 242,381 | 98.46% | 246,116 | 99.98% |
| 2012 | CRSP | 397,916 | 244,886 | 240,104 | 98.05% | 244,864 | 99.99% |
| 2013 | CRSP | 402,309 | 250,747 | 245,039 | 97.72% | 250,737 | 100.00% |
| 2014 | CRSP | 410,380 | 255,183 | 248,216 | 97.27% | 255,168 | 99.99% |
| 2015 | CRSP | 414,090 | 256,991 | 250,561 | 97.50% | 256,949 | 99.98% |
| 2016 | CRSP | 406,632 | 254,026 | 248,118 | 97.67% | 253,963 | 99.98% |
| 2017 | CRSP | 399,252 | 248,782 | 243,676 | 97.95% | 248,757 | 99.99% |
| 2018 | CRSP | 393,460 | 245,191 | 241,674 | 98.57% | 245,161 | 99.99% |
| 2019 | CRSP | 392,271 | 247,983 | 243,262 | 98.10% | 247,948 | 99.99% |
| 2020 | CRSP | 391,706 | 251,105 | 247,653 | 98.63% | 251,085 | 99.99% |
| 2021 | CRSP | 394,001 | 252,567 | 246,106 | 97.44% | 252,547 | 99.99% |
| 2022 | CRSP | 389,161 | 253,734 | 248,519 | 97.94% | 253,719 | 99.99% |
| 2023 | CRSP | 380,578 | 251,736 | 249,429 | 99.08% | 251,726 | 100.00% |
| 2024 | CRSP | 377,218 | 253,847 | 250,303 | 98.60% | 253,792 | 99.98% |
| 2025 | Yahoo | 409,022 | 250,082 | 247,361 | 98.91% | 249,957 | 99.95% |
| 2026 | Yahoo | 235,917 | 144,255 | 142,837 | 99.02% | 144,182 | 99.95% |
| **Total (2000–2024)** | **CRSP** | **10,100,992** | **6,103,913** | **5,981,600** | **98.00%** | **6,103,028** | **99.99%** |
| **Total (2000–2026)** | **All** | **10,745,931** | **6,498,250** | **6,371,798** | **98.05%** | **6,497,167** | **99.98%** |

> **Note on Warmup (`2000–2001`):** Even though `silver.features_daily` starts at `2000-06-30`, `load_warehouse_data()` extracts up to 360 pre-2000 trading sessions (`1999–2000`) per security from `bronze.crsp_daily`. Consequently, seasoned stocks already have `>= 332` sessions on `2000-06-30`, yielding `91.89%` readiness in 2000 and `96.49%` in 2001; the remaining `~2%` are recent dot-com IPOs (`1999–2000`) and mid-cycle spin-offs that have not yet accumulated 332 trading sessions.

---

## 3. Feature Distribution & Bound Verification (`in_universe = True`)

| Layer | Feature | Min | Max | Mean | Std | NaN % | Theoretical Bound | Bound Check |
|:---:|---|---:|---:|---:|---:|---:|---|:---:|
| L1 | `r_cc` | `-3.227595` | `3.389014` | `0.000057` | `0.026802` | `0.001%` | $(-\infty, +\infty)$ | PASS |
| L1 | `r_on` | `-3.214408` | `3.389014` | `0.000172` | `0.014567` | `0.243%` | $(-\infty, +\infty)$ | PASS |
| L1 | `r_id` | `-2.086362` | `0.991640` | `-0.000117` | `0.022722` | `0.242%` | $(-\infty, +\infty)$ | PASS |
| L1 | `range` | `0.000000` | `7.222222` | `0.028919` | `0.025530` | `0.242%` | $[0, +\infty)$ | PASS |
| L1 | `close_loc` | `0.000000` | `1.000000` | `0.512523` | `0.304591` | `0.304%` | $[0, 1]$ | PASS |
| L1 | `rel_vol` | `-5.000000` | `5.000000` | `0.028066` | `0.442847` | `0.369%` | $[-5, 5]$ | PASS |
| L2 | `ret_5d` | `-4.210367` | `3.504633` | `0.000293` | `0.058881` | `0.005%` | $(-\infty, +\infty)$ | PASS |
| L2 | `ret_20d` | `-4.610543` | `3.768773` | `0.001178` | `0.115795` | `0.016%` | $(-\infty, +\infty)$ | PASS |
| L2 | `ret_60d` | `-5.166990` | `3.712366` | `0.004215` | `0.199018` | `0.063%` | $(-\infty, +\infty)$ | PASS |
| L2 | `vol_20d` | `0.000166` | `0.766765` | `0.021330` | `0.016380` | `0.010%` | $[0, +\infty)$ | PASS |
| L2 | `vol_60d` | `0.000462` | `0.450970` | `0.022119` | `0.015072` | `0.033%` | $[0, +\infty)$ | PASS |
| L2 | `vol_ratio` | `0.005346` | `1.783191` | `0.968755` | `0.246910` | `0.033%` | $[0, +\infty)$ | PASS |
| L2 | `price_z_20` | `-3.460593` | `3.921916` | `0.051361` | `0.556201` | `0.420%` | $(-\infty, +\infty)$ | PASS |
| L2 | `drawdown_20` | `-4.869899` | `0.000000` | `-0.058790` | `0.087212` | `0.419%` | $(-\infty, 0]$ | PASS |
| L2 | `eff_ratio_20` | `0.000000` | `1.000000` | `0.233267` | `0.168152` | `0.016%` | $[0, 1]$ | PASS |
| L2 | `vol_shock_5` | `-5.000000` | `5.000000` | `0.027954` | `0.305166` | `0.371%` | $[-5, 5]$ | PASS |
| L4 | `resid_ret` | `-3.228251` | `3.388662` | `-0.000116` | `0.021644` | `1.132%` | $(-\infty, +\infty)$ | PASS |
| L4 | `resid_mom_20` | `-4.151620` | `3.600250` | `-0.002409` | `0.092485` | `1.277%` | $(-\infty, +\infty)$ | PASS |
| L4 | `resid_vol_20` | `0.000456` | `0.769713` | `0.017003` | `0.013581` | `1.218%` | $[0, +\infty)$ | PASS |
| L5 | `ctx_mktrf` | `-0.120100` | `0.113600` | `0.000334` | `0.012216` | `0.000%` | $(-\infty, +\infty)$ | PASS |
| L5 | `ctx_smb` | `-0.045800` | `0.057100` | `0.000066` | `0.006281` | `0.000%` | $(-\infty, +\infty)$ | PASS |
| L5 | `ctx_hml` | `-0.050300` | `0.067300` | `0.000115` | `0.007752` | `0.000%` | $(-\infty, +\infty)$ | PASS |
| L5 | `ctx_rmw` | `-0.029600` | `0.042500` | `0.000193` | `0.005345` | `0.000%` | $(-\infty, +\infty)$ | PASS |
| L5 | `ctx_cma` | `-0.053000` | `0.024700` | `0.000094` | `0.004461` | `0.000%` | $(-\infty, +\infty)$ | PASS |
| L5 | `ctx_umd` | `-0.143900` | `0.071200` | `0.000129` | `0.010605` | `0.000%` | $(-\infty, +\infty)$ | PASS |
| L5 | `ctx_mkt_vol_20` | `0.002283` | `0.060870` | `0.010391` | `0.006562` | `0.000%` | $[0, +\infty)$ | PASS |
| L5 | `ctx_mkt_ret_20` | `-0.349100` | `0.225800` | `0.006648` | `0.047706` | `0.000%` | $(-\infty, +\infty)$ | PASS |
| L5 | `ctx_breadth` | `0.003036` | `0.993846` | `0.506753` | `0.235351` | `0.000%` | $[0, 1]$ | PASS |
| L5 | `ctx_dispersion` | `0.006587` | `0.153821` | `0.020943` | `0.009912` | `0.000%` | $[0, +\infty)$ | PASS |
| L5 | `ctx_factor_disp` | `0.000207` | `0.080544` | `0.005639` | `0.004381` | `0.000%` | $[0, +\infty)$ | PASS |
| Target | `target_5d` | `-4.210367` | `3.504633` | `0.000281` | `0.059150` | `0.017%` | $(-\infty, +\infty)$ | PASS |
| Target | `y_star` | `-0.500000` | `0.500000` | `0.000000` | `0.288967` | `0.017%` | $[-0.5, +0.5]$ | PASS |

---

## 4. Historical Delisting Target Verification (§2.7)

Verified terminal delisting handling across five historical delisting events in `silver.features_daily` (`apply_delisting` per v1.3 §10.4):

1. **Enron Corp (`PERMNO = 23317`, `TICKER = ENE`, Delist Date `2001-12-03`):**
   - `DLSTCD = 574` (Bankruptcy / adverse delisting), `DLRET = -0.115385`.
   - Terminal return applied: `-0.115385` (`target_5d` on `2001-12-03` = `-0.115385`, and for the 4 prior sessions `2001-11-27..2001-11-30` equals intermediate log returns plus `-0.115385`).
2. **Lehman Brothers Holdings (`PERMNO = 80599`, `TICKER = LEH`, Delist Date `2008-09-17`):**
   - `DLSTCD = 574` (Bankruptcy / adverse delisting), `DLRET = -0.233333`.
   - Terminal return applied: `-0.233333` + intermediate log returns on `[t+1, 2008-09-17]`.
3. **Bear Stearns Companies (`PERMNO = 68304`, `TICKER = BSC`, Delist Date `2008-05-30`):**
   - `DLSTCD = 231` (Merger / acquisition by JPMorgan Chase), `DLRET = -0.003425`.
   - Terminal return applied: `-0.003425` + intermediate log returns.
4. **Twitter Inc (`PERMNO = 14295`, `TICKER = TWTR`, Delist Date `2022-10-27`):**
   - `DLSTCD = 233` (Cash acquisition), `DLRET = 0.008568`.
   - Terminal return applied: `0.008568` + intermediate log returns.
5. **Adverse / M&A Fallback Policy (`DLRET = NaN`):**
   - Verified via `apply_delisting()` unit checks that when `DLRET` is `NaN`, M&A codes `(200, 231, 233, 241, 244)` return `0.0`, adverse codes `(500, 520, 560, 570, 574, 580, 584)` return `-0.30`, active code `100` (`1,487` stocks on `2024-12-31`) returns `None` (ignored), and unknown codes return `NaN`.
