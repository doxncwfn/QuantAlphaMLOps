# Spot Check Report — Manual vs. Pipeline Feature Verification (v1.3 §3.8)

**Date:** 2026-11-06  
**Scope:** Independent manual calculation from raw `bronze.crsp_daily` and `silver.market_factor` records vs. stored values in `silver.features_daily` (`data/features/features_YYYY.parquet`) for three benchmark constituent-date pairs:
1. **AAPL (`PERMNO = 14593`) on `2015-06-15`**
2. **MSFT (`PERMNO = 10107`) on `2010-06-15`**
3. **IBM (`PERMNO = 12490`) on `2005-06-15`**

**Acceptance Criterion:** Absolute difference $< 10^{-6}$ across all 19 stock-level features and `target_5d`.  
**Result:** **PASS** — Maximum absolute discrepancy across all $3 \times 20 = 60$ spot checks is **$7.41 \times 10^{-14}$** (floating-point machine epsilon).

---

## 1. AAPL (`PERMNO = 14593`) on `2015-06-15`

- **Raw CRSP Input (`2015-06-15`):** `PRC = 126.92`, `OPENPRC = 126.10`, `ASKHI = 127.24`, `BIDLO = 125.71`, `VOL = 43,988,900`, `RETX = -0.001966`, `RET = -0.001966`, `CFACPR = 4.0`.
- **Universe & Readiness:** `in_universe = True`, `features_ready = True`, `era = 'CRSP'`.

| # | Feature | Manual Calculation | Pipeline (`silver.features_daily`) | Absolute Difference | Status |
|:-:|---|---:|---:|---:|:---:|
| 1 | `r_cc` | `-0.00196794` | `-0.00196794` | `0.00e+00` | PASS |
| 2 | `r_on` | `-0.00844966` | `-0.00844966` | `0.00e+00` | PASS |
| 3 | `r_id` | `0.00648172` | `0.00648172` | `0.00e+00` | PASS |
| 4 | `range` | `0.01205484` | `0.01205484` | `0.00e+00` | PASS |
| 5 | `close_loc` | `0.79084967` | `0.79084967` | `0.00e+00` | PASS |
| 6 | `rel_vol` | `0.11058290` | `0.11058290` | `0.00e+00` | PASS |
| 7 | `ret_5d` | `-0.00690947` | `-0.00690947` | `8.93e-17` | PASS |
| 8 | `ret_20d` | `-0.01447230` | `-0.01447230` | `2.78e-17` | PASS |
| 9 | `ret_60d` | `-0.00452269` | `-0.00452269` | `7.03e-17` | PASS |
| 10 | `vol_20d` | `0.00937347` | `0.00937347` | `2.05e-16` | PASS |
| 11 | `vol_60d` | `0.01247612` | `0.01247612` | `4.34e-16` | PASS |
| 12 | `vol_ratio` | `0.75131318` | `0.75131318` | `4.25e-14` | PASS |
| 13 | `price_z_20` | `-0.50944877` | `-0.50944877` | `1.10e-14` | PASS |
| 14 | `drawdown_20` | `-0.04332744` | `-0.04332744` | `0.00e+00` | PASS |
| 15 | `eff_ratio_20` | `0.10307013` | `0.10307013` | `1.80e-16` | PASS |
| 16 | `vol_shock_5` | `0.04607835` | `0.04607835` | `4.86e-17` | PASS |
| 17 | `resid_ret` | `-0.00039418` | `-0.00039418` | `1.49e-15` | PASS |
| 18 | `resid_mom_20` | `-0.02606177` | `-0.02606177` | `8.10e-15` | PASS |
| 19 | `resid_vol_20` | `0.00484361` | `0.00484361` | `7.01e-16` | PASS |
| — | `target_5d` | `0.00542250` | `0.00542250` | `0.00e+00` | PASS |

---

## 2. MSFT (`PERMNO = 10107`) on `2010-06-15`

- **Raw CRSP Input (`2010-06-15`):** `PRC = 26.58`, `OPENPRC = 25.7475`, `ASKHI = 26.65`, `BIDLO = 25.74`, `VOL = 66,469,200`, `RETX = 0.042655`, `RET = 0.042655`, `CFACPR = 1.0`.
- **Universe & Readiness:** `in_universe = True`, `features_ready = True`, `era = 'CRSP'`.

| # | Feature | Manual Calculation | Pipeline (`silver.features_daily`) | Absolute Difference | Status |
|:-:|---|---:|---:|---:|:---:|
| 1 | `r_cc` | `0.04177034` | `0.04177034` | `0.00e+00` | PASS |
| 2 | `r_on` | `0.00995187` | `0.00995187` | `0.00e+00` | PASS |
| 3 | `r_id` | `0.03181848` | `0.03181848` | `0.00e+00` | PASS |
| 4 | `range` | `0.03423305` | `0.03423305` | `0.00e+00` | PASS |
| 5 | `close_loc` | `0.92582418` | `0.92582418` | `0.00e+00` | PASS |
| 6 | `rel_vol` | `0.02429768` | `0.02429768` | `0.00e+00` | PASS |
| 7 | `ret_5d` | `0.05698654` | `0.05698654` | `1.04e-16` | PASS |
| 8 | `ret_20d` | `-0.08497112` | `-0.08497112` | `1.39e-17` | PASS |
| 9 | `ret_60d` | `-0.10718344` | `-0.10718344` | `2.78e-17` | PASS |
| 10 | `vol_20d` | `0.02407774` | `0.02407774` | `8.67e-17` | PASS |
| 11 | `vol_60d` | `0.01701438` | `0.01701438` | `7.29e-17` | PASS |
| 12 | `vol_ratio` | `1.41514131` | `1.41514131` | `1.13e-14` | PASS |
| 13 | `price_z_20` | `0.16134612` | `0.16134612` | `5.83e-16` | PASS |
| 14 | `drawdown_20` | `-0.07306620` | `-0.07306620` | `0.00e+00` | PASS |
| 15 | `eff_ratio_20` | `0.21475537` | `0.21475537` | `8.33e-17` | PASS |
| 16 | `vol_shock_5` | `-0.07121834` | `-0.07121834` | `4.44e-16` | PASS |
| 17 | `resid_ret` | `0.01944433` | `0.01944433` | `3.87e-15` | PASS |
| 18 | `resid_mom_20` | `-0.09222537` | `-0.09222537` | `7.41e-14` | PASS |
| 19 | `resid_vol_20` | `0.01443233` | `0.01443233` | `3.92e-16` | PASS |
| — | `target_5d` | `-0.03104146` | `-0.03104146` | `0.00e+00` | PASS |

---

## 3. IBM (`PERMNO = 12490`) on `2005-06-15`

- **Raw CRSP Input (`2005-06-15`):** `PRC = 76.29`, `OPENPRC = 75.69`, `ASKHI = 76.49`, `BIDLO = 75.14`, `VOL = 8,284,400`, `RETX = 0.018828`, `RET = 0.018828`, `CFACPR = 1.04579186`.
- **Universe & Readiness:** `in_universe = True`, `features_ready = True`, `era = 'CRSP'`.

| # | Feature | Manual Calculation | Pipeline (`silver.features_daily`) | Absolute Difference | Status |
|:-:|---|---:|---:|---:|:---:|
| 1 | `r_cc` | `0.01865295` | `0.01865295` | `0.00e+00` | PASS |
| 2 | `r_on` | `0.01075817` | `0.01075817` | `0.00e+00` | PASS |
| 3 | `r_id` | `0.00789478` | `0.00789478` | `0.00e+00` | PASS |
| 4 | `range` | `0.01769332` | `0.01769332` | `0.00e+00` | PASS |
| 5 | `close_loc` | `0.85185185` | `0.85185185` | `0.00e+00` | PASS |
| 6 | `rel_vol` | `0.29090266` | `0.29090266` | `0.00e+00` | PASS |
| 7 | `ret_5d` | `0.01985589` | `0.01985589` | `2.08e-17` | PASS |
| 8 | `ret_20d` | `0.02669704` | `0.02669704` | `4.86e-17` | PASS |
| 9 | `ret_60d` | `-0.15967850` | `-0.15967850` | `2.78e-17` | PASS |
| 10 | `vol_20d` | `0.01233407` | `0.01233407` | `5.03e-17` | PASS |
| 11 | `vol_60d` | `0.01682139` | `0.01682139` | `3.12e-17` | PASS |
| 12 | `vol_ratio` | `0.73323742` | `0.73323742` | `1.67e-15` | PASS |
| 13 | `price_z_20` | `0.08694905` | `0.08694905` | `1.65e-14` | PASS |
| 14 | `drawdown_20` | `-0.01366764` | `-0.01366764` | `0.00e+00` | PASS |
| 15 | `eff_ratio_20` | `0.14594499` | `0.14594499` | `2.50e-16` | PASS |
| 16 | `vol_shock_5` | `-0.06658768` | `-0.06658768` | `8.33e-17` | PASS |
| 17 | `resid_ret` | `0.01598094` | `0.01598094` | `1.56e-16` | PASS |
| 18 | `resid_mom_20` | `0.01411171` | `0.01411171` | `6.35e-16` | PASS |
| 19 | `resid_vol_20` | `0.00881901` | `0.00881901` | `1.25e-16` | PASS |
| — | `target_5d` | `0.01211606` | `0.01211606` | `0.00e+00` | PASS |

---

## 4. Algebraic Identity & Lagged-Beta Verification Summary

1. **Intraday-Overnight Return Decomposition:**
   - `AAPL 2015-06-15`: `r_on (-0.00844966) + r_id (0.00648172) = -0.00196794 = r_cc` (exact to $0.00\times 10^0$).
   - `MSFT 2010-06-15`: `r_on (0.00995187) + r_id (0.03181848) = 0.04177034 = r_cc` (exact to $0.00\times 10^0$).
   - `IBM 2005-06-15`: `r_on (0.01075817) + r_id (0.00789478) = 0.01865295 = r_cc` (exact to $0.00\times 10^0$).
2. **Geometric SMA in `price_z_20`:**
   - Verified manually using $\log C_s - \frac{1}{20}\sum_{j=0}^{19}\log C_{s-j}$ divided by $\sigma_{20,s}\sqrt{20}$ (`ddof=1`); matches pipeline to $\le 1.65 \times 10^{-14}$.
3. **Strictly Lagged Beta (`[s-252, s-1]`) and Frozen Historical Residuals:**
   - Each of the 20 historical residuals $\epsilon_{s-19}, \ldots, \epsilon_s$ summed in `resid_mom_20` and taken std in `resid_vol_20` was independently estimated using its own trailing 252-session window $[\tau-252, \tau-1]$ via `np.linalg.lstsq`. Agreement is $\le 7.41 \times 10^{-14}$.
