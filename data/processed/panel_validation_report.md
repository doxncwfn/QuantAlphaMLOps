# Continuous Per-Stock Panel Validation Report

## Executive Summary
This report validates the single continuous per-stock panel (`data/processed/continuous_panel.parquet`) and point-in-time universe mask (`data/processed/universe_mask.parquet`), constructed to eliminate survivorship bias and reconstitution-year truncation in feature computation for the Russell 1000 cross-sectional ranking model.

Pre-index histories for all Russell 1000 constituents have been systematically backfilled using `data/US_history.parquet`, ensuring full historical depth prior to index entry.

---

## 1. Coverage Report (§6.1)
- **Total Unique Securities (Entities):** 2,946
  - Unique PERMNOs: 2,836
  - Unique Tickers: 3,989
- **Total Security-Date Rows:** 18,896,408
- **Date Range:** 1925-12-31 to 2026-09-25
- **Stocks with Pre-2000-06-30 History:** 1,810 stocks (8,520,509 trading sessions backfilled)
- **History Length Distribution (Trading Sessions per Stock):**
  - Minimum: 46
  - 25th Percentile: 2262
  - Median: 5034
  - Mean: 6414.3
  - 75th Percentile: 8679
  - Maximum: 26,485

---

## 2. Deduplication and Overlap Audit (§6.2)
- **Overlapping Rows across Annual WRDS Files:** 15,451 `(PERMNO, date)` rows.
  - Number of numeric field discrepancies: **0** across all 15,451 rows.
  - Number of ticker string discrepancies: **36** (due to share-class notation updates such as `IDT` vs `IDT.C`).
- **Internal Multi-Distribution Duplicates in US_history.parquet:** 5,626 duplicate `(PERMNO, date)` rows.
  - Root cause: Multiple cash/stock distributions declared on the same trading day (e.g., `DISTCD 1232` and `1272`).
  - Pricing, volume, and return values were 100% identical.
  - Deduplicated by retaining a single record per `(PERMNO, date)`.
- **Annual WRDS vs US_history Consistency on Matched Rows (6,337,877 rows):**
  - All price, volume, adjustment factor, and delisting fields (`PRC`, `OPENPRC`, `ASKHI`, `BIDLO`, `VOL`, `CFACPR`, `CFACSHR`, `SHROUT`, `DLSTCD`): **0 differences (100% identical)**.
  - Returns (`RET`, `RETX`, `DLRET`): 100% numerical equality (differences restricted to string float trailing zeroes like `0.01408` vs `0.014080`).
  - Resolution: WRDS annual records retained for shared dates to preserve Russell share-class ticker formatting; `US_history` utilized for all backfill rows.
- **Final Panel Duplicate Count on `(stock, date)`:** **0** (strictly 1 row per stock-date).

---

## 3. Backfill Distribution (§4.4)
- **Stocks Gaining Pre-Index / Out-of-Index History:** 2,799 stocks.
- **Total Backfilled Sessions:** 12,126,609 sessions.
- **Sessions Gained per Stock:**
  - Mean: 4332.5 sessions
  - Median: 2747 sessions
  - Max: 24,795 sessions
- **Impact on 332-Session Feature Warmup:**
  - In the unstitched annual pipeline, 20–30% of each year's constituents were unusable on July 1.
  - In the continuous panel, newly added Russell 1000 stocks (e.g. PERMNO 15490 / LITE in July 2020) immediately possess their full preceding trading sessions.

---

## 4. Identifier Mapping Report (§6.3)
- **Primary Identifier Convention:** `PERMNO` (int64) for CRSP era (2000-2024); `Ticker` (string) with mapped `PERMNO` for Yahoo era (2025-2026).
- **CRSP WRDS PERMNO Coverage in US_history:** 2,834 / 2,834 (100% match, 0 missing).
- **PERMNOs Experiencing Ticker Changes:** 330 PERMNOs.
- **Yahoo Ticker Mapping to Historical PERMNOs:**
  - 1,119 of 1,136 Yahoo tickers (98.5%) mapped to authoritative CRSP PERMNOs.
  - 17 modern listings (post-2024 IPOs/spin-offs) retained with `permno = null` and uniquely identified by `ticker`.

---

## 5. Cross-Era Boundary Consistency (§6.4)
- **Boundary Date:** 2024-12-31 (CRSP) → 2025-01-02 (Yahoo).
- **Common Bridging Tickers:** 998 securities.
- **Median Ratio of First Yahoo Close to Last CRSP Close:** `0.9962` (reflects normal market movement of -0.38%).
- **Jumps > 20%:** 27 securities, strictly caused by corporate stock splits retroactively applied in Yahoo's split-adjusted prices (e.g. FAST 2:1, COKE 10:1, NOW 5:1, LCID 1:10 reverse split).
- **Return Continuity:** Yahoo returns are computed via percent changes within the Yahoo era, avoiding synthetic boundary jumps.

---

## 6. Point-in-Time Universe Coverage (§6.5)
- All 27 reconstitution vintages (2000–2026) verified against `data/processed/russell1000_all_years.csv`.
- Point-in-time universe mask stored in `data/processed/universe_mask.parquet` with columns `(date, permno, in_universe)`.
- Mask is detached from panel to maintain clean feature computation across all available history.

---

## 7. Factor Alignment (§6.6)
- Factor dataset `data/ff.csv` covers 2000-01-03 to 2026-07-31 (6,684 trading days).
- For 2000-01-03 to 2024-12-31: 6,289 dates in continuous panel match `data/ff.csv` with **100% exact alignment (0 missing)**.
- For 2025-01-02 to 2026-07-31: All dates match `data/ff.csv` except 4 foreign holiday records on single ticker LNW.
- Dates post-2026-07-31 (43 days in August–September 2026): Model backtesting is truncated at 2026-07-31 per Risk Register R3.

---

## 8. Feature Readiness (§7.4)
- **Feature Readiness Column:** `features_ready` (boolean).
- **Definition:** True when cumulative trading sessions for an entity reaches $\ge 332$ sessions.
- **Readiness Rate:** 17,942,445 / 18,896,408 (94.95% of all stock-dates).
- Allows instant pre-filtering in training loops without recomputing lookback logic.
