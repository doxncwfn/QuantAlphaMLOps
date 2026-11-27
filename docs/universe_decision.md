# Formal Universe Decision — Baseline 1A on Full `silver.russell` Universe

**Date:** 2026-11-06  
**Decision:** Standardize model baseline training on **`silver.features_daily`** (full Point-in-Time Russell 1000 index constituents, ~1,000 stocks/year) rather than the strict share-code subset `silver."features_daily_v1.3_strict"`.

---

## 1. Background & Context

During the feature table reconstruction, two potential Point-in-Time universe definitions were analyzed:

1. **Full `silver.russell` (`silver.features_daily`):** Uses the FTSE Russell 1000 published constituent snapshot for each reconstitution cycle (`Year = Y`). Universe size is $1{,}000 \pm 5\%$ across all years 2000–2026 (min 969 in 2009, max 1,036 in 2015).
2. **Strict US Common Equity Filter (`silver."features_daily_v1.3_strict"`):** Applies CRSP `shrcd IN (10, 11)` (US-incorporated common equity) on top of the Russell 1000 constituent list.

---

## 2. Rationale for Selecting `silver.features_daily`

1. **Constituent Realism & Index Parity:**  
   The Russell 1000 index explicitly includes real estate investment trusts (REITs, `shrcd = 18, 48`) and US-listed, US-headquartered or US-traded multinational corporations incorporated in Ireland, the UK, or Switzerland (`shrcd = 12`). Excluding them drops major industry leaders:
   - **REITs:** Prologis (`PLD`), American Tower (`AMT`), Crown Castle (`CCI`), Simon Property Group (`SPG`), Realty Income (`O`), Welltower (`WELL`).
   - **Foreign-Incorporated US Equities:** Accenture (`ACN`), Medtronic (`MDT`), Trane Technologies (`TT`), Linde (`LIN`).
   These constituents account for $>10\%$ of the active Russell 1000 market capitalization and are actively held by benchmark-tracking institutional portfolios.

2. **Universe Size Target ($1{,}000 \pm 5\%$):**  
   Filtering `shrcd IN (10, 11)` removes 100–172 stocks per date in 2010–2024, shrinking the universe to 835–948 stocks per date, which violates the $\sim 1{,}000 \pm 5\%$ requirement. Full `silver.russell` maintains 970–1,036 constituents per year.

3. **Coverage & Data Quality:**  
   `silver.features_daily` exhibits a 98.05% `features_ready` rate across 2000–2026 (98.00% CRSP, 98.95% Yahoo) across 6,498,250 in-universe stock-date observations.

---

## 3. Preservation for Ablation

The strict universe table **`silver."features_daily_v1.3_strict"`** (5,847,625 in-universe rows) is permanently preserved in DuckDB warehouse `data/warehouse/russell1000.duckdb` and Parquet partition `data/features_strict/features_YYYY.parquet`. It serves as a pre-computed baseline for future universe ablation experiments comparing strict US common stock performance against full Russell 1000 performance.
