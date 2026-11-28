# Feature Table Universe Comparison: `silver.features_daily` vs. `silver."features_daily_v1.3_strict"`

## Overview

This document compares the two v1.3 feature tables stored in `data/warehouse/russell1000.duckdb`:

1. **`silver.features_daily`** (`data/features/features_YYYY.parquet`): Full Point-in-Time Russell 1000 membership (`filter_shrcd=False`), retaining all Russell 1000 index constituents regardless of CRSP share code (`shrcd`), including REITs (`shrcd IN (18, 48)`), foreign-incorporated ordinary shares (`shrcd = 12`), closed-end funds (`shrcd IN (14, 44)`), ADRs (`shrcd = 31`), and post-2024 synthetic Yahoo tickers (`permno >= 900001`).
2. **`silver."features_daily_v1.3_strict"`** (`data/features_strict/features_YYYY.parquet`): Strict v1.3 §2.2 universe (`filter_shrcd=True`), restricting `in_universe = True` strictly to US common equities with `shrcd IN (10, 11)` (with each security's last known CRSP / `silver.security_history` `shrcd` forward-filled per `permno` into the 2025–2026 Yahoo era, and never admitting `shrcd IS NULL` or non-10/11 codes).

Both tables are built with `seed_everything(42)` and share the identical 37-column schema (`OUTPUT_COLUMNS`), identical total panel rows (`10,745,931`), zero `(permno, date)` duplicates, and bit-for-bit identical stock-level features (Layers 1, 2, and 4, plus `target_5d` and `features_ready`). Cross-sectional quantities that aggregate or rank over `in_universe = True` (`ctx_breadth`, `ctx_dispersion`, and `y_star`) are recomputed over the respective active universe.

---

## Overall Summary Comparison

| Metric | `silver.features_daily` | `silver."features_daily_v1.3_strict"` | Delta (`Strict - Base`) |
| :--- | ---: | ---: | ---: |
| **Total rows (2000-06-30 to 2026-07-31)** | 10,745,931 | 10,745,931 | 0 |
| **In-universe rows (`in_universe = True`)** | 6,498,250 | 5,847,625 | -650,625 (-10.01%) |
| **Total unique `permno`s (all rows)** | 2,977 | 2,977 | 0 |
| **Unique `permno`s (`in_universe = True`)** | 2,782 | 2,477 | -305 (-10.96%) |
| **`features_ready %` (in-universe, 2000–2026)** | 98.05% | 98.10% | +0.05 pp |
| **`features_ready %` (in-universe, CRSP 2000–2024)** | 98.00% | 98.02% | +0.02 pp |
| **Duplicate `(permno, date)` pairs** | 0 | 0 | 0 |
| **Column count & dtypes match** | 37 columns | 37 columns | Identical |

---

## Year-by-Year Comparison (2000–2026)

| Year | Total Rows | In-Universe Rows (Base) | In-Universe Rows (Strict) | Delta In-Universe Rows | Unique In-Universe `permno`s (Base) | Unique In-Universe `permno`s (Strict) | Delta Unique `permno`s | `features_ready %` (Base) | `features_ready %` (Strict) | Delta `features_ready %` (pp) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 229,212 | 125,184 | 121,755 | -3,429 | 1,013 | 986 | -27 | 91.89% | 91.76% | -0.13 |
| 2001 | 441,240 | 240,388 | 233,568 | -6,820 | 1,160 | 1,128 | -32 | 96.49% | 96.44% | -0.05 |
| 2002 | 442,309 | 246,568 | 237,650 | -8,918 | 1,144 | 1,101 | -43 | 97.66% | 97.63% | -0.03 |
| 2003 | 438,765 | 246,935 | 236,627 | -10,308 | 994 | 953 | -41 | 98.93% | 98.96% | +0.03 |
| 2004 | 440,701 | 246,615 | 234,828 | -11,787 | 1,122 | 1,066 | -56 | 98.89% | 98.84% | -0.05 |
| 2005 | 443,536 | 248,582 | 234,945 | -13,637 | 1,097 | 1,038 | -59 | 98.26% | 98.22% | -0.04 |
| 2006 | 436,328 | 245,426 | 232,116 | -13,310 | 1,091 | 1,029 | -62 | 98.17% | 98.06% | -0.11 |
| 2007 | 431,062 | 248,040 | 231,944 | -16,096 | 1,062 | 972 | -90 | 97.75% | 97.85% | +0.10 |
| 2008 | 425,595 | 251,474 | 230,076 | -21,398 | 1,068 | 976 | -92 | 97.98% | 98.04% | +0.06 |
| 2009 | 410,876 | 246,340 | 225,937 | -20,403 | 1,043 | 949 | -94 | 98.98% | 98.98% | 0.00 |
| 2010 | 408,570 | 245,414 | 223,204 | -22,210 | 1,030 | 930 | -100 | 98.90% | 98.96% | +0.06 |
| 2011 | 403,824 | 246,169 | 222,138 | -24,031 | 1,043 | 939 | -104 | 98.46% | 98.57% | +0.11 |
| 2012 | 397,916 | 244,886 | 218,721 | -26,165 | 1,043 | 929 | -114 | 98.05% | 98.28% | +0.23 |
| 2013 | 402,309 | 250,747 | 220,273 | -30,474 | 1,048 | 920 | -128 | 97.72% | 97.92% | +0.20 |
| 2014 | 410,380 | 255,183 | 222,250 | -32,933 | 1,068 | 928 | -140 | 97.27% | 97.49% | +0.22 |
| 2015 | 414,090 | 256,991 | 222,268 | -34,723 | 1,061 | 922 | -139 | 97.50% | 97.57% | +0.07 |
| 2016 | 406,632 | 254,026 | 218,253 | -35,773 | 1,101 | 948 | -153 | 97.67% | 97.81% | +0.14 |
| 2017 | 399,252 | 248,782 | 211,329 | -37,453 | 1,095 | 923 | -172 | 97.95% | 98.15% | +0.20 |
| 2018 | 393,460 | 245,191 | 208,153 | -37,038 | 1,042 | 888 | -154 | 98.57% | 98.79% | +0.22 |
| 2019 | 392,271 | 247,983 | 212,353 | -35,630 | 1,039 | 890 | -149 | 98.10% | 98.07% | -0.03 |
| 2020 | 391,706 | 251,105 | 215,608 | -35,497 | 1,009 | 867 | -142 | 98.63% | 98.43% | -0.20 |
| 2021 | 394,001 | 252,567 | 219,195 | -33,372 | 1,120 | 966 | -154 | 97.44% | 97.36% | -0.08 |
| 2022 | 389,161 | 253,734 | 223,450 | -30,284 | 1,076 | 948 | -128 | 97.94% | 97.91% | -0.03 |
| 2023 | 380,578 | 251,736 | 222,420 | -29,316 | 1,057 | 929 | -128 | 99.08% | 99.00% | -0.08 |
| 2024 | 377,218 | 253,847 | 224,328 | -29,519 | 1,053 | 929 | -124 | 98.60% | 98.69% | +0.09 |
| 2025 | 409,022 | 250,082 | 219,284 | -30,798 | 1,046 | 916 | -130 | 98.91% | 99.05% | +0.14 |
| 2026 | 235,917 | 144,255 | 124,952 | -19,303 | 1,073 | 911 | -162 | 99.02% | 99.58% | +0.56 |

---

## 3-Pair Spot Check: Stock-Level Feature Identity

For securities present in both universes (`in_universe = True` in both tables), stock-level features (`r_cc`, `ret_5d`, `resid_ret`) are bit-for-bit identical (`|diff| = 0.0`), while cross-sectional universe-dependent quantities (`ctx_breadth`, `y_star`) reflect the active constituent set on that date:

| `permno` | Ticker | `date` | `r_cc` (Base) | `r_cc` (Strict) | `|Δ r_cc|` | `ret_5d` (Base) | `ret_5d` (Strict) | `|Δ ret_5d|` | `resid_ret` (Base) | `resid_ret` (Strict) | `|Δ resid_ret|` | `ctx_breadth` (Base) | `ctx_breadth` (Strict) | `y_star` (Base) | `y_star` (Strict) |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14593 | AAPL | `2015-06-15` | -0.001968 | -0.001968 | 0.0 | -0.006909 | -0.006909 | 0.0 | -0.000394 | -0.000394 | 0.0 | 0.196819 | 0.203682 | -0.248259 | -0.252304 |
| 10107 | MSFT | `2020-11-16` | 0.003319 | 0.003319 | 0.0 | -0.005326 | -0.005326 | 0.0 | 0.003435 | 0.003435 | 0.0 | 0.827968 | 0.832553 | -0.346928 | -0.355803 |
| 86580 | NVDA | `2024-03-15` | -0.001223 | -0.001223 | 0.0 | 0.003519 | 0.003519 | 0.0 | 0.020741 | 0.020741 | 0.0 | 0.482143 | 0.481481 | 0.454229 | 0.449324 |

---

## 10 Prominent Stocks Present in `silver.features_daily` (`in_universe = True`) but Excluded from `silver."features_daily_v1.3_strict"`

| # | `permno` | `ticker` | `comnam` | `shrcd` | `shrcd_description` | Excluded In-Universe Sessions | Exclusion Span |
| ---: | ---: | :--- | :--- | ---: | :--- | ---: | :--- |
| 1 | 85592 | PLD | PROLOGIS INC | 18 | Real Estate Investment Trust (REIT) | 6,056 | 2000-06-30 to 2026-07-31 |
| 2 | 86111 | AMT | AMERICAN TOWER CORP NEW | 18 | Real Estate Investment Trust (REIT) | 3,665 | 2012-01-03 to 2026-07-31 |
| 3 | 86339 | CCI | CROWN CASTLE INC | 18 | Real Estate Investment Trust (REIT) | 3,163 | 2014-01-02 to 2026-07-31 |
| 4 | 80100 | SPG | SIMON PROPERTY GROUP INC NEW | 18 | Real Estate Investment Trust (REIT) | 6,559 | 2000-06-30 to 2026-07-31 |
| 5 | 81045 | O | REALTY INCOME CORP | 18 | Real Estate Investment Trust (REIT) | 4,550 | 2005-06-30 to 2026-07-31 |
| 6 | 65947 | WELL | WELLTOWER INC | 18 | Real Estate Investment Trust (REIT) | 5,557 | 2004-06-30 to 2026-07-31 |
| 7 | 89071 | ACN | ACCENTURE PLC IRELAND | 12 | Foreign-incorporated ordinary share | 4,801 | 2007-07-02 to 2026-07-31 |
| 8 | 60097 | MDT | MEDTRONIC PLC | 12 | Foreign-incorporated ordinary share | 2,895 | 2015-01-27 to 2026-07-31 |
| 9 | 12431 | TT | TRANE TECHNOLOGIES PLC | 12 | Foreign-incorporated ordinary share | 4,673 | 2002-01-02 to 2026-07-31 |
| 10 | 18143 | LIN | LINDE PLC NEW | 12 | Foreign-incorporated ordinary share | 1,529 | 2019-07-01 to 2026-07-31 |

*(Additional excluded share code categories in the Russell 1000 history include `shrcd = 48` Shares of Beneficial Interest / REIT Trusts such as pre-merger ProLogis `permno = 80406` with 2,747 excluded sessions, `shrcd = 14` Closed-End Funds, `shrcd = 31` ADRs, `shrcd IN (71, 72)` Limited Partnerships / LLCs, and unlinked 2025–2026 synthetic tickers `permno >= 900001` with `shrcd IS NULL`.)*

---

## Exact SQL Queries Used

### 1. Schema and Duplicate Verification

```sql
-- Verify 37-column schema and types
DESCRIBE silver.features_daily;
DESCRIBE silver."features_daily_v1.3_strict";

-- Verify zero (permno, date) duplicates in both tables
SELECT count(*) - count(DISTINCT (permno, date)) AS duplicate_count
FROM silver.features_daily;

SELECT count(*) - count(DISTINCT (permno, date)) AS duplicate_count
FROM silver."features_daily_v1.3_strict";
```

### 2. Overall Summary Query

```sql
SELECT
    'silver.features_daily' AS table_name,
    count(*) AS total_rows,
    count(*) FILTER (WHERE in_universe) AS in_universe_rows,
    count(DISTINCT permno) AS total_unique_permnos,
    count(DISTINCT permno) FILTER (WHERE in_universe) AS in_universe_unique_permnos,
    round(100.0 * avg(CASE WHEN in_universe THEN cast(features_ready AS DOUBLE) END), 2) AS in_universe_features_ready_pct,
    round(100.0 * avg(CASE WHEN in_universe AND era = 'CRSP' THEN cast(features_ready AS DOUBLE) END), 2) AS crsp_in_universe_ready_pct
FROM silver.features_daily
UNION ALL
SELECT
    'silver."features_daily_v1.3_strict"' AS table_name,
    count(*) AS total_rows,
    count(*) FILTER (WHERE in_universe) AS in_universe_rows,
    count(DISTINCT permno) AS total_unique_permnos,
    count(DISTINCT permno) FILTER (WHERE in_universe) AS in_universe_unique_permnos,
    round(100.0 * avg(CASE WHEN in_universe THEN cast(features_ready AS DOUBLE) END), 2) AS in_universe_features_ready_pct,
    round(100.0 * avg(CASE WHEN in_universe AND era = 'CRSP' THEN cast(features_ready AS DOUBLE) END), 2) AS crsp_in_universe_ready_pct
FROM silver."features_daily_v1.3_strict";
```

### 3. Year-by-Year Comparison Query

```sql
WITH base AS (
    SELECT
        extract(year FROM date)::INT AS year,
        count(*) AS total_rows,
        count(*) FILTER (WHERE in_universe) AS u_rows_base,
        count(DISTINCT permno) FILTER (WHERE in_universe) AS u_permnos_base,
        round(100.0 * avg(CASE WHEN in_universe THEN cast(features_ready AS DOUBLE) END), 2) AS ready_base
    FROM silver.features_daily
    GROUP BY 1
),
strict AS (
    SELECT
        extract(year FROM date)::INT AS year,
        count(*) AS total_rows_strict,
        count(*) FILTER (WHERE in_universe) AS u_rows_strict,
        count(DISTINCT permno) FILTER (WHERE in_universe) AS u_permnos_strict,
        round(100.0 * avg(CASE WHEN in_universe THEN cast(features_ready AS DOUBLE) END), 2) AS ready_strict
    FROM silver."features_daily_v1.3_strict"
    GROUP BY 1
)
SELECT
    b.year,
    b.total_rows,
    b.u_rows_base,
    s.u_rows_strict,
    s.u_rows_strict - b.u_rows_base AS delta_u_rows,
    b.u_permnos_base,
    s.u_permnos_strict,
    s.u_permnos_strict - b.u_permnos_base AS delta_u_permnos,
    b.ready_base,
    s.ready_strict,
    round(s.ready_strict - b.ready_base, 2) AS delta_ready_pct
FROM base b
JOIN strict s USING (year)
ORDER BY b.year;
```

### 4. 3-Pair Spot Check Query

```sql
SELECT
    b.permno,
    b.date,
    b.r_cc AS base_r_cc,
    s.r_cc AS strict_r_cc,
    abs(b.r_cc - s.r_cc) AS diff_r_cc,
    b.ret_5d AS base_ret_5d,
    s.ret_5d AS strict_ret_5d,
    abs(b.ret_5d - s.ret_5d) AS diff_ret_5d,
    b.resid_ret AS base_resid_ret,
    s.resid_ret AS strict_resid_ret,
    abs(b.resid_ret - s.resid_ret) AS diff_resid_ret,
    b.ctx_breadth AS base_ctx_breadth,
    s.ctx_breadth AS strict_ctx_breadth,
    b.y_star AS base_y_star,
    s.y_star AS strict_y_star
FROM silver.features_daily b
JOIN silver."features_daily_v1.3_strict" s USING (permno, date)
WHERE (b.permno = 14593 AND b.date = DATE '2015-06-15')
   OR (b.permno = 10107 AND b.date = DATE '2020-11-16')
   OR (b.permno = 86580 AND b.date = DATE '2024-03-15')
ORDER BY b.date, b.permno;
```

### 5. Excluded Prominent Constituents Query

```sql
WITH excl_days AS (
    SELECT
        b.permno,
        count(*) AS excluded_in_univ_days,
        min(b.date) AS first_excl_date,
        max(b.date) AS last_excl_date
    FROM silver.features_daily b
    JOIN silver."features_daily_v1.3_strict" s USING (permno, date)
    WHERE b.in_universe = TRUE AND s.in_universe = FALSE
    GROUP BY b.permno
),
latest_hist AS (
    SELECT
        security_id AS permno,
        arg_max(ticker, valid_to) AS ticker,
        arg_max(comnam, valid_to) AS comnam,
        arg_max(shrcd, valid_to) AS shrcd
    FROM silver.security_history
    WHERE shrcd IS NOT NULL
    GROUP BY security_id
)
SELECT
    e.permno,
    h.ticker,
    h.comnam,
    h.shrcd,
    CASE h.shrcd
        WHEN 18 THEN 'Real Estate Investment Trust (REIT)'
        WHEN 12 THEN 'Foreign-incorporated ordinary share'
        WHEN 48 THEN 'Shares of beneficial interest / REIT trust'
        WHEN 14 THEN 'Closed-end fund'
        ELSE 'Non-10/11 share code'
    END AS shrcd_description,
    e.excluded_in_univ_days,
    e.first_excl_date,
    e.last_excl_date
FROM excl_days e
LEFT JOIN latest_hist h USING (permno)
WHERE h.ticker IN ('PLD', 'AMT', 'CCI', 'SPG', 'O', 'WELL', 'ACN', 'MDT', 'TT', 'LIN')
  AND e.permno != 80406
ORDER BY e.excluded_in_univ_days DESC;
```
