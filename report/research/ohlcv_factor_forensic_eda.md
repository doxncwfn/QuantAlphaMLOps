# OHLCV & Factor Data Forensic EDA Report

> **Project**: Russell 1000 Quantitative Strategy  
> **Audit Date**: 2026-10-03  
> **Data Period**: 2000-06-30 to 2026-09-25  
> **Auditor**: Data-Audit & Forensic EDA Agent

---

## Executive Summary

```
PRICE_ADJUSTMENT_STATUS      = UNADJUSTED_RAW (CRSP PRC is raw; Yahoo Close is unadjusted, Adj_Close is div+split adjusted)
OHLC_CONSISTENCY             = YES_WITH_CONDITIONS (same basis within each era; CRSP has ~0.5% null Open)
VOLUME_ADJUSTMENT_STATUS     = UNADJUSTED_RAW (raw shares traded in both eras)
TOTAL_RETURN_STATUS          = PARTIAL (CRSP RET available 2000-2024; Yahoo requires Adj_Close computation 2025-2026)
FACTOR_CONVENTION            = DECIMAL_DAILY (0.01 = 1%)
RF_CONVENTION                = DAILY_RISK_FREE_DECIMAL (constant within month, from 1-month T-bill)
DATE_ALIGNMENT               = SAFE_WITH_CAVEAT (WRDS starts 2000-06-30; factor data starts 2000-01-03; join by date is safe where both exist)
DELISTING_INFORMATION        = PARTIAL (CRSP DLSTCD+DLRET available 2000-2024; ABSENT for 2025-2026)
TARGET_t_to_t5_FEASIBILITY   = YES (97.8%+ coverage via trading-session indexing)
SOURCE_TRANSITION_RISK       = MODERATE (63→10 column transition at 2024-12-31→2025-01-02; R_ON std drops 32%; no price discontinuity)
```

---

## 1. Data Inventory

### 1.1 Dataset Summary

| Dataset | Source/Path | Date Range | Rows | Columns | Price Fields | Volume | Factors | Notes |
|---------|------------|-----------|-----:|---------|-------------|--------|---------|-------|
| WRDS 2000 | data/WRDS/2000.parquet | 2000-06-30→2001-06-29 | 258,734 | 63 | PRC, OPENPRC, BIDLO, ASKHI | VOL | — | CRSP, Russell recon year, ~1,085 tickers |
| WRDS 2001–2023 | data/WRDS/{yr}.parquet | July→June each year | ~252K–266K | 63 | Same | Same | — | Consistent CRSP schema |
| WRDS 2024 | data/WRDS/2024.parquet | 2024-07-01→2024-12-31 | 131,758 | 63 | Same | Same | — | **Half-year only** (WRDS ended mid-cycle) |
| Yahoo 2025 | data/WRDS/2025.parquet | 2025-01-02→2025-12-31 | 249,773 | 10 | Open, High, Low, Close, Adj_Close | Volume | — | Yahoo Finance source, ~1,051 tickers |
| Yahoo 2026 | data/WRDS/2026.parquet | 2026-01-02→2026-09-25 | 183,818 | 10 | Same | Same | — | Partial year, ~1,083 tickers |
| Factor data | data/ff.csv | 2000-01-03→2026-07-31 | 6,684 | 8 | — | — | mktrf, smb, hml, rmw, cma, umd, rf | Fama-French 5+UMD, daily, decimal |
| Russell 1000 universe | data/processed/russell1000_all_years.csv | 2000–2026 | 27,130 | 4 | — | — | — | Year, Ticker, Name, Asset_Class |
| CRSP by PERMNO | data/Russell1000/russell1000_by_permno.parquet | 2000-06-30→varies | 6,786,246 | 63 | Same as WRDS | Same | — | Consolidated CRSP file |
| CRSP by year | data/Russell1000/russell1000_by_year.parquet | 2000-06-30→varies | 6,786,246 | 63 | Same | Same | — | Same data, different sort |

> [!IMPORTANT]
> **WRDS files use Russell reconstitution years (June→June), NOT calendar years.** The data begins 2000-06-30, not 2000-01-01. There is NO stock data before 2000-06-30.

### 1.2 Schema Transition

| Period | Schema | Identifier | Columns | Price Convention |
|--------|--------|-----------|---------|-----------------|
| 2000-06-30 → 2024-12-31 | CRSP 63-column | PERMNO (permanent) | PRC, OPENPRC, BIDLO, ASKHI, VOL, RET, RETX, CFACPR, CFACSHR, DLSTCD, DLRET, SHROUT, ... | PRC = **raw unadjusted**, can be negative (bid-ask midpoint) |
| 2025-01-02 → 2026-09-25 | Yahoo 10-column | Ticker (mutable) | Date, Ticker, Name, Sector, Open, High, Low, Close, Adj_Close, Volume | Close = **unadjusted**, Adj_Close = split+dividend adjusted backward |

> [!CAUTION]
> **Critical fields LOST at transition:**
> - PERMNO → Ticker (identifier can change via M&A)
> - RET/RETX (pre-computed returns) → must compute from price
> - CFACPR/CFACSHR (adjustment factors) → implicit in Adj_Close
> - DLSTCD/DLRET (delisting info) → **ABSENT**
> - SHROUT (shares outstanding) → **ABSENT**
> - BID/ASK → **ABSENT**

### 1.3 Data Quality

| Check | CRSP (2000-2024) | Yahoo (2025-2026) |
|-------|------------------|-------------------|
| Duplicate (ticker,date) | **0** across all years | **0** |
| PRC/Close nulls | 11–1,339/yr (declining over time) | **0** |
| PRC negative (bid-ask mid) | 244–1,035/yr (CRSP convention, not error) | N/A |
| Open nulls | 364–3,677/yr (~0.5–1.4%) | **0** |
| Volume zero | 244–1,034/yr | 0 (2025), 1 (2026) |
| H < max(O,C) | 0–3 total (negligible) | **0** |
| L > min(O,C) | 0–1 total | **0** |
| H < L | **0** across all years | **0** |

---

## 2. Price Adjustment Status

### 2.1 CRSP PRC is RAW/UNADJUSTED

**Evidence**: On stock split days, PRC drops by the split ratio:

| Ticker | Date | Split | Prev Close | New PRC | Price Ratio | RET (adjusted) |
|--------|------|-------|----------:|--------:|------------:|------:|
| SUNW | 2000-12-06 | 2:1 | 91.75 | 44.25 | 0.482 | ~0% |
| ORCL | 2000-10-13 | 2:1 | 63.00 | 35.63 | 0.565 | ~0% |
| MSFT | 2003-02-18 | 2:1 | 48.30 | 24.96 | 0.517 | ~0% |
| AAPL | 2014-06-09 | 7:1 | 645.57 | 93.70 | 0.145 | ~0% |
| AAPL | 2020-08-31 | 4:1 | 499.23 | 129.04 | 0.258 | ~0% |
| ANET | 2024-12-04 | 4:1 | 414.46 | 105.43 | 0.254 | ~0% |

**Verification**: `PRC_t / PRC_{t-1} - 1 ≈ RETX` on 99.95–99.99% of non-split days across all sampled years. The few mismatches are exclusively split and large-dividend days.

**Conclusion**: `PRC` stores the actual closing trade price (or bid-ask midpoint when negative). To get split-adjusted prices, divide by `CFACPR`. `RET` and `RETX` are already correctly adjusted by CRSP.

### 2.2 Yahoo Close is UNADJUSTED; Adj_Close is Backward-Adjusted

**Evidence**: `Adj_Close / Close` ratio:

| Ticker | Year | adj_ratio Range | Interpretation |
|--------|------|----------------|----------------|
| AAPL | 2025 | [0.993, 0.997] | Dividends subtracted backward |
| MSFT | 2025 | [0.986, 0.994] | Larger cum. div subtracted |
| JNJ | 2025 | [0.955, 0.984] | High-dividend stock, bigger gap |
| KO | 2025 | [0.953, 0.981] | Similar |

- adj_ratio ≤ 1.000 for ALL observations (0 cases where Adj_Close > Close)
- adj_ratio increases toward 1.0 as dates approach present → **backward-looking adjustment**
- 37 extreme overnight returns (|R_ON| > 0.3) in 2025 are all earnings gaps, not splits

### 2.3 Summary

```
CRSP_PRC = RAW_UNADJUSTED
CRSP_RET = TOTAL_RETURN (split + dividend adjusted)
CRSP_RETX = PRICE_RETURN (split adjusted, no dividends)
YAHOO_CLOSE = UNADJUSTED (split-adjusted, but NOT dividend-adjusted)
YAHOO_ADJ_CLOSE = TOTAL_RETURN_ADJUSTED (split + dividend adjusted backward)
```

---

## 3. OHLC Consistency

### 3.1 CRSP Era

All OHLC fields (PRC, OPENPRC, BIDLO, ASKHI) are on the **same raw/unadjusted basis**. On split days, all four fields show post-split prices.

**However**: OPENPRC has higher null rates (0.5–1.4% per year vs 0.04–0.5% for PRC), which means $R^{ON}$ cannot be computed for ~0.5-1.4% of observations.

### 3.2 Yahoo Era

**100% perfect OHLC consistency**:
- H ≥ max(O,C): 100.00% (all rows)
- L ≤ min(O,C): 100.00%
- H ≥ L: 100.00%
- CC = ON + ID decomposition: max diff = 1.74e-7 (machine precision)

### 3.3 Can OHLC-Derived Features Be Safely Computed?

> **Answer: YES_WITH_CONDITIONS**

**Conditions:**
1. **CRSP era**: Use `abs(PRC)` for close, `abs(OPENPRC)` for open (CRSP negative-price convention). Filter rows where OPENPRC is null (~0.5–1.4%).
2. **Yahoo era**: Use Close and Open directly. No null issues.
3. **Division by zero**: When H = L (same-day flat candle), CloseLoc/Body/Wick features produce division by zero. Observed: ~178 cases/year in CRSP (~0.07%), ~1 case/year in Yahoo. **Must handle with NaN or sentinel**.
4. **Cross-era**: Do NOT compute $R^{ON}$ across the 2024-12-31→2025-01-02 source boundary unless you verify price continuity at the transition.

---

## 4. Volume Adjustment

**Volume is UNADJUSTED (raw shares traded)** in both eras.

Evidence from split events:

| Ticker | Date | Split | Vol Before | Vol After | Vol Ratio |
|--------|------|-------|----------:|----------:|----------:|
| FAST | 2019-05-23 | 2:1 | 1.49M | 4.83M | 3.23 |
| PANW | 2024-12-16 | 2:1 | 1.91M | 6.06M | 3.17 |
| ANET | 2024-12-04 | 4:1 | 1.80M | 5.85M | 3.26 |

Volume ratios are NOT equal to the split ratio (which would indicate pre-adjustment). They reflect actual post-split trading activity.

**Impact on RelVol**: $RelVol_t = \log(V_t / Median_{20}(V)_t)$ requires that the 20-day lookback window does NOT span a split event. Within-split-window contamination would produce a spurious spike. **For CRSP era, filter CFACSHR changes within the 20-day window.**

**Dollar volume**: $\log(C_t \cdot V_t)$ is NOT a valid dollar-volume proxy for CRSP because PRC is raw. Use $\log(|PRC_t| \cdot V_t)$ as an approximation, but note this is NOT split-adjusted. **For a clean series, use SHROUT × |PRC| for market cap, or pre-adjust both.**

---

## 5. Total Return Availability

```
TOTAL_RETURN_AVAILABLE     = PARTIAL
SPLIT_ADJUSTED_AVAILABLE   = YES (CRSP: RETX or PRC/CFACPR; Yahoo: Close is already split-adjusted)
DIVIDEND_ADJUSTED_AVAILABLE = PARTIAL (CRSP: RET; Yahoo: Adj_Close)
```

| Period | Total Return | Price Return | Adjustment Factors | Dividend Amount |
|--------|-------------|-------------|-------------------|-----------------|
| 2000-06-30 → 2024-12-31 | `RET` (pre-computed) | `RETX` (pre-computed) | `CFACPR`, `CFACSHR` | `DIVAMT` |
| 2025-01-02 → 2026-09-25 | Compute from `Adj_Close` | Compute from `Close` | Implicit in Adj_Close | **ABSENT** |

> [!WARNING]
> CRSP `RET` contains string codes (B, C, T, S, A) on ~0–928 rows/year (declining over time). These must be converted to NaN before numeric use.

---

## 6. Factor Data Forensics

### 6.1 Overview

- **6,684 daily observations**, 2000-01-03 to 2026-07-31
- **Zero nulls**, zero duplicates, zero weekend observations
- **Units: DECIMAL** (0.01 = 1%)

### 6.2 Factor Statistics

| Factor | Mean | Std | Min | Max | Median | Annualized Mean |
|--------|------|-----|-----|-----|--------|----------------|
| mktrf | 0.000324 | 0.01232 | -0.1201 | 0.1136 | 0.0007 | 8.2% ✓ |
| smb | 0.000078 | 0.00645 | -0.0458 | 0.0571 | 0.0001 | 2.0% |
| hml | 0.000118 | 0.00780 | -0.0503 | 0.0673 | -0.0001 | 3.0% |
| rmw | 0.000178 | 0.00561 | -0.0296 | 0.0457 | 0.0001 | 4.5% |
| cma | 0.000103 | 0.00455 | -0.0530 | 0.0247 | 0.0000 | 2.6% |
| umd | 0.000141 | 0.01076 | -0.1439 | 0.0712 | 0.0007 | 3.6% |
| rf | 0.000077 | 0.00009 | 0.0000 | 0.0003 | 0.0001 | 1.9% |

**Extreme events validated**: mktrf max +11.36% on 2008-10-13 (GFC rebound); min -12.01% on 2020-03-16 (COVID crash); umd min -14.39% (momentum crash). All consistent with known market events.

---

## 7. RF and MKT-RF Convention

```
STOCK_RETURN_TYPE    = CRSP RET is daily total return (decimal). Yahoo: compute from Adj_Close.
MKT_RF_TYPE          = MKT - RF (market excess return, decimal)
RF_TYPE              = Daily risk-free rate (decimal), constant within month, from 1-month T-bill
EXCESS_RETURN_FORMULA = R_e = R_i - RF (both in decimal)
```

**Verification**: `mktrf + rf` correlates >0.996 with CRSP `vwretd` across all sampled years. Mean absolute difference ~0.0005 (expected due to different market portfolio construction).

**RF properties**:
- Always ≥ 0 (3,313 zero-RF days during ZIRP eras: 2009–2015, 2020–2021)
- Constant within each calendar month (confirmed)
- Never negative (no negative interest rate periods in US)

---

## 8. Stock/Factor Date Alignment

**Key finding**: WRDS data starts 2000-06-30 (Russell reconstitution year convention). Factor data starts 2000-01-03. **No stock data exists for Jan–June 2000.**

When both exist, alignment is safe:

| Check | Result |
|-------|--------|
| Every CRSP stock date has factor data? | **YES** (0 stock-only dates for 2000-2024) |
| Every factor date has stock data? | NO — Jan–June 2000 has factor data but no stock data |
| Yahoo dates align with factors? | **YES** (2025: 250/250 overlap; 2026: factor data ends 2026-07-31, stock data continues to 2026-09-25) |
| Weekend observations? | **NONE** in either dataset |
| Join by calendar date safe? | **YES**, where both sources have data |

> [!NOTE]
> Factor data ends 2026-07-31. Stock data extends to 2026-09-25. **Factor-dependent features cannot be computed for Aug–Sep 2026.**

---

## 9. Delistings and Missing Future Data

### 9.1 CRSP Era (2000-2024)

- **1,936 total delisting events** identified via DLSTCD
- **905 (47%) have numeric DLRET**, 1,031 (53%) are missing or string-coded
- Most common DLSTCD: 231 (merger), 233 (going private), 241 (acquired), 574/584 (bankruptcy)

> [!WARNING]
> 2024 has 1,022 DLSTCD=100 entries — these are **NOT delistings** but end-of-dataset markers for active stocks (DLRET="A"). Filter these out.

### 9.2 Yahoo Era (2025-2026)

- **NO delisting fields** whatsoever
- 2025: 54 tickers disappear >7 days before last date
- 2026: 66 tickers disappear >7 days before last date
- No way to determine if disappearance is delisting, merger, ticker change, or data gap

### 9.3 Ticker Disappearance

- **2,834 unique PERMNOs** in CRSP data
- **1,800 (63.5%)** disappear before 2024-06-30
- **324** had multiple tickers (name changes)
- **184 rows** with NWPERM (successor PERMNO for M&A tracing)

---

## 10. Last Observed Price as Delisting Fallback

```
LAST_PRICE_FALLBACK = CONDITIONALLY_ACCEPTABLE
```

**Evidence**:
- Most disappearances with DLSTCD=200/231/233 (mergers/acquisitions) have meaningful last-day trading: substantial volume, normal price action
- Max calendar gaps are 3-4 days (weekends), not long suspensions
- **Exception**: Bankruptcy cases (DLSTCD=574/584) like SIVB, FRC show extreme price drops; last price may be far from realizable
- **Exception**: Yahoo-era disappearances have no delisting information; fallback quality is unknown

**Conditions for acceptance**:
1. For CRSP era: use DLRET where available; fall back to last price ONLY for merger/acquisition codes (200, 231, 233, 241)
2. For bankruptcy codes (574, 584): use DLRET if available; if DLRET is string-coded, assign conservative estimate (e.g., -100% or -30%)
3. For Yahoo era: flag as UNRESOLVED; cannot determine if disappearance is benign (M&A) or adverse (delisting)

---

## 11. Target Return t→t+5 Feasibility

**YES**, the target $Y_{i,t} = \log(C_{i,t+5} / C_{i,t})$ can be constructed with high coverage:

| Year | Coverage | Unexpected Missing | Calendar Gap Median |
|------|---------|-------------------|-------------------|
| 2005 | 97.8% | 308 | 7 days |
| 2010 | 97.8% | 626 | 7 days |
| 2015 | 97.9% | 349 | 7 days |
| 2020 | 97.9% | 276 | 7 days |
| 2024 | 96.1% | 10 | 7 days |
| 2025 | 97.9% | 0 | 7 days |

Missing ~2% are almost entirely last-5-sessions-of-year tail effects. Calendar gap median of 7 days (5 trading + 2 weekend) confirms correct trading-session alignment.

> [!IMPORTANT]
> **Use trading-session indexing**, NOT calendar arithmetic.
> $t+5$ = 5th next valid trading observation for the same stock. Do NOT add 5 calendar days.

**Which return definition?**

| Era | Price Return Target | Total Return Target |
|-----|-------------------|-------------------|
| CRSP (2000-2024) | $\prod_{s=1}^{5}(1+RETX_{t+s}) - 1$ | $\prod_{s=1}^{5}(1+RET_{t+s}) - 1$ |
| Yahoo (2025-2026) | $Close_{t+5}/Close_t - 1$ | $AdjClose_{t+5}/AdjClose_t - 1$ |

---

## 12. How t+5 Should Be Implemented

1. Within each stock's time series, sort by date ascending
2. $t+5$ = row at position `current_position + 5` for the same stock
3. If the stock has fewer than 5 future observations, the target is NaN
4. All Russell 1000 constituents share the common US trading calendar (confirmed: 0 stocks with dates outside common calendar)
5. Missing stock dates should be interpreted as stock-specific halts or missing data, NOT market holidays

**Cross-year boundary**: WRDS files overlap at June 30 (reconstitution date). When computing t+5 across a file boundary, use the consolidated dataset (concatenate all yearly files). The `russell1000_by_year.parquet` already provides this.

---

## 13. Feature Formula Feasibility

| Feature | 2020 CRSP Computable | 2020 CRSP Missing% | 2025 Yahoo Missing% | Div-by-Zero Risk | Notes |
|---------|--------------------:|-------------------:|--------------------:|-----------------:|-------|
| $R^{CC}$ | 257,710 | 0.51% | 0.42% | None | 28 extreme (>50%) in CRSP |
| $R^{ON}$ | 256,921 | 0.81% | 0.42% | None | Higher missing in CRSP (null OPENPRC) |
| $R^{ID}$ | 257,954 | 0.41% | 0.00% | None | — |
| Range | 258,747 | 0.11% | 0.00% | None | 178 zero-range in CRSP |
| CloseLoc | 258,569 | — | — | 178 (0.07%) | When H = L |
| Body | Same | — | — | 178 (0.07%) | Same div-by-zero |
| UpperWick | Same | — | — | 178 (0.07%) | Same |
| LowerWick | Same | — | — | 178 (0.07%) | Same |
| RelVol | 238,142 | 8.06% | 7.99% | log(0) risk | 20-day warmup; zero-vol days in CRSP |

All formulas are **implementable** with appropriate null handling.

---

## 14. Temporal Lookback Requirements

| Feature | Direct Lookback | Indirect Lookback | Total | With 60-Session Model | Earliest Valid Date |
|---------|:-:|:-:|:-:|:-:|---|
| R_CC_1d | 1 | 0 | 1 | 60 | 2000-09 |
| R_CC_5d | 5 | 0 | 5 | 64 | 2000-09 |
| R_CC_20d | 20 | 0 | 20 | 79 | 2000-10 |
| R_CC_60d | 60 | 0 | 60 | 119 | 2001-01 |
| R_ON | 1 | 0 | 1 | 60 | 2000-09 |
| R_ID | 0 | 0 | 0 | 59 | 2000-09 |
| Range | 0 | 0 | 0 | 59 | 2000-09 |
| CloseLoc | 0 | 0 | 0 | 59 | 2000-09 |
| RelVol | 20 | 0 | 20 | 79 | 2000-10 |
| Vol_20d | 20 | 1 | 21 | 80 | 2000-10 |
| Vol_60d | 60 | 1 | 61 | 120 | 2001-01 |
| Beta_252d | 252 | 1 | 253 | 312 | 2001-09 |
| Alpha_252d | 252 | 1 | 253 | 312 | 2001-09 |
| **ResidMom_20d** | **20** | **252** | **272** | **331** | **~2001-10** |

> [!IMPORTANT]
> **Maximum historical buffer: 331 trading sessions (~1.3 calendar years)**
> ResidMom_20d requires 252d rolling regression + 20d summation + 59d model window.
> Data starts 2000-06-30 → **first fully valid prediction date ≈ late 2001**.

---

## 15. Corporate Action Edge Cases

### 15.1 Stock Splits

On split days, raw PRC drops by the split ratio. `RET`/`RETX` remain correctly adjusted. Impact on features:

- **$R^{CC}$ from raw PRC**: CATASTROPHICALLY WRONG (~-50% for 2:1 split)
- **$R^{ON}$ from raw prices**: Same issue
- **$R^{ID}$**: OK (same-day O and C both post-split)
- **Range**: OK (intraday ratio)
- **RelVol**: Spike due to volume change

> [!CAUTION]
> **NEVER compute returns from raw PRC across days.** Use `RET` or `RETX` for CRSP era. Or adjust PRC by CFACPR first.

### 15.2 Reverse Splits

Examples: JAVA 2007-11-12 (1:4 reverse, PRC jumps 4x), GE 2021-08-02 (1:8 reverse), AIG 2009-07-01 (1:20 reverse). Same issue — use CRSP pre-computed returns.

### 15.3 True Price Crashes

Examples: Enron 2001-11-28 (PRC drops from $4.11→$0.61, RET=-85.2%), SIVB 2023-03-13 (VOL=0, delisting). These are genuine economic events, NOT adjustment artifacts. Features will correctly reflect extreme moves.

---

## 16. Source Transition Audit

### 16.1 Boundary

- **Last CRSP date**: 2024-12-31
- **First Yahoo date**: 2025-01-02
- **Gap**: 2 calendar days (normal New Year's holiday)

### 16.2 Price Continuity

20 common tickers sampled: all show normal overnight returns at transition (range -0.6% to +1.4%). **No price discontinuity detected.**

### 16.3 Distribution Shift at Transition

| Feature | 2024 Std | 2025 Std | Change |
|---------|----------|----------|--------|
| R_CC | 0.0292 | 0.0257 | -12% |
| **R_ON** | **0.0227** | **0.0154** | **-32%** ⚠️ |
| R_ID | 0.0186 | 0.0212 | +14% |
| Range | 0.0180 | 0.0202 | +13% |

> [!WARNING]
> **R_ON standard deviation drops 32%** at the source transition. Possible causes:
> 1. Market regime change (2024 H2 vs 2025 full year)
> 2. Yahoo may report Open prices differently than CRSP OPENPRC
> 3. Universe composition change (33 dropped, 56 added)
>
> **Recommendation**: Investigate further before using R_ON in cross-era models. Consider normalizing features by era or using R_CC only.

### 16.4 Feature Learnability Risk

A model could potentially learn "this came from source A vs B" through:
- Distribution shift in R_ON
- Missing OPENPRC (CRSP) vs perfect Open (Yahoo)
- Different null patterns
- Sector field available only in Yahoo

**Mitigation**: Cross-sectional ranking within each date (planned) naturally mitigates absolute distribution shifts.

---

## 17. Statistical Distribution EDA

### 17.1 R_CC by Period

| Year | Mean | Std | 1% | 99% | Regime |
|------|------|-----|-----|-----|--------|
| 2001 | -0.00137 | **0.0365** | -0.102 | 0.093 | Dot-com bust |
| 2005 | +0.00005 | 0.0223 | -0.045 | 0.048 | Calm |
| 2008 | -0.00184 | **0.0564** | -0.158 | 0.149 | **GFC peak vol** |
| 2010 | +0.00103 | 0.0212 | -0.046 | 0.050 | Recovery |
| 2015 | -0.00042 | 0.0281 | -0.072 | 0.064 | Mid-cycle |
| 2020 | +0.00145 | 0.0275 | -0.061 | 0.067 | COVID recovery |
| 2024 | +0.00037 | 0.0292 | -0.063 | 0.063 | Normal |
| 2025 | +0.00006 | 0.0257 | -0.073 | 0.068 | Normal |
| 2026 | +0.00003 | 0.0279 | -0.077 | 0.076 | Normal |

**No catastrophic structural breaks detected.** Distributions widen during crisis periods (2001, 2008) as expected. The 2024→2025 transition shows gradual change, not a cliff.

---

## Resolved Questions

### Q1: Can we safely implement $Y_{i,t} = \log(C_{i,t+5}/C_{i,t})$?

**YES, with modifications per era:**
- **CRSP era (2000-2024)**: Use `RET` or `RETX` (pre-computed, adjustment-correct). If log return is needed: $Y_{i,t} = \log\prod_{s=1}^{5}(1+RET_{t+s})$. Do NOT compute from raw PRC directly.
- **Yahoo era (2025-2026)**: Use `Close_{t+5}/Close_t` for price return, or `AdjClose_{t+5}/AdjClose_t` for total return. Close is already split-adjusted in Yahoo.
- **Recommended**: Use price return (RETX / Close) for the target, as it avoids dividend timing assumptions. Apply total-return correction only for factor regressions.

### Q2: Are O/H/L/C consistently adjusted for candle features?

**YES_WITH_CONDITIONS.** All OHLC fields are on the same basis within each era. Conditions:
1. Use `abs()` for CRSP prices (neg-price convention)
2. Filter ~0.5–1.4% null OPENPRC rows in CRSP
3. Handle ~0.07% H=L division-by-zero cases

### Q3: Is Volume suitable for relative-volume features?

**YES**, but with care:
- Volume is raw (unadjusted) in both eras
- RelVol requires 20-day warmup
- **Must ensure the 20-day window does not span a stock split** (CRSP: check CFACSHR changes)
- Zero-volume days (~0.2–0.4%/year in CRSP) should be excluded or handled as NaN

### Q4: Does the data provide a reliable total-return series?

**PARTIAL:**
- CRSP era: YES, via `RET` (pre-computed total return with dividends)
- Yahoo era: YES, via `Adj_Close_{t}/Adj_Close_{t-1} - 1` (backward-adjusted for dividends)
- **Caveat**: The two methods may have slight differences in dividend reinvestment assumptions

### Q5: Are the factor columns MKT-RF, SMB, HML, RMW, CMA, UMD with RF separately?

**YES.** Confirmed:
- `mktrf` = market excess return = $R_M - RF$ (correlation >0.996 with CRSP vwretd + rf)
- `rf` = daily risk-free rate (constant within month, from T-bill)
- All in decimal format (0.01 = 1%)
- All 6 factors + rf, daily, no missing values

### Q6: Can we correctly construct $R^e_i = R_i - RF$?

**YES** for all years where both stock and factor data exist:
- CRSP era: $R^e_i = RET_i - rf_t$ (both decimal)
- Yahoo era: $R^e_i = (AdjClose_t/AdjClose_{t-1} - 1) - rf_t$
- **Caveat**: Factor data ends 2026-07-31; stock data extends to 2026-09-25. Cannot compute excess returns for Aug–Sep 2026.

### Q7: Can we construct causal six-factor residuals?

**YES.** The 252-day rolling regression requires:
- Stock excess return: available
- Factor returns (mktrf, smb, hml, rmw, cma, umd): available
- Causal constraint: use $\hat{\beta}_{s-1}$ from regression on $[s-252, s-1]$ applied to factor returns at $s$
- First valid residual: ~252 sessions after data start ≈ 2001-09

### Q8: What is the safest delisting policy?

| Era | Policy |
|-----|--------|
| CRSP (2000-2024) | Use numeric `DLRET` where available (~47%). For string-coded DLRET (B,C,T,S,A), assign NaN. For DLSTCD=100 (active), ignore. For missing DLRET with DLSTCD ∈ {200,231,233,241} (M&A), assign 0. For DLSTCD ∈ {500,520,560,574,584} (adverse), assign -30% conservative penalty. |
| Yahoo (2025-2026) | `UNRESOLVED_FROM_DATA`. No delisting information available. Flag disappeared tickers; do not assume any specific return. |

### Q9: What is the longest historical buffer?

**331 trading sessions** (~1.3 calendar years) for `ResidMom_20d` with the 60-session model window. Data starts 2000-06-30 → first valid prediction ≈ **late 2001**.

### Q10: Are there source-transition inconsistencies that invalidate the feature set?

**The feature set is NOT invalidated**, but requires mitigation:
1. **R_ON distribution shift** (-32% std at transition): use cross-sectional ranking to normalize
2. **OPENPRC null rate** changes from ~0.5% (CRSP) to 0% (Yahoo): minor
3. **No splits detected in Yahoo data**: Yahoo Close appears already split-adjusted, while CRSP PRC is raw → different adjustment basis requires different computation paths
4. **Missing delisting info for 2025-2026**: creates survivorship bias risk in t+5 target for ~120 disappeared tickers

---

## Unresolved Questions

1. **Why does R_ON std drop 32% at source transition?** Could be regime, data source, or Open price definition differences. Requires deeper investigation.
2. **Yahoo-era delistings**: 120 tickers disappear in 2025-2026 with no delisting return. Cannot determine if these are benign (M&A) or adverse.
3. **Jan-June 2000 stock data**: Factor data exists but no stock data. The 252-day rolling window for beta/alpha cannot look back into this period.
4. **Factor data gap Aug-Sep 2026**: Factor data ends 2026-07-31; stock data extends to 2026-09-25.
5. **Yahoo Adj_Close backward adjustment timing**: When Yahoo recalculates Adj_Close (e.g., after a dividend goes ex-date), historical values change. Our snapshot may differ from future downloads.
6. **CRSP RET string codes**: 928 (2000) to 0 (2024) string-coded returns per year. Causes vary (halts, suspensions, no prior price).

---

## Recommended Data Convention

For the downstream feature pipeline:

### Returns

| Purpose | CRSP Formula | Yahoo Formula |
|---------|-------------|--------------|
| Daily price return | `RETX` | `Close_t / Close_{t-1} - 1` |
| Daily total return | `RET` | `AdjClose_t / AdjClose_{t-1} - 1` |
| Excess return | `RET - rf` | `(AdjClose_t/AdjClose_{t-1} - 1) - rf` |
| Target (5-day) | $\log\prod(1+RETX_{t+s})$ | $\log(Close_{t+5}/Close_t)$ |

### OHLC Features

| Feature | CRSP | Yahoo |
|---------|------|-------|
| Close | `abs(PRC)` | `Close` |
| Open | `abs(OPENPRC)` | `Open` |
| High | `ASKHI` | `High` |
| Low | `BIDLO` | `Low` |
| Volume | `VOL` | `Volume` |

### Key Rules

1. **CRSP**: use absolute values for PRC and OPENPRC (negative = bid-ask midpoint)
2. **CRSP**: NEVER compute returns from raw PRC across days; use RET/RETX
3. **Yahoo**: Close is unadjusted for dividends but split-adjusted; safe for cross-day returns
4. **Volume**: raw in both; check for splits within lookback windows
5. **Factors**: decimal, daily; join by date
6. **RF**: subtract from total return for excess return computation
7. **t+5**: use trading-session indexing, not calendar arithmetic

---

## Final Verdict

$$
\boxed{\textbf{READY\_FOR\_FEATURE\_SPECIFICATION}}
$$

**With the following mandatory conditions before freezing the specification:**

1. **Implement dual-path return computation** (CRSP vs Yahoo) with a validation test at the 2024-12-31 boundary
2. **Handle delistings** per the era-specific policy in Q8
3. **Investigate the R_ON distribution shift** before including R_ON in the final feature set, or rely on cross-sectional ranking to normalize
4. **Accept the data gap**: no stock data before 2000-06-30; no factor data after 2026-07-31
5. **First valid prediction date ≈ late 2001** due to 331-session lookback requirement
6. **Filter H=L rows** for candle features (division by zero)
7. **Filter CRSP OPENPRC nulls** (~0.5-1.4%) for overnight/intraday return features
