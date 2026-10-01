# Russell 1000 Historical Dataset: Data Quality & Integrity Audit Report

> **Project**: Specialized Quantitative Finance / Alpha MLOps Platform  
> **Generated UTC**: 2026-09-29 20:31:43 UTC  
> **Audited Universe**: Russell 1000 Historical Constituents (2000–2026, 27 complete annual panels)  
> **Total Annual Holdings Audited**: 27,130 records across 27 annual panels  
> **Total Exceptions Logged**: 4134 findings (255 Critical, 1018 High, 789 Medium)  

---

## Executive Summary

This audit establishes a rigorous, evidence-based data quality, coverage, and survivorship assessment of the historical Russell 1000 constituent membership and WRDS daily market data.
The objective is to determine whether the existing dataset is trustworthy enough to proceed to Point-in-Time (PiT) universe reconstruction and leakage-resistant 40-day historical ML panel construction.

### Key High-Level Findings & Semantic Clarifications
1. **Membership Continuity & Unbroken Panel**: All 27 annual datasets span 2000–2026 without gaps, providing an unbroken continuous 27-year time series across raw, processed, and WRDS tiers.
2. **Correct Membership Period Semantics**: An annual Russell 1000 list does **not** represent membership for January 1 through December 31 of that calendar year. Instead, each list is a constituent snapshot defining the index universe for the subsequent reconstitution cycle ending at the next late-June boundary ($t_{June} \rightarrow t_{June+1}$).
3. **WRDS Availability Cutoff & Crawled Dataset Continuation**: WRDS/CRSP daily market data available to our team terminates on **2024-12-31**. For 2025–2026, daily price data derives from a crawled dataset continuation. This is a data-source boundary caused by WRDS license availability, not a methodology shift in the Russell 1000 index.
4. **The 2024/2025 Multi-Source Stitching**: The 2024 Russell list period (`2024-06-30 -> 2025-06-30`) is covered by WRDS for the first half (128 trading days, July–Dec 2024) and the crawled dataset for the second half (122 trading days, Jan–June 2025), delivering **98.90% total period coverage** across 250 available trading days.
5. **Source Boundary Verification (2024-12-31 / 2025-01-02)**: Validation across 998 securities present on both sides confirms high price continuity (median overnight return = +0.61%). 9 securities exhibit split-ratio jumps resulting from crawled data pre-adjustments.
6. **Continuous 40-Day Lookback Sufficiency**: Under continuous multi-year price stitching across annual file boundaries and the WRDS-to-crawled transition, **94.85% of all historical constituent-days possess complete 40-trading-day feature windows** (with Year 2024 achieving **98.32%**, and all cohorts from 2005 onward exceeding 94.0%). This cross-boundary stitching eliminates artificial annual ramp-up deficits, recovering **1,058,070 observations (+15.92%)** that were previously falsely excluded.
7. **Coverage & Delisting Metrics**: Overall mean membership-period coverage is **97.70%** (calendar-year baseline: 97.44%), with 30 zero-coverage instances and 1936 explicit CRSP delisting events retained.

---

## Systematic Answers to Audit Core Questions

### 1. What date does each Russell list actually represent?
Each Russell list represents a specific constituent snapshot captured on an exact date directly documented in its source file:
- **21 Official Late-June Reconstitution Lists**: 2000 (`2000-06-30`), 2001 (`2001-06-30`), 2002 (`2002-06-30`), 2003 (`2003-06-30`), 2004 (`2004-06-25`), 2005 (`2005-06-24`), 2006 (`2006-06-30`), 2007 (`2007-06-22`), 2009 (`2009-06-29`), 2010 (`2010-06-28`), 2011 (`2011-06-27`), 2012 (`2012-06-25`), 2013 (`2013-06-28`), 2014 (`2014-06-27`), 2015 (`2015-06-26`), 2016 (`2016-06-27`), 2017 (`2017-06-26`), 2018 (`2018-06-25`), 2020 (`2020-06-29`), 2021 (`2021-06-28`), 2022 (`2022-06-24`).
- **2008**: Inferred late-June reconstitution (`2008-06-27`, source `2008.csv`).
- **Off-Cycle Snapshot Dates**:
  - **2019**: `2019-07-31` (iShares ETF export snapshot).
  - **2023**: `2023-11-15` (iShares ETF export snapshot; AAPL=$188.01, MSFT=$369.67).
  - **2024**: `2024-07-01` (iShares ETF export snapshot; MSFT=$456.73, AAPL=$216.75).
  - **2025**: `2025-06-30` (iShares ETF XML export).
  - **2026**: `2026-09-15` (iShares ETF XML export).

### 2. What period was each list actually used for?
Each list was used as the constituent universe for the subsequent Russell annual reconstitution cycle:
- `2000` list $\rightarrow$ used for `2000-06-30 -> 2001-06-29`
- `2001` list $\rightarrow$ used for `2001-07-02 -> 2002-06-28`
- `...`
- `2023` list $\rightarrow$ used for `2023-06-30 -> 2024-06-28` (note: November 15 snapshot applied backwards to June 30, creating a 138-day hindsight mismatch)
- `2024` list $\rightarrow$ used for `2024-06-30 -> 2025-06-30`
- `2025` list $\rightarrow$ used for `2025-06-30 -> 2026-06-30`
- `2026` list $\rightarrow$ used for `2026-06-30 -> 2027-06-30` (partial through 2026-09-25)

### 3. What daily data source covers each part of that period?
- **2000 through 2023 Periods**: 100% covered by WRDS/CRSP daily stock files (`2000.parquet` to `2023.parquet`).
- **2024 Period (`2024-06-30 -> 2025-06-30`)**: Mixed coverage:
  - First half (`2024-07-01 -> 2024-12-31`, 128 days): WRDS/CRSP (`2024.parquet`).
  - Second half (`2025-01-02 -> 2025-06-30`, 122 days): crawled dataset (`2025.parquet`).
- **2025 Period (`2025-06-30 -> 2026-06-30`)**: 100% covered by crawled dataset:
  - First half (`2025-07-01 -> 2025-12-31`, 128 days): `2025.parquet`.
  - Second half (`2026-01-02 -> 2026-06-30`, 127 days): `2026.parquet`.
- **2026 Period (`2026-06-30 -> 2027-06-30`)**: crawled dataset (`2026.parquet`, 61 trading days through 2026-09-25).

### 4. Where does WRDS end?
WRDS/CRSP daily stock data terminates strictly on **2024-12-31** at market close (`data/WRDS/2024.parquet`). Zero observations exist in WRDS for calendar year 2025 onward.

### 5. Where does the crawled dataset begin?
The crawled dataset begins on **2025-01-02** at market open (`data/WRDS/2025.parquet`), which was the immediate consecutive trading day following New Year's Day 2025.

### 6. Are the two daily datasets semantically compatible?
They are **operationally compatible for OHLCV prices with documented caveats**:
- **Close Prices**: Raw WRDS $|PRC|$ and crawled `Close` exhibit $>0.999$ correlation for continuous non-split equities.
- **Volume**: Directly compatible after casting (`VOL` Int64 $\leftrightarrow$ `Volume` Float64).
- **Date**: Directly equivalent after formatting (`date` String $\leftrightarrow$ `Date` Datetime64).
- **Semantic Gaps & Incompatibilities**:
  1. *Identifier Continuity*: WRDS provides CRSP `PERMNO`. Crawled dataset provides **only string ticker**, creating vulnerability to symbol reuse.
  2. *Corporate Actions*: Crawled data pre-applies stock splits into OHLC prices for certain names, whereas WRDS provides raw unadjusted prices with adjustment factor `CFACPR`.
  3. *Delisting Events*: Crawled dataset lacks delisting codes (`DLSTCD`) and terminal returns (`DLRET`).

### 7. Which securities can be reliably linked across the source boundary?
- **998 securities** were successfully linked on both sides across the 2024-12-31 / 2025-01-02 boundary.
- **960+ continuous equities** have overnight price returns $|r| \le 15\%$ and link reliably.
- **9 securities** require split factor reconciliation due to pre-applied split adjustments (e.g. `FAST`, `COKE`, `NFLX`, `MNST`, `BKNG`, `IBKR`).
- **8 dual-class tickers** (`HEI`, `LEN`, `UHAL`, `CWEN`, `WSO`, `BIO`, `MKC`, `TAP`) require class suffix alignment (`HEIA` vs `HEI.A`).
- **24 WRDS securities** absent from Crawled represent year-end delistings or spinoffs.

### 8. Which previous audit conclusions remain valid?
- **VALID**: OHLC bounding integrity checks ($High \ge \max(O,C)$, $Low \le \min(O,C)$).
- **VALID**: Zero duplicate records in processed constituent lists.
- **VALID**: 20 CRSP recycled tickers and >150 ticker rebranding events.
- **VALID**: Reconciled extreme returns ($>90\%$ explained by `CFACPR` stock splits).
- **VALID**: 1,901 CRSP delisting events recorded in WRDS.
- **VALID**: Idempotency acceptance test and deterministic pipeline execution.

### 9. Which conclusions must be recomputed?
- **RECOMPUTED**: Annual constituent coverage: previously calculated as calendar-year coverage; now correctly calculated as **membership-period coverage** (96.5% – 99.0%).
- **RECOMPUTED**: 40-day model eligibility: previously suffered from artificial annual file boundary deficits (where 2024 collapsed to 68.75% and 2026 to 30.51%); now correctly calculated under **continuous cross-boundary stitching** across all 27 annual cohorts (overall 94.85% eligible, 2024 reaching 98.32%), recovering 1,058,070 observations falsely excluded by annual file boundaries.
- **RECOMPUTED**: Year 2024 coverage: previously flagged as 50% missing; now correctly resolved as 128 days WRDS + 122 days Crawled = 250 days (98.9% complete).

### 10. Is the resulting dataset ready for PiT universe construction?
**YES, provided the following 4 mandatory data engineering steps are enforced**:
1. Join WRDS and crawled datasets continuously across the 2024-12-31 / 2025-01-02 boundary.
2. Apply split reconciliation factors to the 27 pre-adjusted crawled securities.
3. Bind crawled 2025–2026 tickers to CRSP `PERMNO` using the 2024-12-31 entity cross-walk.
4. In 2023, account for the 138-day hindsight mismatch by verifying active trading status between June 30, 2023 and November 15, 2023.

---

## Formal Russell List & Intended Period Mapping

| List Year | Source Snapshot Date | Source File | Format | Evidence Type | Intended Period | Price Sources | Schema | Hindsight Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2000** | 2000-06-30 | `2000.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2000-06-30 $\rightarrow$ 2001-06-29 | WRDS/CRSP | `CRSP` | 0 (aligned) |
| **2001** | 2001-06-30 | `2001.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2001-07-02 $\rightarrow$ 2002-06-28 | WRDS/CRSP | `CRSP` | 0 (aligned) |
| **2002** | 2002-06-30 | `2002.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2002-07-01 $\rightarrow$ 2003-06-30 | WRDS/CRSP | `CRSP` | 0 (aligned) |
| **2003** | 2003-06-30 | `2003.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2003-06-30 $\rightarrow$ 2004-06-30 | WRDS/CRSP | `CRSP` | 0 (aligned) |
| **2004** | 2004-06-25 | `2004.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2004-06-30 $\rightarrow$ 2005-06-30 | WRDS/CRSP | `CRSP` | 5 days |
| **2005** | 2005-06-24 | `2005.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2005-06-30 $\rightarrow$ 2006-06-30 | WRDS/CRSP | `CRSP` | 6 days |
| **2006** | 2006-06-30 | `2006.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2006-06-30 $\rightarrow$ 2007-06-29 | WRDS/CRSP | `CRSP` | 0 (aligned) |
| **2007** | 2007-06-22 | `2007.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2007-07-02 $\rightarrow$ 2008-06-30 | WRDS/CRSP | `CRSP` | 10 days |
| **2008** | 2008-06-27 | `2008.csv` | CSV | `INFERRED_PERIOD` | 2008-06-30 $\rightarrow$ 2009-06-30 | WRDS/CRSP | `CRSP` | 3 days |
| **2009** | 2009-06-29 | `2009.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2009-06-30 $\rightarrow$ 2010-06-30 | WRDS/CRSP | `CRSP` | 1 days |
| **2010** | 2010-06-28 | `2010.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2010-06-30 $\rightarrow$ 2011-06-30 | WRDS/CRSP | `CRSP` | 2 days |
| **2011** | 2011-06-27 | `2011.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2011-06-30 $\rightarrow$ 2012-06-29 | WRDS/CRSP | `CRSP` | 3 days |
| **2012** | 2012-06-25 | `2012.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2012-07-02 $\rightarrow$ 2013-06-28 | WRDS/CRSP | `CRSP` | 7 days |
| **2013** | 2013-06-28 | `2013.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2013-07-01 $\rightarrow$ 2014-06-30 | WRDS/CRSP | `CRSP` | 3 days |
| **2014** | 2014-06-27 | `2014.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2014-06-30 $\rightarrow$ 2015-06-30 | WRDS/CRSP | `CRSP` | 3 days |
| **2015** | 2015-06-26 | `2015.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2015-06-30 $\rightarrow$ 2016-06-30 | WRDS/CRSP | `CRSP` | 4 days |
| **2016** | 2016-06-27 | `2016.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2016-06-30 $\rightarrow$ 2017-06-30 | WRDS/CRSP | `CRSP` | 3 days |
| **2017** | 2017-06-26 | `2017.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2017-06-30 $\rightarrow$ 2018-06-29 | WRDS/CRSP | `CRSP` | 4 days |
| **2018** | 2018-06-25 | `2018.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2018-07-02 $\rightarrow$ 2019-06-28 | WRDS/CRSP | `CRSP` | 7 days |
| **2019** | 2019-07-31 | `2019.json` | JSON | `USED_FOR_PERIOD` | 2019-07-01 $\rightarrow$ 2020-06-30 | WRDS/CRSP | `CRSP` | 30 days |
| **2020** | 2020-06-29 | `2020.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2020-06-30 $\rightarrow$ 2021-06-30 | WRDS/CRSP | `CRSP` | 1 days |
| **2021** | 2021-06-28 | `2021.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2021-06-30 $\rightarrow$ 2022-06-30 | WRDS/CRSP | `CRSP` | 2 days |
| **2022** | 2022-06-24 | `2022.pdf` | PDF | `OBSERVED_SNAPSHOT` | 2022-06-30 $\rightarrow$ 2023-06-30 | WRDS/CRSP | `CRSP` | 6 days |
| **2023** | 2023-11-15 | `2023.json` | JSON | `USED_FOR_PERIOD` | 2023-06-30 $\rightarrow$ 2024-06-28 | WRDS/CRSP | `CRSP` | 138 days |
| **2024** | 2024-07-01 | `2024.json` | JSON | `USED_FOR_PERIOD` | 2024-06-30 $\rightarrow$ 2025-06-30 | WRDS/CRSP + crawled | `mixed` | 1 days |
| **2025** | 2025-06-30 | `2025.xls` | XLS | `OBSERVED_SNAPSHOT` | 2025-06-30 $\rightarrow$ 2026-06-30 | crawled | `friend_crawled` | 0 (aligned) |
| **2026** | 2026-09-15 | `2026.xls` | XLS | `USED_FOR_PERIOD` | 2026-06-30 $\rightarrow$ 2027-06-30 | crawled | `friend_crawled` | 77 days |

---

## Membership-Period Price Coverage Summary

| List Year | Period Start | Period End | Total Trading Days | WRDS Days | Crawled Days | Mean Coverage | Median Coverage | Full Coverage ($\ge 95\%$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2000** | 2000-06-30 | 2001-06-29 | 252 | 252 | 0 | 94.42% | 100.00% | 90.1% |
| **2001** | 2001-07-02 | 2002-06-28 | 247 | 247 | 0 | 96.19% | 100.00% | 93.0% |
| **2002** | 2002-07-01 | 2003-06-30 | 252 | 252 | 0 | 98.03% | 100.00% | 96.5% |
| **2003** | 2003-06-30 | 2004-06-30 | 253 | 253 | 0 | 98.37% | 100.00% | 96.8% |
| **2004** | 2004-06-30 | 2005-06-30 | 254 | 254 | 0 | 97.69% | 100.00% | 96.1% |
| **2005** | 2005-06-30 | 2006-06-30 | 253 | 253 | 0 | 96.44% | 100.00% | 92.9% |
| **2006** | 2006-06-30 | 2007-06-29 | 251 | 251 | 0 | 97.03% | 100.00% | 94.3% |
| **2007** | 2007-07-02 | 2008-06-30 | 252 | 252 | 0 | 96.43% | 100.00% | 94.4% |
| **2008** | 2008-06-30 | 2009-06-30 | 253 | 253 | 0 | 97.35% | 100.00% | 95.3% |
| **2009** | 2009-06-30 | 2010-06-30 | 253 | 253 | 0 | 98.23% | 100.00% | 96.7% |
| **2010** | 2010-06-30 | 2011-06-30 | 254 | 254 | 0 | 97.96% | 100.00% | 95.8% |
| **2011** | 2011-06-30 | 2012-06-29 | 253 | 253 | 0 | 97.64% | 100.00% | 95.6% |
| **2012** | 2012-07-02 | 2013-06-28 | 249 | 249 | 0 | 98.45% | 100.00% | 97.3% |
| **2013** | 2013-07-01 | 2014-06-30 | 252 | 252 | 0 | 98.67% | 100.00% | 97.7% |
| **2014** | 2014-06-30 | 2015-06-30 | 253 | 253 | 0 | 98.23% | 100.00% | 96.5% |
| **2015** | 2015-06-30 | 2016-06-30 | 254 | 254 | 0 | 96.65% | 100.00% | 94.2% |
| **2016** | 2016-06-30 | 2017-06-30 | 253 | 253 | 0 | 97.32% | 100.00% | 95.6% |
| **2017** | 2017-06-30 | 2018-06-29 | 252 | 252 | 0 | 97.57% | 100.00% | 95.4% |
| **2018** | 2018-07-02 | 2019-06-28 | 250 | 250 | 0 | 98.05% | 100.00% | 96.6% |
| **2019** | 2019-07-01 | 2020-06-30 | 253 | 253 | 0 | 98.26% | 100.00% | 96.6% |
| **2020** | 2020-06-30 | 2021-06-30 | 253 | 253 | 0 | 98.65% | 100.00% | 97.2% |
| **2021** | 2021-06-30 | 2022-06-30 | 253 | 253 | 0 | 97.75% | 100.00% | 95.7% |
| **2022** | 2022-06-30 | 2023-06-30 | 252 | 252 | 0 | 98.30% | 100.00% | 96.9% |
| **2023** | 2023-06-30 | 2024-06-28 | 251 | 251 | 0 | 99.20% | 100.00% | 97.4% |
| **2024** | 2024-06-30 | 2025-06-30 | 250 | 128 | 122 | 98.49% | 100.00% | 96.9% |
| **2025** | 2025-06-30 | 2026-06-30 | 255 | 0 | 255 | 96.72% | 98.43% | 96.5% |
| **2026** | 2026-06-30 | 2027-06-30 | 61 | 0 | 61 | 99.90% | 100.00% | 99.9% |

---

## Point-in-Time Model Lookback Eligibility ($T=40$ Trading Days)

### 1. Core Semantic Principle
Annual Russell membership files and annual/periodic market price files are **storage and ingestion boundaries, NOT model lookback boundaries**.

In financial machine learning, feature calculation requires a historical lookback window of $T=40$ trading sessions (roughly 8 calendar weeks). Naively truncating or resetting lookback history at each annual snapshot or annual price file boundary causes two catastrophic artifacts:
1. **Artificial Ramp-up Deficit**: Discarding the first 39 trading days of every reconstitution cycle artificially eliminates $15\% - 63\%$ of valid market observations.
2. **False Cutoff Drop in 2024 & 2026**: In partial or partitioned storage files (such as `2024.parquet` where WRDS daily data terminates at 2024-12-31, 128 trading days into the membership cycle), naive within-file counting falsely labels $(128 - 40) / 128 = 68.75\%$ eligible instead of the true $98.32\%$.

### 2. Disentangled Mathematical Components
To guarantee zero look-ahead bias and eliminate synthetic data artifacts, the eligibility of security $s$ on calendar session $t$ is factored into four strictly decoupled components:

$$\mathcal{E}(s, t) = \mathcal{M}(s, t) \land \mathcal{H}(s, t) \land \mathcal{T}(s, t) \land \mathcal{Y}(s, t)$$

1. **Membership State $\mathcal{M}(s, t)$**:
   The security is an active constituent of the Russell 1000 during the active usage period $[\text{start}, \text{end}]$.
   - `CONFIRMED_MEMBER` or `OFF_CYCLE_MEMBER`: $\mathcal{M}(s, t) = \text{True}$.
   - `DELISTED`: If $t > \text{delisting\_date}$, $\mathcal{M}(s, t) = \text{False}$.
2. **Historical Lookback Availability $\mathcal{H}(s, t)$**:
   Continuous security-level OHLCV history exists across annual file boundaries for all 40 trading sessions in the trailing calendar interval $[t - (40-1), \dots, t]$:
   $$\text{valid\_history\_count}(s, t) == 40$$
   - **Pre-membership history counts**: A security added to the index at annual reconstitution on date $t_0$ is immediately eligible if it actively traded over the preceding 39 market sessions outside the index.
   - **Genuinely newly listed securities (IPOs)**: Securities with fewer than 40 sessions since their initial public offering naturally fail $\mathcal{H}(s, t)$ until session 40. No forward-fill or synthetic data is permitted.
3. **Tradability $\mathcal{T}(s, t)$**:
   The security actively traded on day $t$ with non-zero volume/price ($|PRC| > 0$).
4. **Target Constructibility $\mathcal{Y}(s, t)$**:
   A valid forward target can be computed without look-ahead leakage. The forward return $r_{t+1}$ exists, or the security terminates at $t$ with an authentic CRSP delisting return `DLRET`. For the final session of the universal calendar ($t = t_{\max}$), $\mathcal{Y}(s, t) = \text{False}$.

### 3. Quantification of Boundary Exclusions & Recovery
Evaluating all 6,645,449 constituent-days across the 2000–2026 span reveals the dramatic impact of cross-boundary stitching:

| Metric | Naive Isolated Boundaries | Continuous Cross-Boundary Stitching | Net Recovery |
| :--- | :--- | :--- | :--- |
| **Total Constituent-Days** | 6,645,449 | 6,645,449 | — |
| **Eligible Observations** | 5,244,846 (78.92%) | 6,302,916 (94.85%) | **+1,058,070 (+15.92%)** |
| **2024 Cohort Eligibility** | 82.72% (WRDS file cutoff: 68.75%) | **98.32%** (248,264 / 252,500) | **+39,390 (+15.60%)** |
| **2026 Partial Cohort** | 30.51% (19,257 / 63,116) | **93.41%** (58,959 / 63,116) | **+39,702 (+62.90%)** |

### 4. Cohort-by-Cohort Eligibility Summary ($T=40$)

| List Year | Total Obs | Continuous Eligible | % Eligible | Fail $\mathcal{H}$ (History) | Fail $\mathcal{T}$ (Tradable) | Fail $\mathcal{Y}$ (Target) | Fail $\mathcal{M}$ (Delisted) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

### 5. Key Findings & Diagnostic Resolution
1. **Elimination of the 2024 Drop**: Under naive isolated processing, `2024.parquet` (128 WRDS trading days) dropped to 68.75% eligibility. With cross-boundary stitching into 2025 H1 crawled market data, 2024 achieves **98.32% eligibility**.
2. **Handling of CRSP DLSTCD=100**: In CRSP daily stock event records, `DLSTCD = 100` signifies 'active / still trading' at WRDS data cutoff (2024-12-31). Treating this code as a termination event erroneously marked 1,022 active constituents as delisted for 2025. Filtering out `DLSTCD == 100` preserves active membership for ongoing constituents.
3. **Exchange Holiday Filtering in Crawled Data**: Four 2026 US market holidays (MLK Day, Presidents Day, Memorial Day, Juneteenth) appeared in raw crawled data due to ASX dual-listed trading in `LNW`. Requiring exchange-wide market depth ($\ge 100$ securities) purges holiday contamination, preventing spurious window deficits.
4. **Point-in-Time Security ID Continuity**: Reused or rebranded tickers (e.g. AT&T `T`, which transitioned PERMNO 10401 $\rightarrow$ 66093 in 2005) are resolved year-by-year via `(year, TICKER) -> PERMNO` mappings, preventing cross-cohort identity collisions.

### 6. Artifacts and Audit Tables
- `report/quality/tables/model_eligibility_diagnostics.parquet` (6,645,449 constituent-day records with all 13 diagnostic fields: `permno`, `ticker`, `date`, `list_year`, `source_file_start`, `source_file_end`, `membership_state`, `valid_history_count`, `lookback_eligible`, `tradable_at_t`, `target_available`, `delisted_flag`, `model_eligible`).
- `report/quality/tables/model_eligibility_diagnostics_sample.csv` (representative sample rows covering reconstitution boundaries, delistings, and IPO ramps).
- `report/quality/tables/model_boundary_exclusion_comparison.parquet` & `.csv` (quantification of isolated vs continuous eligibility per cohort).
- `report/quality/tables/model_eligibility_summary.parquet` & `.csv` (breakdown of failure modes $\mathcal{H}, \mathcal{T}, \mathcal{Y}, \mathcal{M}$ per cohort).

---

## Recommended Point-in-Time (PiT) Data Model Architecture

```text
Russell List Layer
    ├── list_id: RUSSELL1000_{year}
    ├── source_snapshot_date: Exact document date (e.g. 2023-11-15)
    ├── source_file: Raw archive provenance (e.g. 2023.json)
    └── constituent_tickers: Raw constituent symbols
              │
              ▼
Membership Usage Period Layer
    ├── period_start: Reconstitution effective date (e.g. 2023-06-30)
    ├── period_end: Terminal evaluation date (e.g. 2024-06-28)
    └── membership_evidence: OBSERVED_SNAPSHOT vs USED_FOR_PERIOD
              │
              ▼
Security Identity Layer
    ├── primary_key: PERMNO (for WRDS) / Stable Linked ID (for Crawled)
    ├── time_varying_ticker: Active exchange ticker symbol
    └── delisting_status: Active vs Terminated (with DLRET)
              │
              ▼
Continuous Daily Price Store Layer
    ├── date: Continuous trading calendar (2000-06-30 to 2026-09-25, 6,602 dates)
    ├── security_id: PERMNO
    ├── OHLCV: Standardized unadjusted prices + cumulative split factor
    ├── data_source: WRDS_CRSP (<= 2024-12-31) vs CRAWLED (>= 2025-01-02)
    └── source_symbol: Original string identifier
              │
              ▼
Model Feature Eligibility Layer
    └── valid_40d_window: Rolling 40 trading days available across all boundaries
```

---

## Comprehensive Issue Synthesis & Action Matrix

| Issue | Severity | Affected Scope | Evidence | Must Resolve Before PiT? | Recommended Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2023 Off-Cycle Snapshot Timing** | HIGH | Year 2023 Period | Snapshot dated 2023-11-15; queried from 2023-06-30 | YES | Record 138-day hindsight gap; verify active listing status between June and Nov 2023. |
| **WRDS Cutoff & Source Boundary** | HIGH | 2024-12-31 / 2025-01-02 | WRDS ends 2024-12-31; Crawled begins 2025-01-02 | YES | Stitch 2024 WRDS with 2025 Crawled; apply split adjustments to 27 outlier names. |
| **Crawler Lacks PERMNO Identifier** | CRITICAL | Years 2025–2026 | Only string Ticker present in crawled data | YES | Build PERMNO cross-walk mapping using last known active WRDS PERMNO on 2024-12-31. |
| **Ticker 'NA' Default Missing Trap** | CRITICAL | Year 2000 (Nabisco) | Row 605 in all_years.csv: Parsed as NaN | YES | Enforce keep_default_na=False across all pipeline readers. |
| **Ticker Reuse Across Multiple PERMNOs** | CRITICAL | 20 historical tickers | WRDS PERMNO count > 1 for single TICKER | YES | Disallow joining historical series by string ticker alone; bind all series by PERMNO. |
| **Delisted Securities Retention** | CRITICAL | 1,901 WRDS delistings | Explicit DLSTCD records in WRDS | YES | Retain delisted securities up to exit date with DLRET to prevent survivorship bias. |
| **Cross-Year Lookback Continuity** | HIGH | Days 0–39 post-reconstitution | Deficit in isolated annual files (15.92% falsely excluded) | YES | Stitch multi-year continuous price store; eliminates 40-day ramp-up deficit, recovering 1,058,070 observations (94.85% overall, 2024 at 98.32%). |
| **CRSP Negative Price Conventions** | INFO | Zero-volume days | Negative PRC in CRSP | NO | Take abs(PRC) as bid/ask midpoint quote. |

---

## Reproducibility & Acceptance Test

```text
Audit Provenance & Verification
------------------------------
Git Commit Hash:          616e36a6a0341b86064d9cd36fab511e223dd9ec
Python Version:           3.11.7
Config File:              data/../config/audit.yaml
Input Manifest:           1.0.0 (85 verified input files)
Random Seed:              42
Execution Run 1 vs Run 2: IDENTICAL
Identical Table Hashes:   YES
Non-Deterministic Diffs:  NONE
```