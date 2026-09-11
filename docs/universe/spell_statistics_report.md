# Empirical Statistical Characterization of `data/universe/spells.csv`
## Comprehensive Analysis and Methodological Brief for Supervisor Review

**Target Dataset**: `data/universe/spells.csv`  
**Evaluation Scope**: US Equity Point-in-Time Universe Reconstruction (2004–2026)  
**Task Nature**: Pure Statistical Analysis and Methodological Audit (No Data Modification)  
**Date of Audit**: September 2026  
**Auditor**: Quant Research & Universe Engineering Team  

---

## Executive Abstract: Core Supervisor Findings

This report delivers a rigorous, empirical analysis of the historical ticker-spell dataset (`data/universe/spells.csv`) to determine the statistical scale of the **disappearance, long-gap, and ticker-reuse problem**, directly informing how much historical identity-resolution effort is methodologically justified.

> **Methodological Invariant**: These spells represent **point-in-time ticker availability intervals** derived from daily historical snapshots of actively traded symbols in the US equity universe, **not** guaranteed economically continuous corporate entity lifetimes.

### The Supervisor's Primary Question Answered:
> **"How many ticker cases have long gaps between active periods, and is ticker-reuse/long-gap handling a very common problem or a relatively small number of exceptional cases?"**

The empirical answer is nuanced and bifurcated depending on the analytical lens:

1. **Relative to the Multi-Spell Population (Local Frequency)**:
   - Out of 36,843 total tickers, **5,555 tickers (15.08%)** experience disappearance and subsequent reappearance (generating **6,914 inter-spell gaps**).
   - Among multi-spell tickers, long gaps are **very common**:
     - **3,416 gap events ($\ge 252$ sessions / 1 year)** affect **3,108 unique tickers** (**55.95%** of multi-spell tickers).
     - **1,970 gap events ($\ge 1,000$ sessions / ~4 years)** affect **1,924 unique tickers** (**34.64%** of multi-spell tickers).
   - Thus, among multi-spell tickers, disappearing for years is **the modal behavior**, not an exceptional edge case.

2. **Relative to the Total Historical Universe (Global Impact)**:
   - Long gaps $\ge 252$ sessions affect only **8.44% of all unique tickers** (3,108 / 36,843).
   - Even more critically, from an active observation perspective, excluding tickers with gaps $\ge 252$ sessions would discard **13.19% of historical active ticker-day observations** under full exclusion, or only **7.14%** if only post-gap re-entries are dropped.
   - For an extreme threshold ($\ge 1,000$ sessions), only **5.22% of tickers** (1,924 tickers) and **3.64% of post-gap observations** are affected.

3. **Short Gaps vs. Source Ingestion Dropouts**:
   - Short gaps ($\le 2$ sessions) account for **834 gap events (12.06%)** across **660 tickers**.
   - Cross-referencing existing identity records confirms that **91.37% of these short gaps represent the identical security** (e.g., `CMCSA` dropping out for 1 day across 44 spells due to transient snapshot ingestion glitches).
   - A simple continuity heuristic ($\le 2$ sessions) effectively resolves short data dropouts without risking identity contamination.

---

## 1. Dataset Scope & Schema Integrity

### 1.1 Dataset Summary Metrics
- **Canonical Input File**: `data/universe/spells.csv`
- **Total Spell Records**: **43,757 rows**
- **Total Unique Tickers**: **36,843 symbols**
- **Date Range**: `2004-01-02` to `2026-09-01`
- **Clean NYSE Calendar Sessions**: **5,699 sessions** (out of 5,702 raw snapshot dates; 3 corrupted snapshot dates excluded from timeline)
- **Total Active Ticker-Session Observations**: **51,387,449 ticker-days**
- **Dataset Invariants Verified**:
  - $	ext{start\_date} \le 	ext{end\_date}$: **100% verified (0 violations)**
  - $	ext{duration\_sessions} > 0$: **100% verified (0 violations)**
  - $	ext{gap\_sessions} \ge 0$: **100% verified (0 violations)**
  - Duplicate `(ticker, spell_seq)` keys: **0 duplicate keys**
  - Underlying file SHA-256 hash: Verified bit-for-bit unchanged before and after execution.

### 1.2 Canonical Schema Definition
| Column | Type | Nullable | Semantic Description |
| :--- | :---: | :---: | :--- |
| `ticker` | `String` | No | Historical market equity symbol (as observed in snapshot) |
| `spell_seq` | `Int64` | No | 1-based sequential index of continuous observation for ticker |
| `n_spells_total` | `Int64` | No | Total number of discrete availability spells for this ticker |
| `start_date` | `String` | No | First observed trading session date (`YYYY-MM-DD`) |
| `end_date` | `String` | No | Last observed trading session date (`YYYY-MM-DD`) |
| `n_sessions` | `Int64` | No | Number of active trading sessions observed within the spell |
| `gap_after_sessions` | `Int64` | Yes | Missing trading sessions between `end_date` and next spell's `start_date` (null for final spell) |
| `gap_after_at_break` | `Boolean` | No | `True` if subsequent spell resumes on a known feed break resumption date |

---

## 2. Primary Supervisor Summary: Short vs. Long Gap Distribution

To enable immediate supervisor review of the scale of the problem, the complete inter-spell gap population (**6,914 gap events** across **5,555 multi-spell tickers**) is categorized into critical decision thresholds:

| Gap Threshold | Gap Events | % of All Gaps | Unique Tickers | % of Multi-Spell Tickers (N=5,555) | % of Total Universe (N=36,843) | Operational Interpretation |
| :--- | :---:|---:|---:|---:|---:| :--- |
| **$\le 2$ sessions** | **834** | **12.06%** | **660** | **11.88%** | 1.79% | Probable source snapshot dropouts; candidates for simple continuity bridging |
| **$> 2$ sessions** | **6,080** | **87.94%** | **5,065** | **91.18%** | 13.75% | Non-trivial disappearances exceeding weekend/holiday or short feed glithes |
| **$> 10$ sessions** | **5,794** | **83.80%** | **4,884** | **87.92%** | 13.26% | Prolonged absence (> 2 calendar weeks); unlikely to be transient ingestion drops |
| **$\ge 20$ sessions** | **5,682** | **82.18%** | **4,800** | **86.41%** | 13.03% | Monthly absence; high likelihood of regulatory suspension or identity transition |
| **$\ge 60$ sessions** | **4,824** | **69.77%** | **4,159** | **74.87%** | 11.29% | Quarterly absence; standard corporate restructurings, bankruptcies, or ticker reallocations |
| **$\ge 252$ sessions** | **3,416** | **49.41%** | **3,108** | **55.95%** | **8.44%** | **Annual absence ($\ge 1$ full trading year)**; primary threshold for potential ticker reuse |
| **$\ge 1,000$ sessions** | **1,970** | **28.49%** | **1,924** | **34.64%** | **5.22%** | **Multi-year absence ($\ge 4$ trading years)**; extreme ticker reuse candidates (e.g., `ACMR`) |

> [!IMPORTANT]
> **Key Finding**: Half of all inter-spell gaps (**49.41%**) span **at least 1 full trading year (252 sessions)**. Furthermore, over one-quarter of all gaps (**28.49%**) span **more than 4 trading years (1,000 sessions)**. Gaps in this dataset are predominantly **long-term corporate separations**, not short recording dropouts.

---

## 3. Complete Binned and Cumulative Gap Distribution

### 3.1 Granular Gap Bins
| Bin Range (Sessions) | Gap Events | Share of Gaps | Unique Tickers | Share of Multi-Spell Tickers | Typical Calendar Interpretation |
| :--- | ---:|---:| ---:|---:| :--- |
| **0–1 sessions** | 719 | 10.40% | 598 | 10.77% | Single missed session (1-day feed dropout) |
| **2 sessions** | 115 | 1.66% | 73 | 1.31% | Two consecutive missed sessions |
| **3–5 sessions** | 163 | 2.36% | 126 | 2.27% | Up to 1 calendar week |
| **6–10 sessions** | 123 | 1.78% | 115 | 2.07% | 1 to 2 calendar weeks |
| **11–19 sessions** | 112 | 1.62% | 108 | 1.94% | 2 to 4 calendar weeks |
| **20–29 sessions** | 432 | 6.25% | 406 | 7.31% | ~1 calendar month |
| **30–59 sessions** | 426 | 6.16% | 420 | 7.56% | 1 to 3 calendar months |
| **60–119 sessions** | 302 | 4.37% | 297 | 5.35% | ~1 calendar quarter |
| **120–251 sessions** | 1,106 | 16.00% | 1,053 | 18.96% | Semi-annual absence (~6–12 months) |
| **252–499 sessions** | 587 | 8.49% | 562 | 10.12% | 1 to 2 calendar years |
| **500–999 sessions** | 859 | 12.42% | 827 | 14.89% | 2 to 4 calendar years |
| **1000+ sessions** | 1,970 | 28.49% | 1,924 | 34.64% | 4 to 22 calendar years |
| **Total Gaps** | **6,914** | **100.00%** | — | — | — |

### 3.2 Cumulative Thresholds ($\ge X$ Sessions)
| Cumulative Threshold | Gap Count | % of All Gaps | Unique Tickers | % of Multi-Spell Tickers |
| :--- | ---:|---:| ---:|---:|
| **$\ge 2$ sessions** | 6,195 | 89.60% | 5,108 | 91.95% |
| **$\ge 5$ sessions** | 5,953 | 86.10% | 4,994 | 89.90% |
| **$\ge 10$ sessions** | 5,820 | 84.18% | 4,903 | 88.26% |
| **$\ge 20$ sessions** | 5,682 | 82.18% | 4,800 | 86.41% |
| **$\ge 30$ sessions** | 5,250 | 75.93% | 4,491 | 80.85% |
| **$\ge 60$ sessions** | 4,824 | 69.77% | 4,159 | 74.87% |
| **$\ge 120$ sessions** | 4,522 | 65.40% | 3,955 | 71.20% |
| **$\ge 252$ sessions** | 3,416 | 49.41% | 3,108 | 55.95% |
| **$\ge 500$ sessions** | 2,829 | 40.92% | 2,661 | 47.90% |
| **$\ge 1,000$ sessions** | 1,970 | 28.49% | 1,924 | 34.64% |

---

## 4. Multi-Spell Ticker Segmentation & Concentration

### 4.1 Population Segmentation
- **Total Unique Tickers in Universe**: **36,843**
- **Single-Spell Tickers (No Gaps)**: **31,288 (84.92%)**
- **Multi-Spell Tickers ($\ge 2$ Spells)**: **5,555 (15.08%)**
- **Highly Fragmented Tickers ($\ge 3$ Spells)**: **967 (2.62%)**
- **Extreme Fragmentation ($\ge 5$ Spells)**: **45 (0.12%)**
- **Maximum Spells for a Single Symbol**: **46 spells** (`CMCS.A`), followed by **44 spells** (`CMCSA`), **27 spells** (`LINT.A`), **26 spells** (`LINTA`), and **15 spells** (`DISCA`).

### 4.2 Distribution of Spells per Ticker
| Spells per Ticker | Ticker Count | Share of Tickers | Cumulative Tickers | Cumulative Share | Semantic Category |
| :---: | ---:|---:| ---:|---:| :--- |
| **1 spell** | 31,288 | 84.92% | 31,288 | 84.92% | Continuous observation (Unproblematic) |
| **2 spells** | 4,588 | 12.45% | 35,876 | 97.38% | Single historical gap (Standard transition / reuse) |
| **3 spells** | 795 | 2.16% | 36,671 | 99.53% | Two historical gaps |
| **4 spells** | 127 | 0.34% | 36,798 | 99.88% | Three historical gaps |
| **5+ spells** | 45 | 0.12% | 36,843 | 100.00% | Highly fragmented (Mostly known data feed dropouts) |

---

## 5. Statistical Parametrics of Gaps and Spell Durations

### 5.1 Gap Duration Parametrics (Trading Sessions)
| Metric | All Inter-Spell Gaps (N=6,914) | Long Gaps $\ge 252$ Sessions (N=3,416) | Very Long Gaps $\ge 1,000$ Sessions (N=1,970) |
| :--- | ---:| ---:| ---:|
| **Minimum** | 1 session | 252 sessions | 1,002 sessions |
| **25th Percentile (P25)** | 37.0 sessions (~1.8 mos) | 647.0 sessions (~2.6 yrs) | 1,498.3 sessions (~5.9 yrs) |
| **Median** | **238.0 sessions (~0.95 yrs)** | **1,248.5 sessions (~5.0 yrs)** | **2,097.5 sessions (~8.3 yrs)** |
| **Mean** | 840.4 sessions (~3.3 yrs) | 1,631.8 sessions (~6.5 yrs) | 2,391.3 sessions (~9.5 yrs) |
| **75th Percentile (P75)** | 1,219.0 sessions (~4.8 yrs) | 2,342.3 sessions (~9.3 yrs) | 3,113.0 sessions (~12.4 yrs) |
| **90th Percentile (P90)** | 2,629.1 sessions (~10.4 yrs) | 3,561.5 sessions (~14.1 yrs) | 4,018.1 sessions (~15.9 yrs) |
| **95th Percentile (P95)** | 3,556.3 sessions (~14.1 yrs) | 4,104.8 sessions (~16.3 yrs) | 4,496.1 sessions (~17.8 yrs) |
| **99th Percentile (P99)** | 4,672.0 sessions (~18.5 yrs) | 4,963.9 sessions (~19.7 yrs) | 5,160.3 sessions (~20.5 yrs) |
| **Maximum** | 5,620.0 sessions (~22.3 yrs) | 5,620.0 sessions (~22.3 yrs) | 5,620.0 sessions (~22.3 yrs) |

> [!NOTE]
> The median gap among all multi-spell events is **238 trading sessions (~1 calendar year)**, and for the subset exceeding 252 sessions, the median gap is **1,248.5 sessions (~5 calendar years)**. This proves that long gaps are not marginal over-the-threshold occurrences, but represent multi-year historical absences.

### 5.2 Spell Duration Parametrics
| Metric | Spell Duration (Trading Sessions) | Approximate Calendar Duration |
| :--- | ---:| ---:|
| **Minimum** | 1 session | 1 day |
| **25th Percentile (P25)** | 214.0 sessions | 0.85 years (~10 months) |
| **Median** | **624.0 sessions** | **2.48 years** |
| **Mean** | 1,174.38 sessions | 4.66 years |
| **75th Percentile (P75)** | 1,526.0 sessions | 6.06 years |
| **90th Percentile (P90)** | 3,272.0 sessions | 12.98 years |
| **95th Percentile (P95)** | 4,608.4 sessions | 18.29 years |
| **99th Percentile (P99)** | 5,699.0 sessions | 22.62 years (Full Span) |
| **Maximum** | 5,699.0 sessions | 22.62 years (Full Span) |

### 5.3 Short-Lived Spell Proportions
- **Single-session spells ($n=1$)**: **738 spells (1.69%)** across **668 tickers**
- **Spells $\le 5$ sessions ($\le 1$ week)**: **1,885 spells (4.31%)** across **1,618 tickers**
- **Spells $\le 20$ sessions ($\le 1$ month)**: **4,576 spells (10.46%)** across **4,022 tickers**
- **Spells $\le 50$ sessions ($\le 1$ quarter)**: **6,455 spells (14.75%)** across **5,802 tickers**

---

## 6. Counterfactual Impact of Possible Exclusion Rules

A central question for the supervisor is:
> **"What would happen if we decided that cases with very long gaps are too expensive to resolve and can simply be excluded from the universe?"**

To answer this objectively, we evaluate two counterfactual exclusion policies across candidate thresholds:
- **Policy A (Drop Entire Ticker)**: If a ticker has *any* gap meeting or exceeding the threshold, exclude all spells and active sessions for that symbol from the universe.
- **Policy B (Truncate at Gap / Drop Post-Gap Spells)**: Retain the initial continuous spell; discard only subsequent spells that resume after the qualifying long gap.

| Gap Threshold | Exclusion Policy | Affected Tickers | % of All Tickers (N=36,843) | Discarded Spells | % of All Spells (N=43,757) | Discarded Ticker-Sessions | % of Historical Universe (N=51.39M) |
| :--- | :--- | ---:|---:| ---:|---:| ---:|---:|
| **$> 10$ sessions** | **Policy A (Drop All)** | 4,884 | 13.26% | 11,045 | 25.24% | 11,046,585 | **21.50%** |
| | **Policy B (Post-Gap Only)** | 4,884 | 13.26% | 6,032 | 13.79% | 7,068,582 | **13.76%** |
| **$\ge 20$ sessions** | **Policy A (Drop All)** | 4,800 | 13.03% | 10,823 | 24.73% | 10,901,901 | **21.22%** |
| | **Policy B (Post-Gap Only)** | 4,800 | 13.03% | 5,907 | 13.50% | 6,949,345 | **13.52%** |
| **$\ge 60$ sessions** | **Policy A (Drop All)** | 4,159 | 11.29% | 9,499 | 21.71% | 9,860,808 | **19.19%** |
| | **Policy B (Post-Gap Only)** | 4,159 | 11.29% | 5,046 | 11.53% | 6,089,147 | **11.85%** |
| **$\ge 252$ sessions** | **Policy A (Drop All)** | 3,108 | 8.44% | 7,244 | 16.56% | 6,777,864 | **13.19%** |
| | **Policy B (Post-Gap Only)** | 3,108 | 8.44% | 3,672 | 8.39% | 3,669,058 | **7.14%** |
| **$\ge 1,000$ sessions** | **Policy A (Drop All)** | 1,924 | 5.22% | 4,450 | 10.17% | 3,867,178 | **7.53%** |
| | **Policy B (Post-Gap Only)** | 1,924 | 5.22% | 2,148 | 4.91% | 1,872,081 | **3.64%** |

### Critical Analytical Insight:
- Dropping tickers with gaps $>10$ or $\ge 20$ sessions under Policy A eliminates **over 21% of the entire historical research universe** (~11 million observation days). This would introduce severe selection bias.
- In contrast, under **Policy B at $\ge 252$ sessions**, only **7.14% of historical observations** are discarded.
- At an extreme threshold of **$\ge 1,000$ sessions (~4 years)**, only **3.64% of observations** (and 1,924 tickers) are affected under Policy B.

---

## 7. Spell-Only vs. Identity-Informed Perspective

To give the supervisor complete clarity, we strictly distinguish **raw spell observations** from **external identity evidence**:

```
Raw Spell Observations (spells.csv)
├── Single-Spell Tickers (N = 31,288 / 84.92%) ─── No Gaps (Identity Continuity Trivial)
└── Multi-Spell Tickers (N = 5,555 / 15.08%) ───── 6,914 Gaps Total
     │
     ├── Confirmed Same Security (N = 2,257 / 40.63%)
     │   └── Examples: CMCSA (44 spells, 43 short gaps), DISCA, LINTA
     │   └── Short Gaps (<= 2 sessions): 91.37% confirmed identical share_class_figi
     │
     └── Multi-Security / Unresolved Buckets (N = 3,298 / 59.37%)
         ├── Confirmed Distinct Security Ticker-Reuse (N = 1 flagship: ACMR)
         │   └── ACMR Spell 1 (2004–2011, A.C. Moore) ≠ ACMR Spell 2 (2017–2026, ACM Research)
         └── Provisional Unresolved Buckets (N = 3,297 tickers)
             └── Delisted OTC/warrants/pre-2010 tickers kept isolated to prevent contamination
```

### Gap Length vs. Security Identity Continuity
| Gap Category | Total Gaps | Confirmed Same Security | Same Security Share | Unresolved / Conflicting Identities |
| :--- | ---:| ---:| ---:| ---:|
| **Short Gaps ($\le 2$ sessions)** | 834 | 762 | **91.37%** | 72 (8.63%) |
| **Gaps 3–251 sessions** | 2,664 | 1,227 | **46.06%** | 1,437 (53.94%) |
| **Long Gaps ($\ge 252$ sessions)** | 3,416 | 268 | **7.85%** | **3,148 (92.15%)** |
| **Extreme Gaps ($\ge 1,000$ sessions)** | 1,970 | 97 | **4.92%** | **1,873 (95.08%)** |

> [!TIP]
> **Definitive Correlation**: Short gaps ($\le 2$ sessions) are overwhelming evidence of **identical security continuity (91.4%)**. Conversely, gaps $\ge 252$ sessions are overwhelming evidence of **identity discontinuity or external registry absence (92.2%)**. This justifies a bifurcated pipeline: automatic bridging for $\le 2$ sessions, and strict isolation / manual review for $\ge 252$ sessions.

---

## 8. Left and Right Boundary Effects (Censoring)

A crucial consideration for empirical research is distinguishing dataset truncation from actual economic listing/delisting:

| Boundary Condition | Date | Spell Count | Unique Tickers | Share of Tickers | Methodological Interpretation |
| :--- | :---: | ---:| ---:| ---:| :--- |
| **Left-Censored** | `2004-01-02` | 8,164 | 8,164 | 22.16% | Active on initial dataset date; listing date preceded 2004 |
| **Right-Censored** | `2026-09-01` | 13,149 | 13,149 | 35.69% | Active on final observation date; still trading / not delisted |
| **Both Censored (Full Span)** | Both | 1,378 | 1,378 | 3.74% | Continuously active for all 5,699 sessions without interruption |
| **Touching Either Boundary** | Either | 19,935 | 18,557 | 50.37% | Spells touching at least one truncation boundary |

> **Methodological Rule**: An observed `start_date = 2004-01-02` must **never** be interpreted as an IPO/listing event, and `end_date = 2026-09-01` must **never** be interpreted as a delisting/liquidation event.

---

## 9. Anomalous Snapshot Dates and Break Resumptions

### 9.1 The Three Corrupted Raw Snapshot Dates
Historical inspection of raw source snapshots identified 3 dates with catastrophic symbol loss:
- `2009-10-29`: Observed count = **5,587** (Drop of **2,268 symbols / 28.9%** from baseline ~7,855)
- `2010-03-30`: Observed count = **7,012** (Drop of **808 symbols / 10.3%** from baseline ~7,820)
- `2010-03-31`: Observed count = **6,712** (Drop of **1,108 symbols / 14.2%** from baseline ~7,820)

**Impact on Spell Reconstruction**: These 3 dates were excluded from the clean 5,699 session timeline. Had they been naively included, they would have artificially fractured over **4,000 continuous equity spells** into artificial 1-day dropouts.

### 9.2 Feed Break Resumption Dates
- **28 known historical break resumption dates** were identified in the source feed.
- **859 inter-spell gaps (12.42% of all gaps)** terminate on one of these break resumption dates (`gap_after_at_break = True`).
- This confirms that a substantial cluster of gaps represents historical feed maintenance windows rather than individual corporate disappearances.

---

## 10. Illustrative Empirical Case Studies

The following concrete examples illustrate the spectrum of inter-spell gap behaviors:

### Example A: Short Gap ($\le 2$ sessions) — Source Dropout Candidate
- **Ticker**: `CMCSA` (Comcast Corporation Class A)
- **Spell 1**: `2004-01-02` $ightarrow$ `2014-06-18` (Duration: **2,630 sessions**, ~10.4 years)
- **Gap**: `2014-06-19` $ightarrow$ `2014-06-19` (**1 trading session**, 1 calendar day)
- **Spell 2**: `2014-06-20` $ightarrow$ `2014-06-23` (Duration: **2 sessions**)
- **Empirical Fact**: Ticker `CMCSA` was actively observed for 2,630 sessions, was absent from the snapshot on June 19, 2014, and reappeared on June 20, 2014. External records confirm active NASDAQ trading on June 19 with millions of shares traded. This is an ingestion dropout, not an economic event.

### Example B: Medium Gap (20–59 sessions) — Temporary Disappearance
- **Ticker**: `AAUK` (Anglo American plc ADR)
- **Spell 1**: `2004-01-02` $ightarrow$ `2007-07-24` (Duration: **893 sessions**, ~3.5 years)
- **Gap**: `2007-07-25` $ightarrow$ `2007-08-21` (**20 trading sessions**, 27 calendar days)
- **Spell 2**: `2007-08-22` $ightarrow$ `2009-07-31` (Duration: **490 sessions**, ~2.0 years)
- **Empirical Fact**: Ticker `AAUK` disappeared from snapshots for exactly 20 trading sessions in summer 2007 before reappearing for another 2 years.

### Example C: Long Gap (60–251 sessions) — Extended Absence
- **Ticker**: `AAAP` (Advanced Accelerator Applications S.A.)
- **Spell 1**: `2015-02-04` $ightarrow$ `2015-02-05` (Duration: **2 sessions**)
- **Gap**: `2015-02-06` $ightarrow$ `2015-11-10` (**193 trading sessions**, 277 calendar days)
- **Spell 2**: `2015-11-11` $ightarrow$ `2018-02-09` (Duration: **566 sessions**, ~2.2 years)
- **Empirical Fact**: Ticker `AAAP` appeared briefly for 2 days, disappeared for 193 trading sessions (~9 calendar months), then traded continuously until 2018 buyout.

### Example D: Very Long Gap (252–999 sessions) — Multi-Year Inactivity
- **Ticker**: `AAC` (AAC Holdings, Inc. / Multiple Entities)
- **Spell 2**: `2010-12-14` $ightarrow$ `2012-10-15` (Duration: **464 sessions**)
- **Gap**: `2012-10-16` $ightarrow$ `2014-10-01` (**492 trading sessions**, 715 calendar days, ~2 years)
- **Spell 3**: `2014-10-02` $ightarrow$ `2019-10-25` (Duration: **1,276 sessions**)
- **Empirical Fact**: Ticker `AAC` disappeared for almost two full calendar years between October 2012 and October 2014 before reappearing.

### Example E: Extremely Long Gap ($\ge 1,000$ sessions) — Classic Ticker Reuse
- **Ticker**: `ACMR`
- **Spell 1**: `2004-01-02` $ightarrow$ `2011-11-18` (Duration: **1,984 sessions**, ~7.9 years)
- **Gap**: `2011-11-21` $ightarrow$ `2017-11-02` (**1,499 trading sessions**, 2,175 calendar days, **~6 calendar years**)
- **Spell 2**: `2017-11-03` $ightarrow$ `2026-09-01` (Duration: **2,217 sessions**, ~8.8 years)
- **Empirical Fact**: Ticker `ACMR` disappeared for 1,499 trading sessions between November 2011 and November 2017. External SEC records confirm Spell 1 was A.C. Moore Arts & Crafts (delisted 2011), while Spell 2 is ACM Research, Inc. (IPO 2017). They are completely unrelated companies sharing the same symbol across time.

---

## 11. Top 50 Longest Inter-Spell Gap Cases in the Dataset

The following table reports the 50 largest inter-spell gaps in the historical universe:

| Rank | Ticker | Prev Spell Dates | Next Spell Dates | Gap Sessions | Gap Cal Days | Prev Dur | Next Dur | Total Spells |
| :---: | :--- | :---: | :---: | ---:| ---:| ---:| ---:| :---: |
| 1 | `HXF` | 2004-01-02 $\rightarrow$ 2004-04-23 | 2026-09-01 $\rightarrow$ 2026-09-01 | 5,620 | 8,165 | 78 | 1 | 2 |
| 2 | `HXE` | 2004-01-02 $\rightarrow$ 2004-04-23 | 2026-08-21 $\rightarrow$ 2026-09-01 | 5,613 | 8,154 | 78 | 8 | 2 |
| 3 | `AESP` | 2004-01-02 $\rightarrow$ 2004-08-19 | 2026-07-01 $\rightarrow$ 2026-09-01 | 5,496 | 7,985 | 159 | 44 | 2 |
| 4 | `LYNX` | 2004-01-02 $\rightarrow$ 2005-03-02 | 2026-08-18 $\rightarrow$ 2026-09-01 | 5,395 | 7,838 | 293 | 11 | 2 |
| 5 | `VRC` | 2004-01-02 $\rightarrow$ 2005-03-11 | 2026-07-14 $\rightarrow$ 2026-09-01 | 5,363 | 7,794 | 300 | 36 | 2 |
| 6 | `CSCS` | 2004-01-02 $\rightarrow$ 2004-03-01 | 2025-06-25 $\rightarrow$ 2026-09-01 | 5,360 | 7,785 | 40 | 299 | 2 |
| 7 | `BETA` | 2004-06-25 $\rightarrow$ 2004-07-15 | 2025-11-03 $\rightarrow$ 2026-09-01 | 5,357 | 7,780 | 14 | 208 | 3 |
| 8 | `METG` | 2004-01-02 $\rightarrow$ 2005-03-31 | 2026-07-07 $\rightarrow$ 2026-09-01 | 5,345 | 7,767 | 313 | 41 | 2 |
| 9 | `BULL` | 2004-01-02 $\rightarrow$ 2004-01-14 | 2025-04-11 $\rightarrow$ 2026-09-01 | 5,341 | 7,757 | 9 | 349 | 2 |
| 10 | `CRAN` | 2004-04-20 $\rightarrow$ 2004-10-29 | 2026-01-12 $\rightarrow$ 2026-09-01 | 5,329 | 7,744 | 135 | 161 | 3 |
| 11 | `TSIC` | 2004-01-02 $\rightarrow$ 2004-12-23 | 2025-12-30 $\rightarrow$ 2026-09-01 | 5,283 | 7,676 | 247 | 169 | 2 |
| 12 | `FMKT` | 2004-01-02 $\rightarrow$ 2004-06-30 | 2025-06-10 $\rightarrow$ 2026-09-01 | 5,266 | 7,649 | 124 | 309 | 2 |
| 13 | `AGNT` | 2004-01-02 $\rightarrow$ 2005-07-07 | 2026-05-08 $\rightarrow$ 2026-09-01 | 5,238 | 7,609 | 381 | 80 | 2 |
| 14 | `CMRC` | 2004-01-02 $\rightarrow$ 2004-10-11 | 2025-08-01 $\rightarrow$ 2026-09-01 | 5,231 | 7,598 | 195 | 273 | 2 |
| 15 | `SPCH` | 2004-01-02 $\rightarrow$ 2005-09-20 | 2026-06-15 $\rightarrow$ 2026-09-01 | 5,211 | 7,572 | 433 | 55 | 2 |
| 16 | `WSBK` | 2004-01-02 $\rightarrow$ 2004-09-01 | 2025-05-02 $\rightarrow$ 2026-09-01 | 5,196 | 7,547 | 168 | 335 | 2 |
| 17 | `FTG` | 2004-01-02 $\rightarrow$ 2005-12-29 | 2026-08-18 $\rightarrow$ 2026-09-01 | 5,185 | 7,536 | 503 | 11 | 2 |
| 18 | `GTM` | 2004-01-02 $\rightarrow$ 2004-09-30 | 2025-05-13 $\rightarrow$ 2026-09-01 | 5,183 | 7,529 | 188 | 328 | 2 |
| 19 | `TRNI` | 2004-01-02 $\rightarrow$ 2005-12-14 | 2026-07-27 $\rightarrow$ 2026-09-01 | 5,179 | 7,529 | 493 | 27 | 2 |
| 20 | `LGLrw` | 2005-11-16 $\rightarrow$ 2005-11-17 | 2026-06-03 $\rightarrow$ 2026-06-05 | 5,161 | 7,502 | 2 | 3 | 2 |
| 21 | `FFF` | 2004-01-02 $\rightarrow$ 2005-06-10 | 2025-12-18 $\rightarrow$ 2026-09-01 | 5,160 | 7,495 | 363 | 176 | 2 |
| 22 | `FPWR` | 2004-01-02 $\rightarrow$ 2005-04-13 | 2025-09-03 $\rightarrow$ 2026-09-01 | 5,126 | 7,447 | 322 | 251 | 2 |
| 23 | `TMAR` | 2004-01-02 $\rightarrow$ 2004-12-16 | 2025-03-24 $\rightarrow$ 2026-09-01 | 5,094 | 7,402 | 242 | 363 | 2 |
| 24 | `PENG` | 2004-01-02 $\rightarrow$ 2004-07-28 | 2024-10-15 $\rightarrow$ 2026-09-01 | 5,085 | 7,383 | 143 | 471 | 2 |
| 25 | `AGCC` | 2004-01-02 $\rightarrow$ 2005-08-09 | 2025-10-22 $\rightarrow$ 2026-09-01 | 5,079 | 7,378 | 404 | 216 | 2 |
| 26 | `MHX` | 2004-01-02 $\rightarrow$ 2006-05-02 | 2026-07-14 $\rightarrow$ 2026-09-01 | 5,076 | 7,377 | 587 | 36 | 2 |
| 27 | `AIX` | 2004-01-02 $\rightarrow$ 2006-04-17 | 2026-06-25 $\rightarrow$ 2026-09-01 | 5,075 | 7,373 | 576 | 48 | 2 |
| 28 | `REMC` | 2005-06-21 $\rightarrow$ 2005-10-12 | 2025-12-11 $\rightarrow$ 2026-09-01 | 5,069 | 7,364 | 80 | 181 | 3 |
| 29 | `BMM` | 2004-09-16 $\rightarrow$ 2006-02-09 | 2026-01-26 $\rightarrow$ 2026-09-01 | 5,016 | 7,290 | 354 | 152 | 2 |
| 30 | `AGH` | 2004-01-02 $\rightarrow$ 2005-03-08 | 2025-02-12 $\rightarrow$ 2026-05-15 | 5,012 | 7,280 | 297 | 316 | 2 |
| 31 | `LSCP` | 2004-01-02 $\rightarrow$ 2006-07-25 | 2026-06-24 $\rightarrow$ 2026-09-01 | 5,005 | 7,273 | 645 | 49 | 2 |
| 32 | `GBND` | 2004-01-02 $\rightarrow$ 2005-08-16 | 2025-06-26 $\rightarrow$ 2026-09-01 | 4,992 | 7,253 | 409 | 298 | 2 |
| 33 | `CATG` | 2004-01-02 $\rightarrow$ 2006-07-27 | 2026-05-12 $\rightarrow$ 2026-09-01 | 4,974 | 7,228 | 647 | 78 | 2 |
| 34 | `AACB` | 2004-01-02 $\rightarrow$ 2005-07-07 | 2025-04-07 $\rightarrow$ 2026-08-19 | 4,965 | 7,213 | 381 | 344 | 2 |
| 35 | `GDT` | 2004-01-02 $\rightarrow$ 2006-04-21 | 2026-01-22 $\rightarrow$ 2026-09-01 | 4,965 | 7,215 | 580 | 154 | 2 |
| 36 | `HCOW` | 2004-01-02 $\rightarrow$ 2004-01-02 | 2023-09-20 $\rightarrow$ 2026-09-01 | 4,958 | 7,200 | 1 | 740 | 2 |
| 37 | `SAX` | 2005-03-18 $\rightarrow$ 2006-12-04 | 2026-08-27 $\rightarrow$ 2026-09-01 | 4,958 | 7,205 | 433 | 4 | 2 |
| 38 | `DVS` | 2004-01-02 $\rightarrow$ 2005-08-05 | 2025-04-21 $\rightarrow$ 2026-03-26 | 4,953 | 7,198 | 402 | 235 | 2 |
| 39 | `TP` | 2004-01-02 $\rightarrow$ 2006-12-01 | 2026-08-07 $\rightarrow$ 2026-09-01 | 4,945 | 7,188 | 736 | 18 | 2 |
| 40 | `BVC` | 2004-01-02 $\rightarrow$ 2006-04-28 | 2025-12-24 $\rightarrow$ 2026-09-01 | 4,942 | 7,179 | 585 | 172 | 2 |
| 41 | `DJT` | 2004-01-02 $\rightarrow$ 2004-08-10 | 2024-03-26 $\rightarrow$ 2026-09-01 | 4,936 | 7,167 | 152 | 611 | 2 |
| 42 | `ASBP` | 2004-01-02 $\rightarrow$ 2005-07-08 | 2025-02-20 $\rightarrow$ 2026-09-01 | 4,932 | 7,166 | 382 | 385 | 2 |
| 43 | `ALA` | 2004-01-02 $\rightarrow$ 2006-11-30 | 2026-07-10 $\rightarrow$ 2026-09-01 | 4,926 | 7,161 | 735 | 38 | 2 |
| 44 | `NXUS` | 2005-11-16 $\rightarrow$ 2006-02-17 | 2025-09-24 $\rightarrow$ 2026-09-01 | 4,926 | 7,158 | 64 | 236 | 2 |
| 45 | `VTP` | 2004-01-02 $\rightarrow$ 2005-12-02 | 2025-07-09 $\rightarrow$ 2026-09-01 | 4,924 | 7,158 | 485 | 290 | 2 |
| 46 | `DMN` | 2004-01-02 $\rightarrow$ 2005-05-13 | 2024-11-18 $\rightarrow$ 2025-05-19 | 4,908 | 7,128 | 344 | 124 | 2 |
| 47 | `BBHL` | 2005-08-18 $\rightarrow$ 2006-05-12 | 2025-11-17 $\rightarrow$ 2026-09-01 | 4,906 | 7,128 | 185 | 198 | 2 |
| 48 | `ETS` | 2004-01-02 $\rightarrow$ 2006-03-01 | 2025-08-21 $\rightarrow$ 2026-09-01 | 4,896 | 7,112 | 544 | 259 | 2 |
| 49 | `NBP` | 2004-01-02 $\rightarrow$ 2006-05-19 | 2025-10-30 $\rightarrow$ 2026-09-01 | 4,889 | 7,103 | 600 | 210 | 2 |
| 50 | `HTO` | 2004-01-02 $\rightarrow$ 2005-11-25 | 2025-05-06 $\rightarrow$ 2026-09-01 | 4,886 | 7,101 | 480 | 333 | 2 |

---

## 12. Supervisor-Oriented Methodological Interpretation

We summarize the core strategic questions for faculty discussion:

### Q1: Are long gaps common or rare?
**Answer**: Long gaps are **common among multi-spell tickers (55.95% have gaps $\ge 252$ sessions)**, but **moderately rare across the full historical universe (8.44% of all unique tickers)**. Long-term corporate absence is the dominant mode of ticker disappearance.

### Q2: What percentage of tickers would require identity investigation under thresholds of 20, 60, and 252 sessions?
**Answer**:
- Threshold $\ge 20$ sessions: **4,800 tickers (13.03% of all tickers, 86.41% of multi-spell tickers)**.
- Threshold $\ge 60$ sessions: **4,159 tickers (11.29% of all tickers, 74.87% of multi-spell tickers)**.
- Threshold $\ge 252$ sessions: **3,108 tickers (8.44% of all tickers, 55.95% of multi-spell tickers)**.

### Q3: Are most multi-spell cases short gaps, suggesting a simple continuity rule?
**Answer**: **No.** Short gaps ($\le 2$ sessions) account for only **12.06% of gap events (834 gaps)**. While a simple continuity rule is highly accurate for this 12% subset (91.4% confirmed same security), it leaves **88% of gap events unresolved**.

### Q4: Would excluding very long-gap cases materially reduce the historical universe?
**Answer**: **It depends strictly on the exclusion mechanism.**
- Under **Policy A (Dropping the entire ticker)** at $\ge 252$ sessions: Discards **13.19% of historical observations** (6.78 million observation days). This materially degrades historical coverage.
- Under **Policy B (Dropping only subsequent spells after a gap $\ge 252$)**: Discards only **7.14% of historical observations**.
- At an extreme threshold of $\ge 1,000$ sessions under Policy B: Discards only **3.64% of observations** across 1,924 tickers.

### Q5: Is the problem concentrated in a relatively small number of unusual tickers?
**Answer**: **Yes, for extreme multi-spell fragmentation, but no for general long gaps.**
- Extreme fragmentation ($\ge 5$ spells) is concentrated in only **45 tickers (0.12%)**, driven by feed anomalies (`CMCSA`, `DISCA`, `LINTA`).
- However, standard long gaps ($\ge 252$ sessions) are broadly distributed across **3,108 distinct corporate symbols**.

### Q6: Based purely on spell statistics, what threshold is reasonable to discuss with the supervisor?
**Answer**:
- **Candidate 1 ($\ge 252$ sessions / 1 Trading Year)**: Affected population is **3,108 tickers (8.44% of universe)**. Represents 92.2% non-continuity/unresolved rate. Best balance between research integrity and manual review scale.
- **Candidate 2 ($\ge 1,000$ sessions / ~4 Trading Years)**: Affected population is **1,924 tickers (5.22% of universe)**. Captures extreme multi-year ticker reuses with only 3.64% post-gap observation loss.

---

## 13. Summary of Generated Artifacts

The following analysis artifacts have been created under `data/quality/spell_statistics/`:
- `spell_summary.csv`: Master tabular summary of all aggregate metrics.
- `gap_distribution.csv`: Granular binned and cumulative gap distributions.
- `spell_duration_distribution.csv`: Distribution of continuous spell durations and percentiles.
- `ticker_spell_count_distribution.csv`: Distribution of discrete active spells per ticker.
- `long_gap_cases.csv`: Ranked database of all 5,682 gap cases $\ge 20$ trading sessions.
- `threshold_impact.csv`: Counterfactual impact matrix across exclusion policies.
- `anomalous_date_impact.csv`: Audit of corrupted snapshots and feed break dates.
- `gap_distribution.png`: Histogram of gap lengths with log-scale X axis.
- `gap_survival_curve.png`: Complementary cumulative survival curve for gaps.
- `spell_duration_distribution.png`: Histogram of active spell durations.
- `spells_per_ticker.png`: Bar chart of spell fragmentation per ticker symbol.
- `supervisor_summary.md`: Concise 2-page decision brief for faculty review.
