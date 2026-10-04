# v1.3 Feature Specification — Cross-Sectional Russell 1000 Ranking Model

> **Version**: 1.3 (FROZEN)  
> **Date**: 2026-10-03  
> **Architecture**: LSTM + Temporal Attention → Cross-Sectional Transformer + Market-Context Branch  
> **Target**: 5-trading-day forward cross-sectional rank of price returns  
> **Universe**: Point-in-time Russell 1000 ($U_t$)

---

## 0. Target Definition

### 0.1 Exact Formula per Era
The model prediction target is the forward 5-trading-session cumulative price return, evaluated cross-sectionally:

- **CRSP Era (2000-06-30 to 2024-12-31):**
  $$
  Y_{i,t} = \log \left( \prod_{s=1}^{5} \left(1 + RETX_{i,t+s}\right) \right) = \sum_{s=1}^{5} \log\left(1 + RETX_{i,t+s}\right)
  $$
  where $t+s$ denotes the $s$-th next valid trading session for stock $i$.
- **Yahoo Era (2025-01-02 to 2026-09-25):**
  $$
  Y_{i,t} = \log\left(\frac{Close_{i,t+5}}{Close_{i,t}}\right)
  $$
  where $Close$ is the split-adjusted closing price.

### 0.2 Trading-Session Indexing
$t+5$ strictly represents the **5th next valid trading session** for the same security, **not 5 calendar days**.
- If a stock is halted or suspended for intermediate market sessions, indexing advances across trading days on which the stock traded.
- If fewer than 5 future trading sessions exist within the dataset (e.g., end of data period or persistent cessation of trading without delisting code), $Y_{i,t} = \text{NaN}$.

### 0.3 Cross-Sectional Rank Computation
For each decision date $t$, the cross-sectional ranking vector $\boldsymbol{\pi}_t$ is computed across active constituents $i \in U_t$:
- Raw ranks: $\text{Rank}(Y_{i,t})$ using average rank for ties.
- Target representation for evaluation and loss computation: Normalized rank mapped to $[-0.5, +0.5]$:
  $$
  y^*_{i,t} = \frac{\text{Rank}(Y_{i,t}) - 1}{|U_t^{\text{valid}}| - 1} - 0.5
  $$
  where $U_t^{\text{valid}}$ is the set of active constituents with non-NaN targets (§0.5.1). Alternatively, normal scores $\text{NS}(Y_{i,t})$ may be used where Gaussian target marginals are required.

### 0.4 Delisting Handling
If stock $i$ delists between $t$ and $t+5$:
- Apply the conservative delisting policy defined in §10.4.
- In the CRSP era: if $DLRETX$ is available, compound intermediate price returns up to the delisting date and apply $DLRETX$. If $DLRETX$ is missing:
  - Mergers/Acquisitions (codes 200, 231, 233, 241, 244): terminal return is $0.0$.
  - Adverse delistings (codes 500, 520, 560, 570, 574, 580, 584): terminal penalty is $-0.30$.
  - Unknown/unresolved codes: $Y_{i,t} = \text{NaN}$.
- In the Yahoo era: delisting metadata is absent; disappearing securities yield $Y_{i,t} = \text{NaN}$.

---

## 0.5. Loss Function

### 0.5.1 Target Universe & Recommended Default: ListMLE
Not all constituents in $U_t$ will possess valid targets due to delistings, suspensions, or missing data. We define the **valid target universe** as:
$$
U_t^{\text{valid}} = \left\{ i \in U_t : Y_{i,t} \text{ is not NaN} \right\}
$$
While the model computes representations $\hat{y}_{i,t}$ for all $i \in U_t$, only stocks in $U_t^{\text{valid}}$ enter the ranking loss objective.

The primary objective function is **ListMLE** (Listwise Maximum Likelihood Estimation) evaluated strictly over $U_t^{\text{valid}}$:
$$
\mathcal{L}_{\text{ListMLE}} = - \sum_{t} \log P(\pi_t^{\text{valid}} \mid \hat{\mathbf{y}}_t^{\text{valid}}) = - \sum_{t} \sum_{k=1}^{|U_t^{\text{valid}}|} \log \left( \frac{\exp(\hat{y}_{\pi_t(k), t})}{\sum_{j=k}^{|U_t^{\text{valid}}|} \exp(\hat{y}_{\pi_t(j), t})} \right)
$$
where $\pi_t(k)$ is the index of the stock with the $k$-th highest realized return in $U_t^{\text{valid}}$ at date $t$, and $\hat{\mathbf{y}}_t^{\text{valid}}$ is the predicted score vector restricted to $U_t^{\text{valid}}$.

### 0.5.2 Alternative Loss Formulations (Ablation Candidates)
1. **Pairwise Ranking Loss (RankNet):**
   $$
   \mathcal{L}_{\text{Pairwise}} = \sum_{t} \frac{1}{|U_t^{\text{valid}}|(|U_t^{\text{valid}}|-1)} \sum_{i, j \in U_t^{\text{valid}}, i \neq j} \log \left( 1 + \exp\left( - \text{sign}(Y_{i,t} - Y_{j,t}) \cdot (\hat{y}_{i,t} - \hat{y}_{j,t}) \right) \right)
   $$
2. **Soft-Rank Loss:** Differentiable smooth approximation of the cross-sectional Spearman Rank Correlation (Rank-IC).
3. **Pointwise Regression Baseline:** Mean squared error or Huber loss on normalized return $y^*_{i,t}$, followed by post-hoc cross-sectional sorting.

### 0.5.3 Ablation Protocol
The four loss functions are evaluated under the fixed architecture defined in §9.7. Selection criterion: highest validation **Rank-IC** and **top-decile precision**.

---

## 0.6. Portfolio Construction

### 0.6.1 Execution and Rebalancing Schedule
- **Signal Generation:** Generated at market close of session $t$ using information set $\mathcal{F}_t$.
- **Execution Timestamp:** Market close of session $t$ (immediate closing cross execution).
- **Rebalance Frequency:** Non-overlapping 5 trading sessions (matching the target horizon $t \to t+5$). Alternatively, a rolling 5-day cycle using 5 staggered sub-portfolios can be deployed in production; v1.3 benchmarks evaluate the non-overlapping 5-session cycle.

### 0.6.2 Default Portfolio Weights: Equal-Weighted Deciles
Let $k = \lfloor |U_t^{\text{valid}}| / 10 \rfloor$ be the decile size.
- **Long Basket ($Q_1$):** Top $k$ stocks by predicted score $\hat{y}_{i,t}$.
  $$
  w_{i,t} = +\frac{1}{k} \quad \text{for } i \in Q_1
  $$
- **Short Basket ($Q_{10}$):** Bottom $k$ stocks by predicted score $\hat{y}_{i,t}$.
  $$
  w_{i,t} = -\frac{1}{k} \quad \text{for } i \in Q_{10}
  $$
- **Neutrality:** Dollar-neutral by construction: $\sum_{i} w_{i,t} = 0$, with gross leverage $\sum_{i} |w_{i,t}| = 2.0$.

### 0.6.3 Transaction Cost Model
- **Base Cost:** 10 bps round-trip (5 bps per one-way trade) assumed across all constituents to reflect liquid institutional Russell 1000 execution.
- **Net Portfolio Return:**
  $$
  R_{t \to t+5}^{\text{net}} = \sum_{i \in Q_1 \cup Q_{10}} w_{i,t} \cdot \left(e^{Y_{i,t}} - 1\right) - \text{Cost} \cdot \sum_{i} |w_{i,t} - w_{i,t-5}^+|
  $$
  where $w_{i,t-5}^+$ represents the drifted weight prior to rebalancing.

### 0.6.4 Portfolio Construction Ablation Plan
- **PORT-A (v1.3 Default):** Equal-weight top decile / bottom decile ($Q_1 - Q_{10}$).
- **PORT-B (Score-Weighted):** Weights proportional to demeaned score $\hat{y}_{i,t} - \bar{y}_t$, clipped at $\pm 3\sigma$.
- **PORT-C (Rank-Weighted):** Weights linear in normalized rank $y^*_{i,t}$.

---

## 1. Complete Feature Table

### 1.1 Notation & Conventions

| Symbol | Definition |
|--------|-----------|
| $C_s$ | Split-adjusted close at session $s$. CRSP: $\|PRC_s\|/CFACPR_s$. Yahoo: $Close_s$. |
| $O_s$ | Split-adjusted open at session $s$. CRSP: $\|OPENPRC_s\|/CFACPR_s$. Yahoo: $Open_s$. |
| $H_s$ | Split-adjusted high. CRSP: $ASKHI_s/CFACPR_s$. Yahoo: $High_s$. |
| $L_s$ | Split-adjusted low. CRSP: $BIDLO_s/CFACPR_s$. Yahoo: $Low_s$. |
| $V_s$ | Raw volume. CRSP: $VOL_s$. Yahoo: $Volume_s$. |
| $r_s$ | Daily log price return. CRSP: $\log(1+RETX_s)$. Yahoo: $\log(C_s/C_{s-1})$. |
| $r^{tot}_s$ | Daily log total return. CRSP: $\log(1+RET_s)$. Yahoo: $\log(AdjClose_s/AdjClose_{s-1})$. |
| $R^e_s$ | Excess return: $r^{tot}_s - rf_s$. |
| $F_s$ | Factor vector $[mktrf, smb, hml, rmw, cma, umd]_s^\top$. |
| $U_s$ | Point-in-time Russell 1000 constituents at session $s$. |
| $\text{NS}_s(\cdot)$ | Cross-sectional normal-score transform across $U_s$ at date $s$ (§7.1). |

> **Required History Convention:**  
> $\text{Req. History} = 60 + \text{max\_lookback}$, where 60 is the temporal model window length $[t-59, t]$ and $\text{max\_lookback}$ is the longest trailing window used by the feature. For features with lookback 0, $\text{Req. History} = 60$.  
> A feature with lookback $L$ strictly requires $\text{Req. History} = 60 + L$.

> **Warmup & Truncation Rule:**  
> Dates before the first valid prediction date (per the Req. History convention of 332 sessions, approximately late 2001) are strictly excluded from model training, validation, and backtesting. The model is **not** trained on partial, zero-filled, or padded historical windows.

### 1.2 Layer 1 — Primitive Daily Transformations

All Layer-1 features are computed at **every session** $s$ in the model input window $[t-59, t]$.

| # | Name | Formula | Lookback | Req. History | CRSP Path | Yahoo Path | Missing Policy | Outlier | CS Transform | TS Norm | Leakage Risk | Redundancy | Priority |
|:-:|------|---------|:--------:|:------------:|-----------|-----------|---------------|---------|:------------:|:-------:|:------------:|-----------|:--------:|
| 1 | `r_cc` | $\log(C_s/C_{s-1})$ | 1 | 61 | $\log(1+RETX_s)$ | $\log(Close_s/Close_{s-1})$ | NaN if RETX string-coded or first obs | None | NS | None | NONE | Basis for ret_Nd | **Core** |
| 2 | `r_on` | $\log(O_s/C_{s-1})$ | 1 | 61 | $\log(\|OPENPRC_s\|/\|PRC_{s-1}\|)$ ÷ matched CFACPR | $\log(Open_s/Close_{s-1})$ | NaN if OPENPRC null (~0.5–1.4%) | None | NS | None | NONE | Complement of `r_id`; R_ON std drops 32% at transition | **Core** |
| 3 | `r_id` | $\log(C_s/O_s)$ | 0 | 60 | $\log(\|PRC_s\|/\|OPENPRC_s\|)$ | $\log(Close_s/Open_s)$ | NaN if OPENPRC null | None | NS | None | NONE | $r_{cc}=r_{on}+r_{id}$ | **Core** |
| 4 | `range` | $(H_s-L_s)/C_s$ | 0 | 60 | $(ASKHI_s-BIDLO_s)/\|PRC_s\|$ | $(High_s-Low_s)/Close_s$ | NaN if any OHLC null | None | NS | None | NONE | Corr with `vol_20d` ~0.6 | **Core** |
| 5 | `close_loc` | $(C_s-L_s)/(H_s-L_s)$ | 0 | 60 | Per above | Per above | NaN if $H_s=L_s$ (~0.07%) | None | NS | None | NONE | Buying-pressure proxy | **Core** |
| 6 | `rel_vol` | $\log(V_s/\text{Med}_{20}(V)_s)$ | 20 | 80 | $\log(VOL_s/\text{Med}_{20}(VOL)_s)$ | $\log(Vol_s/\text{Med}_{20}(Vol)_s)$ | NaN if $V_s=0$ or warmup<20 | Clip at $[-5,5]$ | NS | None | LOW | Corr with `vol_shock_5` ~0.7 | **Core** |

### 1.3 Layer 2 — Temporal Statistics

Computed at every session $s$ in $[t-59, t]$. Lookback extends before the model window.

| # | Name | Formula | Lookback | Req. History | Missing Policy | CS Transform | Leakage Risk | Redundancy | Priority |
|:-:|------|---------|:--------:|:------------:|---------------|:------------:|:------------:|-----------|:--------:|
| 7 | `ret_5d` | $\sum_{j=0}^{4} r_{s-j}$ | 5 | 65 | NaN if any $r$ missing in window | NS | NONE | Contains `r_cc` | **Core** |
| 8 | `ret_20d` | $\sum_{j=0}^{19} r_{s-j}$ | 20 | 80 | NaN if >4 missing in window | NS | NONE | Contains `ret_5d` | **Core** |
| 9 | `ret_60d` | $\sum_{j=0}^{59} r_{s-j}$ | 60 | 120 | NaN if >12 missing in window | NS | NONE | Contains `ret_20d` | **Core** |
| 10 | `vol_20d` | $\sqrt{\frac{1}{19}\sum_{j=0}^{19}(r_{s-j}-\bar{r})^2}$ | 20 | 80 | NaN if <10 valid returns | NS | NONE | Corr with `range` ~0.6 | **Core** |
| 11 | `vol_60d` | Same as above with 60d window | 60 | 120 | NaN if <30 valid returns | NS | NONE | Smoothed `vol_20d` | **Core** |
| 12 | `vol_ratio` | $vol_{20d,s}\;/\;vol_{60d,s}$ | 60 | 120 | NaN if either missing | NS | NONE | Derived from 10, 11 | **Core** |
| 13 | `price_z_20` | $\log(C_s / \text{SMA}_{20,s}) \big/ \left(\sigma_{20,s} \cdot \sqrt{20}\right)$ | 20 | 80 | NaN if <10 valid prices or split in window | NS | LOW | Mean-reversion signal | **Core** |
| 14 | `drawdown_20` | $\log(C_s/\max_{j \in [0,19]} C_{s-j})$ | 20 | 80 | NaN if <10 valid prices or split in window | NS | NONE | Always $\le 0$ | **Core** |
| 15 | `eff_ratio_20` | $\|ret_{20d,s}\|/\sum_{j=0}^{19}\|r_{s-j}\|$ | 20 | 80 | NaN if denominator=0 | NS | NONE | $\in [0,1]$. Trend efficiency | **Core** |
| 16 | `vol_shock_5` | $\frac{1}{5}\sum_{j=0}^{4} rel\_vol_{s-j}$ | 24 | 84 | NaN if >2 missing in window | NS | NONE | Smoothed `rel_vol` | **Core** |

> **Geometric Moving Average & Dimensional Definition:**  
> In `price_z_20` (and `price_z_60`), $\text{SMA}_{h,s}$ is defined as the **geometric mean** of split-adjusted closing prices over the $h$-session window:
> $$
> \text{SMA}_{h,s} = \exp\left( \frac{1}{h} \sum_{j=0}^{h-1} \log C_{s-j} \right) \implies \log\left(\frac{C_s}{\text{SMA}_{h,s}}\right) = \log C_s - \frac{1}{h} \sum_{j=0}^{h-1} \log C_{s-j}
> $$
> **Dimensional Analysis:** The numerator $\log(C_s / \text{SMA}_{h,s})$ is a dimensionless log-ratio representing the deviation of the current log-price from the window mean log-price. The denominator $\sigma_{h,s} \cdot \sqrt{h}$ is the realized $h$-day standard deviation of the log-price random walk, which is also dimensionless. The resulting ratio is scale-free and dimensionless.

> **Split-Contamination Rule:**  
> If `CFACPR` changes within the trailing $h$-day lookback window in the CRSP era, the feature is assigned **NaN** for that date. This rule strictly applies to `price_z_20`, `price_z_60`, `drawdown_20`, and `drawdown_60`. (See §10.3 for era-specific behavior).

### 1.4 Layer 4 — Six-Factor Residualization

Computed at every session $s$ in $[t-59, t]$. Requires 252-session trailing regression window.

| # | Name | Formula | Span | Req. History | Missing Policy | CS Transform | Leakage Risk | Priority |
|:-:|------|---------|:----:|:------------:|---------------|:------------:|:------------:|:--------:|
| 17 | `resid_ret` | $\epsilon_s = R^e_s - \hat\alpha_{s-1} - \hat\beta_{s-1}^\top F_s$ | 253 | 313 | NaN if $\hat\beta_{s-1}$ unavailable or factors missing | NS | NONE | **Core** |
| 18 | `resid_mom_20` | $\sum_{j=0}^{19}\epsilon_{s-j}$ | 272 | 332 | NaN if >4 residuals missing in window | NS | NONE | **Core** |
| 19 | `resid_vol_20` | $\text{std}(\epsilon_{s-19},\ldots,\epsilon_s)$ | 272 | 332 | NaN if <10 valid residuals | NS | NONE | **Core** |
| — | `beta_mkt` | $\hat\beta_{MKT,s-1}$ from OLS on $[s{-}252,s{-}1]$ | 252 | 312 | NaN if <126 valid observations or collinear | NS | NONE | **Extended (E13)** |

> **Span Footnote:**  
> **Span** = the calendar span from the earliest data point used by the feature to session $s$, inclusive.  
> - For `resid_ret`, span is 253 sessions: 252 for beta estimation window $[s-252, s-1]$ plus 1 for evaluation at $s$. Required history = $60 + 253 = 313$.  
> - For `resid_mom_20` and `resid_vol_20`, summing $\epsilon$ over $[s-19, s]$ means the earliest data point is $s - 19 - 252 = s - 271$. Span = 272 sessions. Required history = $60 + 272 = 332$.  
> - `beta_mkt` resides in the Extended set (§4, E13).

### 1.5 Layer 5 — Market-Context Branch [v1.3]

Shared across all stocks. Computed once per session $s$. **Not cross-sectionally transformed**.

| # | Name | Formula | Lookback | Missing Policy | Priority |
|:-:|------|---------|:--------:|---------------|:--------:|
| M1 | `ctx_mktrf` | $mktrf_s$ | 0 | NaN if factor date missing | **Core** |
| M2 | `ctx_smb` | $smb_s$ | 0 | Same | **Core** |
| M3 | `ctx_hml` | $hml_s$ | 0 | Same | **Core** |
| M4 | `ctx_rmw` | $rmw_s$ | 0 | Same | **Core** |
| M5 | `ctx_cma` | $cma_s$ | 0 | Same | **Core** |
| M6 | `ctx_umd` | $umd_s$ | 0 | Same | **Core** |
| M7 | `ctx_mkt_vol_20` | $\text{std}(mktrf_{s-19},\ldots,mktrf_s)$ | 20 | NaN if <10 obs | **Core** |
| M8 | `ctx_mkt_ret_20` | $\sum_{j=0}^{19}mktrf_{s-j}$ | 20 | Same | **Core** |
| M9 | `ctx_breadth` | $\frac{1}{\|U_s\|}\sum_{i\in U_s}\mathbb{1}[r_{i,s}>0]$ | 0 | 0.5 if $U_s$ empty | **Core** |
| M10 | `ctx_dispersion` | $\text{std}_{i\in U_s}(r_{i,s})$ | 0 | NaN if $|U_s|<30$ | **Core** |
| M11 | `ctx_factor_disp` | $\text{std}(smb_s, hml_s, rmw_s, cma_s, umd_s)$ | 0 | NaN if factor missing | **Core** |

> **Computation Order & Raw Return Requirement [v1.3]:**  
> `ctx_breadth` and `ctx_dispersion` are computed from **raw** daily log returns $r_{i,s}$, strictly **before** any cross-sectional transformation. They are market-level aggregates across active constituents $U_s$, not stock-level features, and must **never** be computed from normal-scored values (which would render dispersion identically $1.0$ by mathematical construction and collapse breadth to a trivial median distance).

> **Context Normalization Protocol:**  
> Standardized using a strictly causal **rolling 252-session window** ending at $s-1$ with minimum 60 observations:
> $$
> \tilde{x}_s = \frac{x_s - \bar{x}_{[s-251, s-1]}}{\sigma_{[s-251, s-1]}}
> $$
> If fewer than 60 observations exist in $[s-251, s-1]$, $\tilde{x}_s = \text{NaN}$.

---

### 1.6 Feature Dependency Graph

```
[Raw Stock OHLCV] (CRSP: PRC, OPENPRC, ASKHI, BIDLO, VOL, RETX, RET, CFACPR)
                  (Yahoo: Open, High, Low, Close, Volume, Adj_Close)
  │
  ├──► r_cc (raw) ─────────┬──► ret_5d
  │        │               ├──► ret_20d ───┬──► eff_ratio_20
  │        │               ├──► ret_60d    │
  │        │               ├──► vol_20d ───┼──► vol_ratio
  │        │               │        │      └──► price_z_20 (also uses C_s, SMA_20)
  │        │               │        └───────────────────────────┐
  │        │               └──► vol_60d ──────► vol_ratio       │
  │        │                                                    │
  │        ├──► ctx_breadth (M9, aggregated across U_s raw r_cc)│
  │        └──► ctx_dispersion (M10, aggregated across U_s raw) │
  │                                                             │
  ├──► r_on                                                     │
  ├──► r_id                                                     │
  ├──► range                                                    │
  ├──► close_loc                                                │
  ├──► rel_vol ────────────► vol_shock_5                        │
  ├──► drawdown_20 (uses C_s)                                   │
  │                                                             │
  └──► (R^e = ret_total - rf)                                  │
              │                                                 │
[Factor Data: mktrf, smb, hml, rmw, cma, umd, rf]              │
  │           │                                                 │
  ├── Rolling OLS (trailing 252d ending s-1)                    │
  │     ├──► beta_mkt (E13)                                     │
  │     └──► (alpha, beta_vec) ──► resid_ret (epsilon_s)        │
  │                                      │                      │
  │                                      ├──► resid_mom_20      │
  │                                      └──► resid_vol_20      │
  │                                                             │
  └── Market Context Layer                                      │
        ├──► ctx_mktrf .. ctx_umd (M1-M6)                       │
        ├──► ctx_mkt_vol_20, ctx_mkt_ret_20 (M7-M8)            │
        └──► ctx_factor_disp (M11)                              │
```

- **Execution Order:** 1. Base series $\to$ 2. Primitives $\to$ 3. Market aggregates M9, M10 from raw $r_{i,s} \to$ 4. Primary rolling statistics $\to$ 5. Composite statistics $\to$ 6. Factor regressions $\to$ 7. Residual statistics $\to$ 8. Cross-sectional normal scores across $U_s \to$ 9. Rolling context z-scores.
- **Null Propagation:** If a parent feature evaluates to NaN, all dependent child features evaluate to NaN.
- **Cache Invalidation:** Modification to raw data triggers complete recomputation of all downstream nodes.

---

## 2. Per-Feature Verification Plan

### 2.1 Universal Tests

```
TEST_CAUSALITY:
  For each feature f and date t:
    1. Compute f(t) using data up to session t.
    2. Append 5 additional future trading sessions.
    3. Recompute f(t).
    4. ASSERT f(t) is bitwise identical.

TEST_LEAKAGE:
  For each feature f:
    1. Compute f on complete dataset.
    2. Compute f on truncated dataset ending at session T-100.
    3. ASSERT f(t) for all t <= T-100 is identical across both runs.

TEST_NUMERICAL_STABILITY:
  For each feature f and every (stock, date):
    1. ASSERT f is not +/- inf.
    2. Verify NaN occurrences strictly match documented missing-data policies.
    3. For ratio features: ASSERT zero-denominator checks guard against division by zero.

TEST_CROSS_ERA_CONSISTENCY:
  For each feature f:
    1. Compute empirical distribution for 2024-H2 (CRSP) and 2025-H1 (Yahoo).
    2. Evaluate Kolmogorov-Smirnov (KS) test and inspect QQ-plots.
    3. FLAG if KS p-value < 0.01 for subsequent investigation.
```

### 2.2 Feature-Specific Tests

| Feature | Causality Test | Leakage Test | Stability & Split-Contamination Test | Cross-Era Test |
|---------|---------------|-------------|---------------------------------------|---------------|
| `r_cc` | Standard | Standard | Verify split-adjusted return; assert no raw price returns used | Compare 2024-H2 vs 2025-H1 |
| `r_on` | Standard | Standard | Propagate NaN on OPENPRC null | **Inspect R_ON std & shape**: expect ~32% drop across transition. Cross-era shape test: compute KS test, skewness, and kurtosis of normal_scores(r_on) across 2024-H2 and 2025-H1. |
| `r_id` | Standard | Standard | Same as `r_on` | Compare distributions |
| `range` | Standard | Standard | Verify $H \ge L$ everywhere; check zero-range instances | Compare CRSP vs Yahoo range |
| `close_loc` | Standard | Standard | Verify $H = L \implies$ NaN | Confirm [0, 1] bounded range |
| `rel_vol` | Standard | Standard | Verify $V = 0 \implies$ NaN; verify clipping at $[-5, 5]$ | Volume scale check across eras |
| `ret_Nd` | Standard | Standard | Verify additive identity: $ret_{5d,s} = \sum_{j=0}^{4} r_{cc,s-j}$ | Standard |
| `vol_Nd` | Standard | Standard | Verify minimum observation thresholds enforced | Standard |
| `vol_ratio` | Standard | Standard | Verify denominator zero-guard | Standard |
| `price_z_20` | Standard | Check that geometric SMA uses causal prices | **Split-contamination test**: verify NaN if CFACPR changes in $[s-19, s]$; check zero $\sigma \sqrt{20}$ guard | Compare distribution |
| `drawdown_20` | Standard | Standard | **Split-contamination test**: verify NaN if CFACPR changes in $[s-19, s]$; assert $\le 0$ | Standard |
| `eff_ratio_20` | Standard | Standard | Verify bounded $\in [0, 1]$; check denom=0 | Standard |
| `vol_shock_5` | Standard | Verify span = 24 | Propagate `rel_vol` NaNs correctly | Standard |
| `resid_ret` | Lagged beta check | Verify beta window $[s{-}252, s{-}1]$ | **Collinearity test**: verify NaN when $\text{cond}(X) > 10^{10}$; check extreme residuals | Flag post-2026-07-31 as NaN |
| `resid_mom_20` | Span = 272 check | Verify no future beta leakage | Sum over stored historical residuals | Same |
| `resid_vol_20` | Same | Same | Check minimum 10 residuals required | Same |
| `beta_mkt` (E13) | Verify window $[s{-}252, s{-}1]$ | Standard | Verify condition number guard; bounds $|\hat\beta| < 10$ | Standard |
| Context M1–M11 | Verify window $[s{-}251, s{-}1]$ | Standard | Min 60 obs required for rolling 252d z-score | Factor end date boundary test |

---

## 3. Frozen Core Feature Set (19 Stock-Level + 11 Context)

### 3.1 Stock-Level Core (19 Features)

| # | Feature | Economic Rationale | Empirical Hypothesis |
|:-:|---------|-------------------|---------------------|
| 1 | `r_cc` | Captures daily price information arrival and short-term microstructural reversal/momentum at a 1-day horizon. | 1-day returns exhibit serial correlation and regime-dependent reversal that temporal attention decodes. |
| 2 | `r_on` | Overnight returns isolate non-trading-hour news arrival, earnings releases, and international market spillovers. | Overnight moves reflect informed trading and overnight news absorption distinct from intraday liquidity noise. |
| 3 | `r_id` | Intraday returns isolate trading-session price discovery, institutional execution flows, and liquidity imbalances. | Intraday price discovery carries distinct continuation dynamics relative to overnight sentiment gaps. |
| 4 | `range` | Normalized daily price range (Parkinson volatility proxy) estimates realized intra-session uncertainty. | Range provides an efficient, non-parametric volatility estimator superior to return variance for tail risk. |
| 5 | `close_loc` | Relative bar close location proxies for net buying/selling pressure into the closing cross. | Strong closes near session highs predict short-horizon momentum persistence across liquid large-caps. |
| 6 | `rel_vol` | Volume relative to trailing 20-day median proxies for abnormal attention and institutional participation. | Significant volume expansion signals informed institutional accumulation or distribution preceding price trends. |
| 7 | `ret_5d` | 5-day cumulative return captures short-horizon weekly reversal and momentum dynamics. | Weekly price drift exhibits robust cross-sectional predictability conditioned on realized volatility. |
| 8 | `ret_20d` | 1-month cumulative return captures intermediate-term momentum (Jegadeesh & Titman). | 20-day returns capture monthly price drift and analyst earnings estimate revision cycles. |
| 9 | `ret_60d` | 3-month cumulative return captures quarterly trend continuation and fundamental post-earnings drift. | Quarterly return persistence reflects gradual information diffusion across the institutional investor base. |
| 10 | `vol_20d` | Short-term realized return volatility quantifies total firm uncertainty and risk regime. | Realized volatility proxies for risk and anchors the low-volatility cross-sectional anomaly. |
| 11 | `vol_60d` | Intermediate realized volatility provides a smoothed baseline estimate of underlying total asset risk. | Serves as an anchor against which short-term volatility innovations are evaluated. |
| 12 | `vol_ratio` | Ratio of 20-day to 60-day volatility captures volatility compression or breakout dynamics. | Volatility expansion (ratio > 1.5) or compression (ratio < 0.7) signals regime shifts and trend transitions. |
| 13 | `price_z_20` | Volatility-standardized log distance from the 20-day moving average proxies for trend deviation. | Standardized distance from trend predicts mean-reversion in calm markets and breakout momentum in volatile markets. |
| 14 | `drawdown_20` | Log distance from the trailing 20-day high measures local peak-to-trough price destruction. | Nearness to 20-day high predicts breakout momentum; extreme drawdowns condition recovery or distress risk. |
| 15 | `eff_ratio_20` | Kaufman efficiency ratio measures price trend efficiency versus noise path length. | Highly efficient directional moves continue; inefficient random walks mean-revert. |
| 16 | `vol_shock_5` | 5-day smoothed abnormal volume isolates sustained institutional accumulation over transient noise. | Multi-day abnormal volume accumulation exhibits higher signal-to-noise ratio than single-day spikes. |
| 17 | `resid_ret` | Daily idiosyncratic return orthogonal to 6 systematic Fama-French + Momentum risk factors. | Factor-neutral residual returns isolate pure firm-specific price shocks unconfounded by macro tilts. |
| 18 | `resid_mom_20` | 20-day cumulative idiosyncratic momentum captures pure stock-specific drift (Blitz et al.). | Residual momentum exhibits superior Sharpe ratio and lower tail risk than unadjusted price momentum. |
| 19 | `resid_vol_20` | Realized idiosyncratic volatility measures firm-specific risk independent of factor exposures. | High idiosyncratic volatility predicts low future risk-adjusted returns (idiosyncratic volatility puzzle). |

### 3.2 Market-Context Core (11 Features)

| # | Feature | Rationale |
|:-:|---------|-----------|
| M1–M6 | `ctx_mktrf`, `ctx_smb`, `ctx_hml`, `ctx_rmw`, `ctx_cma`, `ctx_umd` | Informs the model of the macro factor return environment to condition cross-sectional stock rankings. |
| M7 | `ctx_mkt_vol_20` | Realized market volatility conditions cross-sectional signal confidence and factor dispersion. |
| M8 | `ctx_mkt_ret_20` | Trailing market trend identifies broad risk-on versus risk-off market regimes. |
| M9 | `ctx_breadth` | Market breadth (% advancers across $U_s$) measures whether rallies/declines are broad or concentrated. |
| M10 | `ctx_dispersion` | Cross-sectional return standard deviation quantifies the opportunity set available for long/short alpha. |
| M11 | `ctx_factor_disp` | Cross-sectional dispersion of the 5 style factors identifies active factor rotation intensity. |

---

## 4. Frozen Extended Feature Set (13 Ablation Candidates)

| # | Feature | Formula | Lookback / Span | Rationale | Ablation Experiment |
|:-:|---------|---------|:---------------:|-----------|-------------------|
| E1 | `body` | $\|C_s-O_s\|/(H_s-L_s)$ | 0 | Candle conviction: large body indicates directional force | ADD to core, measure $\Delta$Rank-IC |
| E2 | `upper_wick` | $(H_s-\max(O_s,C_s))/(H_s-L_s)$ | 0 | Intraday price rejection at highs signals buying exhaustion | ADD to core, measure $\Delta$Rank-IC |
| E3 | `lower_wick` | $(\min(O_s,C_s)-L_s)/(H_s-L_s)$ | 0 | Intraday price rejection at lows signals dip buying / support | ADD to core, measure $\Delta$Rank-IC |
| E4 | `price_z_60` | $\log(C_s/\text{SMA}_{60,s})\big/(\sigma_{60,s}\cdot\sqrt{60})$ | 60 | Intermediate-horizon mean-reversion signal ($\text{SMA}_{60}$ is geometric mean) | ADD alongside `price_z_20`, measure marginal IC |
| E5 | `drawdown_60` | $\log(C_s/\max_{60}(C))$ | 60 | Intermediate drawdown captures structural trend decay | ADD alongside `drawdown_20` |
| E6 | `eff_ratio_60` | $\|ret_{60d}\|/\sum_{j=0}^{59}\|r_j\|$ | 60 | Intermediate-horizon trend efficiency | ADD alongside `eff_ratio_20` |
| E7 | `log_dollar_vol` | $\log(\|PRC_s\| \cdot V_s)$ | 0 | Absolute dollar liquidity; proxy for size/turnover capacity | ADD, test if it adds capacity filtering |
| E8 | `amihud_20` | $\frac{1}{20}\sum_{j=0}^{19}\frac{\|r_{s-j}\|}{dollarVol_{s-j}}$ | 20 | Amihud illiquidity: price impact per dollar volume traded | ADD, measure marginal IC |
| E9 | `ret_skew_20` | $\text{skew}(r_{s-19},\ldots,r_s)$ | 20 | Return skewness proxies for retail lottery preference | ADD, test lottery-demand anomaly |
| E10 | `max_ret_20` | $\max(r_{s-19},\ldots,r_s)$ | 20 | Maximum single-day return in 20d (Bali et al. MAX effect) | ADD, compare marginal value vs skewness |
| E11 | `resid_mom_60` | $\sum_{j=0}^{59}\epsilon_{s-j}$ | Span 312 | Extended-horizon residual momentum | ADD alongside `resid_mom_20` |
| E12 | `beta_smb` | $\hat\beta_{SMB,s-1}$ from 252d OLS | Span 252 | Size factor sensitivity within the Russell 1000 | ADD, test conditional size tilt |
| E13 | `beta_mkt` | $\hat\beta_{MKT,s-1}$ from 252d OLS on $[s{-}252,s{-}1]$ | Span 252 | Systematic market risk exposure | ADD `beta_mkt` to core set, measure $\Delta$Rank-IC |

---

## 5. Frozen Reject Set

| Feature | Reason for Rejection |
|---------|---------------------|
| Raw price level ($PRC$, $Close$) | Not scale-free; cross-sectionally meaningless; captured scale-free by `price_z_20`. |
| Market capitalization ($SHROUT \times PRC$) | `SHROUT` is completely absent in Yahoo era (2025–2026); creates severe source asymmetry. |
| Bid-ask spread ($ASK - BID$) | Completely absent in Yahoo era (2025–2026). |
| Number of trades ($NUMTRD$) | Completely absent in Yahoo era; activity is adequately captured by `rel_vol`. |
| Sector dummies | Present in Yahoo era, absent in CRSP; introduces uncalibrated categorical asymmetry. |
| 252d price momentum ($ret_{252d}$) | Requires 312 sessions history at $t-59$. Redundant with `ret_60d` and `resid_mom_20`. |
| 252d momentum minus 20d ($ret_{252d} - ret_{20d}$) | High latency and redundancy with intermediate momentum and residual momentum features. |
| Short interest | Not available in provided historical dataset. |
| Analyst consensus / earnings estimates | Not available in provided historical dataset. |
| News sentiment / alternative data | Not available in provided historical dataset. |
| Option-implied volatility (IV) | Not available in provided historical dataset. |
| Days since IPO | Unreliable / non-computable across mixed data sources. |
| Absolute return ($\|r_{cc}\|$) | Redundant with `range` and `vol_20d`; unnormalized scale. |
| Dollar-volume rank | Redundant with `rel_vol` (core) and `log_dollar_vol` (extended). |
| Rolling betas to HML, RMW, CMA individually | Combinatorial model expansion; style rotation captured via context M1–M6 and residualization. |

---

## 6. Residualization Procedure

### 6.1 Pseudocode

```python
import numpy as np
import pandas as pd

def compute_residuals(
    stock_excess: pd.Series,    # R^e_{i,s} indexed by date s
    factors: pd.DataFrame,       # columns: mktrf, smb, hml, rmw, cma, umd
    min_obs: int = 126,          # minimum observations for valid regression
    n_obs: int = 252             # 252 observations, spanning dates[s-252] through dates[s-1] inclusive
) -> dict:
    """
    Computes strictly causal 6-factor residuals.
    Beta is estimated over the trailing 252 sessions ending strictly at s-1.
    Residual at s is computed using the lagged beta vector beta_{s-1}.
    Historical residuals are permanently stored and never re-estimated.
    Guards against collinear factor regimes via condition number thresholding.
    """
    dates = stock_excess.index.sort_values()
    epsilon = pd.Series(np.nan, index=dates)
    beta_mkt = pd.Series(np.nan, index=dates)
    alpha = pd.Series(np.nan, index=dates)
    
    # Step 1: Rolling OLS strictly lagged
    for idx, s in enumerate(dates):
        train_end_pos = idx - 1        # strictly s-1
        train_start_pos = idx - n_obs  # s-252
        
        if train_end_pos < 0 or train_start_pos < 0:
            continue
        
        train_dates = dates[train_start_pos : train_end_pos + 1]
        
        y_train = stock_excess.loc[train_dates].dropna()
        X_train = factors.loc[train_dates].dropna()
        common = y_train.index.intersection(X_train.index)
        
        if len(common) < min_obs:
            continue   # Insufficient history -> parameters remain NaN
        
        y = y_train.loc[common].values
        # Add constant for alpha estimation
        X = np.column_stack([np.ones(len(common)), X_train.loc[common].values])
        
        # Rank deficiency check: guard against severe factor multicollinearity
        if np.linalg.cond(X) > 1e10:
            continue  # Collinear factors -> parameters remain NaN
        
        try:
            params = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        
        alpha.iloc[idx] = params[0]
        beta_mkt.iloc[idx] = params[1]     # Coefficient on mktrf (E13)
        beta_vec = params[1:]             # Full 6-factor slope vector
        
        # Step 2: Evaluate causal as-of residual at session s
        if s in factors.index:
            F_s = factors.loc[s].values
            predicted = params[0] + np.dot(beta_vec, F_s)
            epsilon.iloc[idx] = stock_excess.iloc[idx] - predicted
    
    # Step 3: Rolling statistics across stored causal residuals
    resid_mom_20 = epsilon.rolling(20, min_periods=16).sum()
    resid_vol_20 = epsilon.rolling(20, min_periods=10).std()
    
    return {
        'epsilon': epsilon,            # resid_ret
        'resid_mom_20': resid_mom_20,  # 20d residual momentum
        'resid_vol_20': resid_vol_20,  # 20d residual volatility
        'beta_mkt': beta_mkt,          # market beta (E13)
        'alpha': alpha
    }
```

### 6.2 Causality Guarantees
- **Strictly Lagged Beta:** $\hat\alpha_{s-1}$ and $\hat\beta_{s-1}$ utilize data up to $s-1$ only (`train_end_pos = idx - 1`).
- **One-Step Forward Innovation:** $\epsilon_s = R^e_s - \hat\alpha_{s-1} - \hat\beta_{s-1}^\top F_s$. No data from session $s$ enters parameter estimation.
- **Permanent Residual Storage:** $\epsilon_s$ is saved upon computation. As new dates arrive, past residuals are never recomputed.
- **Residual Momentum:** Computed as a rolling sum over the frozen historical residual series.

### 6.3 Edge Cases
- **Fewer than 126 observations:** $\hat\beta$, $\epsilon$, and derivative features remain NaN.
- **Multicollinearity / Ill-Conditioning:** If $\text{cond}(X) > 10^{10}$, parameters and residual evaluate to NaN to prevent explosive unidentifiable slope estimates.
- **Missing factor data (post 2026-07-31):** $\epsilon_s = \text{NaN}$; derivative features cascade to NaN.
- **String return codes:** Filtered to NaN prior to regression; skipped in estimation.

---

## 7. Cross-Sectional Transformation Procedure

### 7.1 Three Transformations

```python
import numpy as np
import pandas as pd
import scipy.stats as st

def normal_scores(x: pd.Series, min_universe: int = 100) -> pd.Series:
    """
    Rank-based inverse-normal (Blom) transform.
    Maps cross-section to exact standard normal marginals N(0, 1).
    Robust to outliers, split jumps, and zero-variance degenerate sections.
    """
    valid = x.dropna()
    n = len(valid)
    if n < min_universe:
        return pd.Series(np.nan, index=x.index)
    if valid.std() < 1e-10:
        return pd.Series(0.0, index=x.index)
    ranks = valid.rank(method='average')
    uniform = (ranks - 0.375) / (n + 0.25)  # Blom plotting position
    scores = pd.Series(np.nan, index=x.index)
    scores.loc[valid.index] = st.norm.ppf(uniform)
    return scores

def cs_zscore(x: pd.Series) -> pd.Series:
    """Standard cross-sectional z-score."""
    mu, sigma = x.mean(), x.std()
    if sigma < 1e-12 or pd.isna(sigma):
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sigma

def robust_zscore(x: pd.Series) -> pd.Series:
    """Robust cross-sectional z-score using median and MAD."""
    med = x.median()
    mad = (x - med).abs().median()
    if mad < 1e-12 or pd.isna(mad):
        return pd.Series(0.0, index=x.index)
    return (x - med) / (mad * 1.4826)
```

### 7.2 Decision Rule & Application Policy

| Feature Category | v1.3 Transform | Rationale |
|-----------------|:-:|-----------|
| **All 19 stock-level features** (Layers 1, 2, 4) | **Normal Scores** | Ranks are invariant to monotonic transformations, eliminate extreme corporate action outliers, and make the LSTM invariant to cross-era scale shifts. |
| **All 11 market-context features** (Layer 5) | **Rolling Causal z-score** (252d, min 60) | Macro quantities have no cross-section; rolling time-series z-score enforces stationarity. |

> **Application Rule:**  
> The cross-sectional transform is applied **independently at every session** $s \in [t-59, t]$ across the active point-in-time universe $U_s$. The stock's feature value at session $s$ is its normal score within $U_s$. The temporal model therefore receives a 60-step temporal trajectory of cross-sectionally normalized feature values $\mathbf{X}_i \in \mathbb{R}^{60 \times 19}$, not a single static cross-sectional snapshot at $t$.

> **Missing-Value Policy for Cross-Sectional Transforms:**  
> Stocks with NaN feature values at session $s$ are excluded from the cross-sectional ranking at that session. They receive NaN normal scores. In the loss function, they are masked out. In portfolio construction, they are ineligible for selection. This is intentional: imputing missing values would inject model assumptions into the ranking and create a spurious distinction between "missing" and "extreme".

> **Minimum Universe Size:**  
> If $|U_s| < 100$, the date is skipped by default in training/evaluation, or falls back to cross-sectional z-scores.

### 7.3 Ablation Plan for Transforms
- **CS-A (v1.3 Default):** Normal scores on all stock features.
- **CS-B:** Cross-sectional z-scores on all stock features.
- **CS-C:** Robust z-scores (Median/MAD) on all stock features.
- **CS-D:** Raw features (no cross-sectional transform).

---

## 8. Market-Context Branch Specification

### 8.1 Input Features
The 11 context features (M1–M11) form a market vector $\mathbf{c}_s \in \mathbb{R}^{11}$ at each session $s$. Assembled across the 60-day window: $\mathbf{C} \in \mathbb{R}^{60 \times 11}$.

### 8.2 Architecture (v1.3 Default — Concatenation)

```
Context Branch:
  Input:  C = [c_{t-59}, ..., c_t] in R^{60 x 11} (rolling 252d z-scored)
  Layer:  Bidirectional LSTM(hidden_size=32) -> last hidden state -> FC(64) -> ReLU -> FC(d_model)
  Output: Context embedding z_ctx in R^{d_model}

Stock Branch (per stock i in U_t):
  Input:  X_i = [x^i_{t-59}, ..., x^i_t] in R^{60 x 19} (normal-scored)
  Layer:  LSTM(hidden_size=128, 2 layers) + Temporal Self-Attention -> h_i in R^{d_model}

Fusion (v1.3 Default):
  Concatenation:  h'_i = [h_i ; z_ctx] in R^{2 * d_model}
  Projection:     e_i = Linear(2 * d_model -> d_model)(h'_i)

Cross-Sectional Transformer:
  Input:  {e_1, e_2, ..., e_N} where N = |U_t|
  Layers: 2 Transformer Encoder layers with multi-head self-attention
  Output: Scalar score y_hat_i in R
```

### 8.3 FiLM Ablation (Candidate v2.0)
Feature-wise Linear Modulation (FiLM):
$$
\boldsymbol{\gamma}, \boldsymbol{\beta} = \text{Linear}(d_{\text{model}} \to 2 \cdot d_{\text{model}})(\mathbf{z}_{\text{ctx}})
$$
$$
\mathbf{e}_i = \boldsymbol{\gamma} \odot \mathbf{h}_i + \boldsymbol{\beta}
$$

### 8.4 Ablation Plan
- **CTX-0:** No context branch (stock features only).
- **CTX-A (v1.3 Default):** Bidirectional LSTM context branch with concatenation fusion.
- **CTX-B:** Bidirectional LSTM context branch with FiLM modulation fusion.
- **CTX-C:** Step-wise concatenation (context appended to each stock step $s$).

---

## 9. Ablation Framework

### 9.1 Feature-Family Ablations

| ID | Features Included | Total Features | Description |
|:--:|:----------------:|:--------------:|-------------|
| **A** | Layer 1 only | 6 | Primitives: `r_cc`, `r_on`, `r_id`, `range`, `close_loc`, `rel_vol` |
| **B** | Layer 1 + 2 | 16 | + Temporal statistics (momentum, volatility, drawdowns) |
| **C** | Layer 1 + 2 + 4 | 19 | + Residualization (`resid_ret`, `resid_mom_20`, `resid_vol_20`) |
| **D** | Layer 1 + 2 + 4 + 5 | 19 + 11 | Full v1.3 model (+ Market context branch) |

### 9.2 Architecture Ablations
- **X:** Stock LSTM only (per-stock temporal processing, no cross-sectional attention, no context).
- **Y:** Stock LSTM + Cross-Sectional Transformer (no context branch).
- **Z:** Stock LSTM + Cross-Sectional Transformer + Context Branch (Full v1.3 architecture).

### 9.3 Cross-Sectional Transform Ablations
CS-A (Normal scores), CS-B (Standard z-scores), CS-C (Robust z-scores), CS-D (Raw).

### 9.4 Extended Feature Ablations
- **E-candle:** Core + E1 (`body`), E2 (`upper_wick`), E3 (`lower_wick`).
- **E-long:** Core + E4 (`price_z_60`), E5 (`drawdown_60`), E6 (`eff_ratio_60`).
- **E-liq:** Core + E7 (`log_dollar_vol`), E8 (`amihud_20`).
- **E-lottery:** Core + E9 (`ret_skew_20`), E10 (`max_ret_20`).
- **E-resid:** Core + E11 (`resid_mom_60`), E12 (`beta_smb`).
- **E-beta:** Core + E13 (`beta_mkt`).

### 9.5 Walk-Forward Protocol

```python
def purged_walk_forward_cv(data, n_splits=5, horizon=5, embargo=5):
    """
    Purged Walk-Forward Cross-Validation protocol with embargo.
    
    Derivation: A training sample at t uses forward target information up to t+horizon.
    To prevent overlap and feature serial correlation with the test set starting at
    test_start, the last valid training date is: test_start - (horizon + embargo).
    
    Dataset Size Assumption:
    The dataset is assumed large enough (26 years, ~6,600 sessions) that each fold has
    a non-empty training set. If fold_size is excessively small such that
    last_train_idx <= 0, the fold is safely skipped.
    """
    unique_dates = np.sort(data['date'].unique())
    n_dates = len(unique_dates)
    fold_size = n_dates // n_splits
    
    for k in range(1, n_splits):
        test_start_idx = k * fold_size
        test_end_idx = min((k + 1) * fold_size, n_dates)
        
        test_start_date = unique_dates[test_start_idx]
        test_end_date = unique_dates[test_end_idx - 1]
        
        # Enforce purge and embargo
        last_train_idx = test_start_idx - (horizon + embargo)
        if last_train_idx <= 0:
            continue  # Assumption check: non-empty training history required
        last_valid_train_date = unique_dates[last_train_idx]
        
        # Partition training set
        train_full = data[data['date'] <= last_valid_train_date]
        
        # Internal validation partition with identical purge logic
        train_dates = unique_dates[: last_train_idx + 1]
        val_size = int(len(train_dates) * 0.15)
        val_start_idx = len(train_dates) - val_size
        val_start_date = train_dates[val_start_idx]
        
        last_subtrain_idx = val_start_idx - (horizon + embargo)
        subtrain_date = train_dates[last_subtrain_idx]
        
        train_data = data[data['date'] <= subtrain_date]
        val_data = data[(data['date'] >= val_start_date) & (data['date'] <= last_valid_train_date)]
        test_data = data[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)]
        
        yield train_data, val_data, test_data
```

### 9.6 Evaluation Metrics
- **Rank IC:** Spearman correlation $\rho(\hat{\mathbf{y}}_t, \mathbf{Y}_t)$ across $U_t^{\text{valid}}$.
- **IC:** Pearson correlation $r(\hat{\mathbf{y}}_t, \mathbf{Y}_t)$.
- **IC_IR:** Newey-West HAC-adjusted information ratio: $\text{mean}(IC_t)/\text{std}_{\text{HAC}}(IC_t)$.
- **Top-$k$ / Bottom-$k$ Precision:** Proportion of predicted top/bottom decile realized in actual top/bottom decile.
- **Long-Short Spread & Sharpe:** Net portfolio return of top decile minus bottom decile (HAC-adjusted annualization, §0.6).
- **Turnover & Maximum Drawdown.**
- **Regime Performance Breakdown:** Stratified across VIX quartiles and market trend states.

### 9.7 Hyperparameter Protocol

To avoid confounding feature selection with model capacity and optimizer tuning, all feature ablations are conducted under a **strictly frozen model architecture**:

- **Stock Temporal Branch:**
  - LSTM: 2 layers, hidden size = 128, bidirectional = False.
  - Temporal Attention: 4 heads, head dimension = 32, dropout = 0.1.
- **Cross-Sectional Transformer:**
  - Encoder Layers: 2.
  - Multi-head Attention: 4 heads, model dimension $d_{\text{model}} = 128$.
  - Feedforward Dimension: 256.
  - Dropout: 0.1.
- **Context Branch:**
  - Bidirectional LSTM: 1 layer, hidden size = 32.
  - Dense projection: FC(64) $\to$ ReLU $\to$ FC(128).
- **Optimization:**
  - Optimizer: AdamW, initial learning rate = $1 \times 10^{-3}$, weight decay = $1 \times 10^{-4}$.
  - Batching: Full cross-section per date, batch size = 64 dates.
  - Early Stopping: Monitored on validation Rank-IC with patience = 10 epochs.

> **Protocol Rule:** Hyperparameter optimization (learning rate, layer count, dimension sizes) is strictly prohibited during feature ablation. Hyperparameters will be tuned in a separate phase **after** the feature set is frozen.

---

## 10. Data-Engineering Checklist

### 10.1 Point-in-Time Universe Construction
```python
def build_universe(reconstitution_csv: pd.DataFrame, date: pd.Timestamp) -> set:
    """
    Points to data/processed/russell1000_all_years.csv
    Reconstitution takes place at end of June annually.
    Dates in [July Y, June Y+1) use Year=Y constituents.
    """
    recon_year = date.year if date.month >= 7 else date.year - 1
    members = reconstitution_csv[reconstitution_csv['Year'] == recon_year]
    return set(members['Ticker'].dropna().unique())
```

### 10.2 Join Keys and Date Alignment
- **CRSP Era (2000-06-30 to 2024-12-31):** Primary key `(PERMNO, date)`. Map `TICKER` $\to$ `PERMNO`. Calendar-day join with `data/ff.csv`.
- **Yahoo Era (2025-01-02 to 2026-09-25):** Primary key `(Ticker, Date)`. Calendar-day join with `data/ff.csv`.
- **Source Transition Boundary:** 995 common tickers between 2024-12-31 and 2025-01-02.

### 10.3 Schema Transition Normalization & Unified Mapping [v1.3]

To avoid ambiguous column lookups and subtle bugs in downstream feature transformations, the mathematical notations from §1.1 map directly to normalized DataFrame columns across both data eras as follows:

| Notation | Concept | Unified Column | CRSP Construction (2000–2024) | Yahoo Construction (2025–2026) |
|---|---|---|---|---|
| $C_s$ | Split-adjusted close | `split_adj_close` | $\|PRC_s\| / CFACPR_s$ | $Close_s$ (already split-adjusted) |
| $O_s$ | Split-adjusted open | `split_adj_open` | $\|OPENPRC_s\| / CFACPR_s$ | $Open_s$ (already split-adjusted) |
| $H_s$ | Split-adjusted high | `split_adj_high` | $ASKHI_s / CFACPR_s$ | $High_s$ (already split-adjusted) |
| $L_s$ | Split-adjusted low | `split_adj_low` | $BIDLO_s / CFACPR_s$ | $Low_s$ (already split-adjusted) |
| $V_s$ | Raw volume | `volume` | $VOL_s$ | $Volume_s$ |
| $r_s$ | Daily log price return | `ret_price` | $\log(1 + RETX_s)$ | $\log(Close_s / Close_{s-1})$ |
| $r^{tot}_s$ | Daily log total return | `ret_total` | $\log(1 + RET_s)$ | $\log(AdjClose_s / AdjClose_{s-1})$ |

> **Split-Contamination Handling Across Eras [v1.3]:**  
> In the CRSP era, `CFACPR` is used to detect splits within lookback windows (assigning NaN when a split occurs within the window). In the Yahoo era, `CFACPR` is a placeholder ($1.0$), and the split-contamination check is a **no-op** because Yahoo's `Close` is already split-adjusted at source. The feature pipeline must **not** assume `cfacpr != 1.0` is required to detect splits in the Yahoo era; it must instead rely on Yahoo's adjusted series directly.

> **High/Low Field Semantics (CRSP):**  
> `ASKHI` and `BIDLO` represent the CRSP-provided daily high and low. For liquid Russell 1000 names, these represent the true intraday high and low. For illiquid names with wide bid-ask spreads, quote midpoints may influence these values. The impact on `range` and `close_loc` is bounded and was validated during the forensic EDA ($H \ge \max(O,C)$ and $L \le \min(O,C)$ held across 99.999% of observations).

```python
def normalize_schema(df: pd.DataFrame, era: str) -> pd.DataFrame:
    """
    Normalizes vendor-specific schemas into a unified, era-agnostic DataFrame.
    Guarantees consistent split-adjusted prices and returns across 2000-2026.
    """
    if era == 'CRSP':
        cfacpr = df['CFACPR'].replace(0, np.nan)
        out = pd.DataFrame({
            'date': pd.to_datetime(df['date']),
            'ticker': df['TICKER'],
            'permno': df['PERMNO'],
            'close_raw': df['PRC'].abs(),
            'open_raw': df['OPENPRC'].abs(),
            'high_raw': df['ASKHI'],
            'low_raw': df['BIDLO'],
            'volume': df['VOL'],
            'split_adj_close': df['PRC'].abs() / cfacpr,
            'split_adj_open': df['OPENPRC'].abs() / cfacpr,
            'split_adj_high': df['ASKHI'] / cfacpr,
            'split_adj_low': df['BIDLO'] / cfacpr,
            'ret_price': np.log1p(pd.to_numeric(df['RETX'], errors='coerce')),
            'ret_total': np.log1p(pd.to_numeric(df['RET'], errors='coerce')),
            # Note: div_adj_close is intentionally omitted for CRSP because RET already
            # encodes total returns directly. split_adj_close is used for price returns;
            # ret_total is used for excess return regression against factors.
            'cfacpr': df['CFACPR'],
            'cfacshr': df['CFACSHR'],
            'dlstcd': df['DLSTCD'],
            'dlret': pd.to_numeric(df['DLRET'], errors='coerce'),
            'shrout': df['SHROUT'],
        })
    elif era == 'Yahoo':
        out = pd.DataFrame({
            'date': pd.to_datetime(df['Date']),
            'ticker': df['Ticker'],
            'permno': np.nan,
            'close_raw': df['Close'],          # Yahoo Close is already split-adjusted
            'open_raw': df['Open'],
            'high_raw': df['High'],
            'low_raw': df['Low'],
            'volume': df['Volume'],
            'split_adj_close': df['Close'],
            'split_adj_open': df['Open'],
            'split_adj_high': df['High'],
            'split_adj_low': df['Low'],
            'ret_price': np.nan,
            'ret_total': np.nan,
            'cfacpr': 1.0,                     # Placeholder: splits already adjusted in Close
            'cfacshr': 1.0,
            'div_adj_close': df['Adj_Close'],  # Backward dividend+split adjusted
            'dlstcd': np.nan,
            'dlret': np.nan,
            'shrout': np.nan,
        })
        out = out.sort_values(['ticker', 'date'])
        out['ret_price'] = np.log(out.groupby('ticker')['split_adj_close'].pct_change() + 1.0)
        out['ret_total'] = np.log(out.groupby('ticker')['div_adj_close'].pct_change() + 1.0)
    return out
```

### 10.4 Delisting Policy Implementation
```python
def apply_delisting(row: pd.Series) -> float:
    """Returns terminal return for delisting event."""
    if pd.isna(row.get('dlstcd')) or row['dlstcd'] == 100:
        return None  # Still active
    
    dlret = row.get('dlret')
    if pd.notna(dlret):
        return float(dlret)
        
    code = int(row['dlstcd'])
    if code in (200, 231, 233, 241, 244):
        return 0.0  # M&A: deal price proxy
    elif code in (500, 520, 560, 570, 574, 580, 584):
        return -0.30  # Adverse failure penalty
    return np.nan
```

### 10.5 Handling CRSP String Return Codes and Factor End Date
- `pd.to_numeric(df['RET'], errors='coerce')` converts codes 'B', 'C', 'T', 'S', 'A' to NaN.
- Factor series ends on **2026-07-31**. All factor-dependent features (`resid_*`, `ctx_*`) evaluate to NaN thereafter. Training and backtesting are strictly truncated at 2026-07-31.

### 10.6 Reproducibility Protocol
To ensure strict forensic reproducibility across all model experiments and data pipeline runs:
1. **Random Seed Management:**
   ```python
   import random, os, numpy as np, torch
   def seed_everything(seed=42):
       random.seed(seed)
       os.environ['PYTHONHASHSEED'] = str(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.use_deterministic_algorithms(True)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
   ```
2. **Deterministic Backing Operations:** Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in runtime environments to enforce deterministic GEMM and embedding reductions.
3. **Data Verification Hashes:** Input Parquet and factor CSV files must match verified SHA-256 manifest records before execution:
   - `data/ff.csv`: checked against baseline sha256 checksum.
   - `data/WRDS/{2000..2026}.parquet`: audited and verified read-only.
4. **Environment Specification:** Python 3.11 with pinned versions (`torch==2.2.*`, `pandas==2.2.*`, `polars==0.20.*`, `scipy==1.12.*`, `pyarrow==15.0.*`).

### 10.7 Consolidated Loading
```python
def load_all_data():
    frames = []
    for yr in range(2000, 2025):
        df = pd.read_parquet(f'data/WRDS/{yr}.parquet')
        frames.append(normalize_schema(df, 'CRSP'))
    for yr in (2025, 2026):
        df = pd.read_parquet(f'data/WRDS/{yr}.parquet')
        frames.append(normalize_schema(df, 'Yahoo'))
    
    stock = pd.concat(frames, ignore_index=True)
    stock = stock.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    factors = pd.read_csv('data/ff.csv', parse_dates=['date'])
    universe = pd.read_csv('data/processed/russell1000_all_years.csv')
    
    return stock, factors, universe
```

---

## 11. Risk Register

| # | Issue | Impact | Mitigation | Blocks v1.3? |
|:-:|-------|--------|-----------|:------------:|
| R1 | $R^{ON}$ std drops 32% at transition | Risk of learning source indicator | Normal score transformation applied per-date eliminates cross-era scale shifts. | **No** |
| R2 | Yahoo era lacks delisting data | Survivorship bias in forward target | Set target to NaN for stocks disappearing before $t+5$. | **No** |
| R3 | Factor data ends 2026-07-31 | Missing factor features post-July 2026 | Truncate model backtest and training at 2026-07-31. | **No** |
| R4 | Data begins 2000-06-30 | Warmup constraint | 332-session required history pushes first prediction to late 2001. | **No** |
| R5 | Null `OPENPRC` (~0.5–1.4% CRSP) | Missing overnight returns | Normal scores exclude NaNs; graceful rank imputation. | **No** |
| R6 | $H = L$ division-by-zero (~0.07%) | `close_loc` NaNs | Zero-range instances evaluate to NaN. | **No** |
| R7 | 1,022 fake delistings in 2024 | False terminal exit | Explicit filter: `if dlstcd == 100: ignore`. | **No** |
| R8 | Split contamination in rolling prices | Artificial jumps in price z-score | Split-contamination rule: assign NaN if CFACPR changes in window (CRSP era). | **No** |
| R9 | Factor matrix rank deficiency | Exploding unidentifiable OLS betas | Condition number check: assign NaN if $\text{cond}(X) > 10^{10}$. | **No** |

---

## Appendix A: Complete Pipeline Diagram

```
Raw Data (WRDS Parquet: 2000-2026 + ff.csv)
  │
  ├─ normalize_schema() ──► Unified Schema (OHLCV, returns, adjustment factors)
  ├─ build_universe() ────► Point-in-time Russell 1000 constituent mask U_s
  └─ apply_delisting() ──► Causal terminal returns
  │
  ▼
Per-Stock Feature Computation (for each session s in [t-59, t], for each stock i in U_s):
  │
  ├─ Layer 1 Primitives (6 features):
  │     r_cc, r_on, r_id, range, close_loc, rel_vol
  │
  ├─ Layer 2 Temporal Statistics (10 features):
  │     ret_5d, ret_20d, ret_60d, vol_20d, vol_60d, vol_ratio,
  │     price_z_20, drawdown_20, eff_ratio_20, vol_shock_5
  │
  ├─ Layer 4 Factor Residualization (3 core features):
  │     resid_ret, resid_mom_20, resid_vol_20  (beta_mkt moved to E13)
  │
  ▼
Cross-Sectional Normal Scores Transform:
  │
  └─ For each feature f in 1..19:  NS_s(f) across U_s  ==>  X_i in R^{60 x 19}
  │
Market-Context Branch:
  │
  ├─ Aggregates M9 (ctx_breadth) & M10 (ctx_dispersion) computed from RAW r_cc
  ├─ Factor Context Features: M1..M8, M11
  └─ Rolling 252d causal z-score standardization  ==>  C in R^{60 x 11}
  │
  ▼
Model Architecture:
  │
  ├─ Stock Temporal Branch:  LSTM(128, 2L) + Temporal Self-Attention  ──► h_i in R^{128}
  ├─ Market Context Branch:  BiLSTM(32) + MLP Projection               ──► z_ctx in R^{128}
  ├─ Fusion:                 e_i = Linear([h_i ; z_ctx])               ──► e_i in R^{128}
  │
  ├─ Cross-Sectional Transformer Encoder (2 Layers, 4 Heads)           ──► y_hat_i in R
  │
  ▼
Optimization (§0.5):
  Target:  y*_i,t = Normalized cross-sectional rank of Y_{i,t} over U_t^valid
  Loss:    ListMLE ranking loss over score vector y_hat_t restricted to U_t^valid
  │
Portfolio Construction (§0.6):
  Long top decile Q1 (+1/k), Short bottom decile Q10 (-1/k), 10 bps round-trip cost model
```

---

## Appendix B: Dimensional Analysis

| Feature | Raw Unit | After Transformation |
|---------|----------|----------------------|
| `r_cc`, `r_on`, `r_id` | Dimensionless (log return) | $\sim N(0, 1)$ |
| `range` | Dimensionless (ratio) | $\sim N(0, 1)$ |
| `close_loc` | Dimensionless $\in [0, 1]$ | $\sim N(0, 1)$ |
| `rel_vol` | Dimensionless (log ratio) | $\sim N(0, 1)$ |
| `ret_Nd` | Dimensionless (cumulative return) | $\sim N(0, 1)$ |
| `vol_Nd` | Dimensionless (return standard deviation) | $\sim N(0, 1)$ |
| `vol_ratio` | Dimensionless (variance ratio) | $\sim N(0, 1)$ |
| `price_z_20` | Dimensionless (log-ratio / $h$-day volatility) | $\sim N(0, 1)$ |
| `drawdown_20` | Dimensionless (log drawdown $\le 0$) | $\sim N(0, 1)$ |
| `eff_ratio_20` | Dimensionless $\in [0, 1]$ | $\sim N(0, 1)$ |
| `vol_shock_5` | Dimensionless (smoothed log ratio) | $\sim N(0, 1)$ |
| `resid_ret` | Dimensionless (residual return) | $\sim N(0, 1)$ |
| `resid_mom_20` | Dimensionless (cumulative residual) | $\sim N(0, 1)$ |
| `resid_vol_20` | Dimensionless (residual volatility) | $\sim N(0, 1)$ |
| Context M1–M6 | Dimensionless (factor decimal returns) | $\sim N(0, 1)$ via rolling 252d z-score |
| Context M7–M11 | Dimensionless (volatilities, fractions, ratios) | $\sim N(0, 1)$ via rolling 252d z-score |

All model inputs and features are strictly scale-free and dimensionless.

---

**END OF v1.3 FEATURE SPECIFICATION**

```
STATUS: FROZEN
SPECIFICATION_ID: v1.3-2026-10-03
CORE_STOCK_FEATURES: 19
EXTENDED_FEATURES: 13
CONTEXT_FEATURES: 11
MAX_LOOKBACK: 332 trading sessions
FIRST_VALID_DATE: ~2001-10
LAST_VALID_DATE: 2026-07-31
```
