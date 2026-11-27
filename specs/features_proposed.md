# FEATURES_BASELINE_PROPOSED_v1.0

**Status:** PROPOSED (pending ablation validation)
**Date:** 2026-10-11
**Supersedes:** Nothing (proposal, not frozen)
**Parent specification:** `v1.3_feature_specification.md` (FROZEN)
**Target:** Compact ~20-feature representation of the frozen v1.3 feature space for the LSTM–Transformer cross-sectional ranking model.

---

## 0. Preamble and Scope

### 0.1 What this document is

This is a **proposal** to reduce the frozen v1.3 stock-level feature set from 19 to 17 features, and the context feature set from 11 to 3 features, for a total of 20 model inputs. The reduction is achieved by identifying **mathematically derivable features** that can be reconstructed inside the first linear layer of the model without information loss.

This document:
- Preserves all v1.3 definitions, formulas, transforms, missing-data policies, and causality guarantees.
- Identifies the specific features that are excluded, with a proof of derivability.
- Specifies the exact feature tensor that feeds the model.
- Specifies the ablation protocol to verify that the compact set retains ≥ 98% of baseline net Sharpe.

### 0.2 What this document is not

This document is **not** a redesign of the feature engineering. It is a **compaction** of the frozen v1.3 specification. It does not introduce new features, does not modify formulas, and does not change the loss, architecture, or evaluation protocol.

### 0.3 Relationship to v1.3

| Layer | v1.3 count | Proposed count | Change |
|---|---|---|---|
| Layer 1 — Primitive | 6 | 5 | `r_id` dropped (derivable) |
| Layer 2 — Temporal | 10 | 9 | `vol_60d` dropped (derivable) |
| Layer 4 — Factor residual | 3 | 3 | unchanged |
| **Stock-level total** | **19** | **17** | **−2** |
| Layer 5 — Context | 11 | 3 | 8 dropped (derivable or redundant) |
| **Model input total** | **30** | **20** | **−10** |

---

## 1. Notation

All quantities are defined at session \(s\). We write \(C_s, O_s, H_s, L_s, V_s\) for split-adjusted close, open, high, low, and raw volume at session \(s\). \(r_s = \log(C_s/C_{s-1})\). \(U_s\) is the point-in-time Russell 1000 common-stock universe at \(s\). \(N_s = |U_s|\).

| Symbol | Definition |
|---|---|
| \(C_s\) | Split-adjusted close: CRSP \(\|PRC_s\| / CFACPR_s\); Yahoo \(Close_s\) |
| \(O_s\) | Split-adjusted open: CRSP \(\|OPENPRC_s\| / CFACPR_s\); Yahoo \(Open_s\) |
| \(H_s\) | Split-adjusted high: CRSP \(ASKHI_s / CFACPR_s\); Yahoo \(High_s\) |
| \(L_s\) | Split-adjusted low: CRSP \(BIDLO_s / CFACPR_s\); Yahoo \(Low_s\) |
| \(V_s\) | Raw volume: CRSP \(VOL_s\); Yahoo \(Volume_s\) |
| \(r_s\) | Daily log return: CRSP \(\log(1+RETX_s)\); Yahoo \(\log(C_s/C_{s-1})\) |
| \(R_s\) | Total return: CRSP \(RET_s\); Yahoo \(AdjClose_s/AdjClose_{s-1} - 1\) |
| \(RF_s\) | Daily risk-free rate from `silver.market_factor.rf` |
| \(R^e_s\) | Excess return: \(\log(1+R_s) - RF_s\) |
| \(F_s\) | Six-factor vector \([MKT\_RF, SMB, HML, RMW, CMA, UMD]_s^\top\) |
| \(\text{Med}_{h}(X)_s\) | Median of \(X\) over \([s-h+1, s]\) |
| \(\text{SMA}_{h,s}\) | Geometric mean of \(C\) over \([s-h+1, s]\): \(\exp\left(\frac{1}{h}\sum_{j=0}^{h-1}\log C_{s-j}\right)\) |
| \(\text{NS}_s(\cdot)\) | Cross-sectional Blom normal-score transform across \(U_s\) |
| \(U_s^{\text{valid}}\) | Subset of \(U_s\) with non-NaN target |

---

## 2. Feature Tensor Definitions

### 2.1 Primitive Layer — 5 features

| # | Name | Definition | Lookback | Bounds | Leakage |
|:-:|---|---|:-:|:-:|:-:|
| 1 | `r_cc` | \(r_{cc,s} = \log(C_s / C_{s-1})\) | 1 | none | NONE |
| 2 | `r_on` | \(r_{on,s} = \log(O_s / C_{s-1})\) | 1 | none | NONE |
| 3 | `range` | \(R_s = (H_s - L_s) / C_s\) | 0 | \(R_s \geq 0\) | NONE |
| 4 | `close_loc` | \(CL_s = (C_s - L_s) / (H_s - L_s)\) | 0 | \(CL_s \in [0,1]\) | NONE |
| 5 | `rel_vol` | \(\rho_s = \log\left(V_s / \text{Med}_{20}(V)_s\right)\), clipped to \([-5, 5]\) | 20 | \([-5, 5]\) | LOW (20d median is causal) |

**Edge cases:**
- `r_cc`: NaN if `RETX` string-coded in CRSP, or first observation per stock.
- `r_on`: NaN if `OPENPRC` null (~0.5–1.4% CRSP).
- `range`: NaN if any OHLC null.
- `close_loc`: NaN if \(H_s = L_s\) (~0.07% CRSP).
- `rel_vol`: NaN if \(V_s = 0\) or warmup < 20.

**Dropped relative to v1.3:** `r_id`. See §4.

### 2.2 Temporal Layer — 9 features

| # | Name | Definition | Lookback | Bounds | Leakage |
|:-:|---|---|:-:|:-:|:-:|
| 6 | `ret_5d` | \(\text{R}_{5,s} = \sum_{j=0}^{4} r_{s-j}\) | 5 | none | NONE |
| 7 | `ret_20d` | \(\text{R}_{20,s} = \sum_{j=0}^{19} r_{s-j}\) | 20 | none | NONE |
| 8 | `ret_60d` | \(\text{R}_{60,s} = \sum_{j=0}^{59} r_{s-j}\) | 60 | none | NONE |
| 9 | `vol_20d` | \(\sigma_{20,s} = \sqrt{\frac{1}{19}\sum_{j=0}^{19}(r_{s-j} - \bar r_{20,s})^2}\) | 20 | \(\geq 0\) | NONE |
| 10 | `vol_ratio` | \(\text{VR}_s = \sigma_{20,s} / \sigma_{60,s}\) | 60 | \(\geq 0\) | NONE |
| 11 | `price_z_20` | \(\text{PZ}_{20,s} = \log(C_s / \text{SMA}_{20,s}) \,/\, (\sigma_{20,s} \cdot \sqrt{20})\) | 20 | none | LOW (split-guard) |
| 12 | `drawdown_20` | \(\text{DD}_{20,s} = \log\left(C_s / \max_{j \in [0,19]} C_{s-j}\right)\) | 20 | \(\leq 0\) | LOW (split-guard) |
| 13 | `eff_ratio_20` | \(\text{ER}_{20,s} = |\text{R}_{20,s}| / \sum_{j=0}^{19}|r_{s-j}|\) | 20 | \([0,1]\) | NONE |
| 14 | `vol_shock_5` | \(\text{VS}_{5,s} = \frac{1}{5}\sum_{j=0}^{4} \rho_{s-j}\) | 24 | none | NONE |

**Internal computation for `vol_ratio`:** \(\sigma_{60,s}\) is computed internally as
\[
\sigma_{60,s} = \sqrt{\frac{1}{59}\sum_{j=0}^{59}(r_{s-j} - \bar r_{60,s})^2}
\]
but **is not exposed as a model input**. Only the ratio is fed to the model.

**Dropped relative to v1.3:** `vol_60d`. See §4.

**Split-contamination rule (v1.3 §1.3):** For `price_z_20` and `drawdown_20`, if `CFACPR` changes within \([s-19, s]\), the feature is NaN.

**Edge cases (from v1.3):**
- `ret_20d`: NaN if >4 missing `r_cc` values in the 20d window.
- `ret_60d`: NaN if >12 missing `r_cc` values in the 60d window.
- `vol_20d`: NaN if <10 valid returns.
- `vol_ratio`: NaN if either numerator or denominator is NaN.
- `price_z_20`, `drawdown_20`: NaN if <10 valid prices or split-contamination.
- `eff_ratio_20`: NaN if denominator is zero.
- `vol_shock_5`: NaN if >2 of 5 `rel_vol` values are NaN.

### 2.3 Factor Residual Layer — 3 features

**Six-factor OLS, strictly lagged.**

For each session \(s\), define the estimation window \(\mathcal{W}_s = [s-252, s-1]\) (252 sessions, ending strictly at \(s-1\)). Let
\[
\hat\alpha_{i,s-1}, \hat\beta_{i,s-1}
= \arg\min_{\alpha, \beta} \sum_{\tau \in \mathcal{W}_s} \left(R^e_{i,\tau} - \alpha - \beta^\top F_\tau\right)^2
\]
with the conditioning guard: if \(\text{cond}([1, F_{\tau \in \mathcal{W}_s}]) > 10^{10}\), or if \(|\mathcal{W}_s \cap \text{valid}| < 126\), the estimate is undefined (all residual features are NaN).

| # | Name | Definition | Span | Req. History | Bounds | Leakage |
|:-:|---|---|:-:|:-:|:-:|:-:|
| 15 | `resid_ret` | \(\epsilon_s = R^e_s - \hat\alpha_{i,s-1} - \hat\beta_{i,s-1}^\top F_s\) | 253 | 313 | none | **NONE** (lagged β) |
| 16 | `resid_mom_20` | \(\text{RM}_{20,s} = \sum_{j=0}^{19} \epsilon_{s-j}\) | 272 | 332 | none | NONE |
| 17 | `resid_vol_20` | \(\text{RV}_{20,s} = \text{std}(\epsilon_{s-19}, \ldots, \epsilon_s)\) | 272 | 332 | \(\geq 0\) | NONE |

**Causality guarantees (from v1.3 §6.2):**
1. \(\hat\beta_{i,s-1}\) is estimated on \([\tau : \tau \leq s-1]\). No data from session \(s\) enters the estimation.
2. \(\epsilon_s\) uses \(\hat\beta_{i,s-1}\) applied to \(F_s\). One-step-ahead innovation.
3. Historical residuals are **stored as-of \(s\)** and never recomputed. `resid_mom_20` and `resid_vol_20` sum over the stored historical residuals.

**Edge cases:**
- Insufficient history (< 126 valid obs): NaN.
- Multicollinearity (cond > 10¹⁰): NaN.
- Missing factor data: NaN.

### 2.4 Context Layer — 3 features

All context features are computed once per session \(s\) (shared across all stocks) and standardized with a **strictly causal rolling 252-session z-score** (v1.3 §1.5).

| # | Name | Raw definition | Lookback | Transform |
|:-:|---|---|:-:|---|
| 18 | `ctx_mkt_vol_20` | \(m^{\text{vol}}_s = \text{std}(MKT\_RF_{s-19}, \ldots, MKT\_RF_s)\) | 20 | rolling 252d causal z-score |
| 19 | `ctx_mkt_ret_20` | \(m^{\text{ret}}_s = \sum_{j=0}^{19} MKT\_RF_{s-j}\) | 20 | rolling 252d causal z-score |
| 20 | `ctx_dispersion` | \(d_s = \text{std}_{i \in U_s}(r_{cc,i,s})\) | 0 | rolling 252d causal z-score |

**Rolling causal z-score definition:**
\[
\tilde{x}_s = \frac{x_s - \bar x_{[s-251, s-1]}}{\sigma_{x, [s-251, s-1]}}
\]
where the window is strictly ending at \(s-1\). If fewer than 60 observations exist in \([s-251, s-1]\), the feature is NaN.

**Note on `ctx_dispersion`:** computed from the **raw** daily log returns \(r_{cc,i,s}\) across \(U_s\), **before** any cross-sectional transformation (v1.3 §1.5). Cross-sectionally normal-scored returns have standard deviation 1 by construction and would render `ctx_dispersion` degenerate.

**Dropped relative to v1.3:** `ctx_mktrf`, `ctx_smb`, `ctx_hml`, `ctx_rmw`, `ctx_cma`, `ctx_umd`, `ctx_breadth`, `ctx_factor_disp`. See §4.

---

## 3. Transform Pipeline

### 3.1 Stock-level cross-sectional transform

At each session \(s\), the 17 stock-level features are transformed with **Blom normal scores** across \(U_s\):

\[
\tilde{x}_{i,s,f} = \Phi^{-1}\!\left(\frac{\text{rank}_{i,s,f} - 0.375}{N_s^{\text{valid},f} + 0.25}\right), \quad \forall f \in \{1,\ldots,17\}
\]

**Guards:**
- If \(N_s^{\text{valid},f} < 100\), skip the date for feature \(f\) (NaN).
- If the cross-sectional std \(< 10^{-10}\), return 0 for all valid stocks.
- Stocks with NaN in feature \(f\) at \(s\) are excluded from the ranking and receive NaN normal scores (v1.3 §7.2).

### 3.2 Context transform

Rolling causal z-score per §2.4.

### 3.3 Feature tensor for the model

For a stock \(i\) at prediction date \(t\), the model input is:

\[
\mathbf{X}_i^{(t)} = \left[\tilde{x}_{i,s,1}, \ldots, \tilde{x}_{i,s,17}\right]_{s \in [t-59, t]} \in \mathbb{R}^{60 \times 17}
\]

The context branch input is:

\[
\mathbf{C}^{(t)} = \left[\tilde{c}_{s,1}, \tilde{c}_{s,2}, \tilde{c}_{s,3}\right]_{s \in [t-59, t]} \in \mathbb{R}^{60 \times 3}
\]

Both feed the architecture Y model (v1.3 §8).

---

## 4. Dropped Features — Derivability Proofs

### 4.1 `r_id` (Layer 1, position 3 in v1.3)

**v1.3 definition:** \(r_{id,s} = \log(C_s / O_s)\).

**Derivability:** By construction,
\[
r_{cc,s} = \log(C_s / C_{s-1}) = \log(C_s / O_s) + \log(O_s / C_{s-1}) = r_{id,s} + r_{on,s}
\]
Therefore,
\[
\boxed{\, r_{id,s} = r_{cc,s} - r_{on,s} \,}
\]

**Proof of non-loss:** Given \(r_{cc}\) and \(r_{on}\) in the input tensor, the first linear layer \(\mathbf{W}\) can construct \(r_{id}\) via \(\mathbf{W}_{:,1} = (1, 0, \ldots, 0)\) and \(\mathbf{W}_{:,2} = (-1, 0, \ldots, 0)\) with the third column zero. The information content is preserved exactly.

**Empirical support (Round 9 permutation importance):** `r_id` contributes 20.4% of test IC. `r_on` (44.4%) and `r_cc` (26.9%) together cover 71.3%. The joint contribution of `r_on + r_cc` exceeds `r_id` alone.

**Caveat:** the first layer must learn the subtraction. This requires a small amount of training signal. With 5.3M rows and standard gradient descent, the subtraction is learned within a few epochs.

### 4.2 `vol_60d` (Layer 2, position 11 in v1.3)

**v1.3 definition:** \(\sigma_{60,s}\) (60-day realized volatility).

**Derivability:** By definition of `vol_ratio`,
\[
\text{VR}_s = \frac{\sigma_{20,s}}{\sigma_{60,s}} \quad \Rightarrow \quad \sigma_{60,s} = \frac{\sigma_{20,s}}{\text{VR}_s}
\]
Given `vol_20d` and `vol_ratio` in the input tensor,
\[
\boxed{\, \sigma_{60,s} = \sigma_{20,s} / \text{VR}_s \,}
\]

**Note:** `vol_60d` is still **computed internally** to derive `vol_ratio` (§2.2). It is not exposed as an input.

**Empirical support:** `vol_60d` contributes 14.8% of IC, `vol_ratio` contributes 13.9%. Combined 28.7%. The pair covers substantially the same information as `vol_20d` + `vol_ratio` alone would.

### 4.3 Context features `ctx_mktrf`, `ctx_smb`, `ctx_hml`, `ctx_rmw`, `ctx_cma`, `ctx_umd` (positions M1–M6 in v1.3)

**Argument for dropping:**

1. **The factor returns are already in the residualization.** Layer 4 uses the six FF factors to define the factor model. The residual return `resid_ret`, residual momentum `resid_mom_20`, and residual volatility `resid_vol_20` are all constructed by removing the factor component. The model therefore already has access to the **effect** of these factors via the residual features.

2. **Factor timing has been repeatedly falsified in the literature.** Using factor returns as a conditioning signal for cross-sectional stock selection is a factor-timing overlay. Long-horizon factor-timing evidence (Asness et al., 2017) shows weak and unstable predictability. For a 5-day horizon, the signal-to-noise ratio is worse.

3. **The interaction between stock features and factor returns can be learned by the model.** The cross-sectional Transformer can learn that a stock's `resid_ret` is especially informative when `ctx_smb` is positive if it had access to the factor — but the residual already encodes this in a factor-neutral manner.

**Empirical support:** The context branch was not independently ablated in Round 9. However, `ctx_mkt_vol_20` and `ctx_mkt_ret_20` are the ones the model uses most (per regime analysis: `ctx_mkt_vol_20` quartiles drive 76.7% of the alpha concentration). The other FF factors have no documented regime-conditional contribution.

**Caveat:** if a subsequent ablation shows that adding `ctx_umd` alone (the factor with the strongest cross-sectional correlation to the target, as UMD is a return-based factor rather than a risk factor) improves net Sharpe by ≥ 0.02, add it back as feature #21.

### 4.4 `ctx_breadth` (position M9 in v1.3)

**Argument for dropping:**

`ctx_breadth` = fraction of stocks with positive returns. `ctx_dispersion` = cross-sectional std of returns. These are highly correlated in normal regimes (both measure dispersion of returns).

**Empirical check required:** the ablation must verify that keeping `ctx_dispersion` alone retains the regime signal that `ctx_breadth` provides. If not, re-add.

### 4.5 `ctx_factor_disp` (position M11 in v1.3)

**Argument for dropping:**

Cross-sectional dispersion of the 5 style factors. This is a second-order moment-of-moments feature. No prior in the literature supports its predictive power for cross-sectional stock ranking at a 5-day horizon.

**Empirical support:** no ablation in the project so far. Absence of evidence is not evidence of absence, but the theoretical prior is weak and the feature count budget is tight.

---

## 5. Coverage Matrix

| Dimension | Coverage | Contributing features |
|---|---|---|
| Momentum (short) | Strong | `ret_5d`, `resid_mom_20` |
| Momentum (medium) | Strong | `ret_20d`, `resid_mom_20` |
| Momentum (long) | Moderate | `ret_60d` |
| Price behavior | Strong | `r_cc`, `r_on`, `close_loc`, `price_z_20`, `range` |
| Trend | Strong | `eff_ratio_20`, `drawdown_20`, `price_z_20` |
| Mean reversion | Moderate | `price_z_20`, `close_loc`, `ret_5d` |
| Volatility | Strong | `range`, `vol_20d`, `vol_ratio`, `resid_vol_20`, `ctx_mkt_vol_20` |
| Market regime | Moderate | `vol_ratio`, `ctx_mkt_vol_20`, `ctx_mkt_ret_20`, `ctx_dispersion` |
| Liquidity / volume | Moderate | `rel_vol`, `vol_shock_5` |
| Market context | Moderate | `ctx_mkt_vol_20`, `ctx_mkt_ret_20`, `ctx_dispersion` |
| Idiosyncratic alpha | Strong | `resid_ret`, `resid_mom_20`, `resid_vol_20` |

**Underrepresented dimensions:**
- **Fundamental quality / growth.** Not present. Fundamental features are the subject of the pending F1–F3 experiments.
- **Long-horizon momentum (>60d).** Excluded by the 332-session lookback budget.
- **Sector / industry.** Not available consistently across CRSP and Yahoo eras.

---

## 6. Missing-Data and Null-Propagation Policy

### 6.1 Per-feature NaN policy

Every NaN in a stock-level feature is propagated to the cross-sectional transform: the stock is excluded from the ranking for that feature at session \(s\), and receives a NaN normal-score.

The model is trained with a **loss mask** that ignores any stock with NaN in any of the 17 stock-level features. This is per v1.3 §7.2.

### 6.2 Null-propagation rule

If a parent feature is NaN, all child features that depend on it are NaN. The dependency chain (from v1.3 §1.6) is preserved. In particular:
- `vol_ratio` is NaN if either `vol_20d` or internal `vol_60d` is NaN.
- `vol_shock_5` is NaN if >2 of its 5 `rel_vol` values are NaN.
- `resid_mom_20` and `resid_vol_20` are NaN if >4 of their 20 `resid_ret` values are NaN.

### 6.3 Warmup policy

Required history per feature (from v1.3 §1.1):
\[
\text{Req. History} = 60 + \max(\text{lookback})
\]
The maximum across the 17 stock features is 332 sessions (for `resid_mom_20` and `resid_vol_20`, requiring 252 for β estimation + 20 for momentum + 60 for the model window). Dates before the 332nd session per stock are excluded from the feature tensor and from the loss.

### 6.4 Cross-sectional minimum

If \(|U_s^{\text{valid}}| < 100\), the date is skipped for training and evaluation (v1.3 §7.2).

---

## 7. Causality and Leakage Guarantees

| Risk | Mitigation |
|---|---|
| Look-ahead in feature computation | All trailing windows end at \(s\). No centered or expanding-into-future windows. |
| Look-ahead in β estimation | β estimated on \([s-252, s-1]\), strictly lagged. |
| Look-ahead in cross-sectional transform | Blom ranks computed on \(\{x_{i,s}\}_{i \in U_s}\) only, using no future information. |
| Look-ahead in context z-score | Rolling window \([s-251, s-1]\), strictly ending at \(s-1\). |
| Padding-induced leakage | \(N_{\max}^{\text{fixed}} = 1{,}200\) is a design constant, not derived from data. Bitwise-invariant attention masks. Verified by `test_cross_section_padding.py`. |
| Survivorship bias | PiT Russell 1000 membership from `silver.russell_membership`, effective-date convention, backfilled from CRSP history. |
| Delisting bias | Delisting policy per v1.3 §10.4. Targets are NaN for unresolved delistings. |
| Corporate-action contamination | Returns computed via `RETX` in CRSP, `Close` ratios in Yahoo. Split-contamination guard on price features. |
| Determinism | `seed_everything(42, deterministic=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, math SDPA forced. |

**No change to causality relative to v1.3.** The dropped features (`r_id`, `vol_60d`, six FF factors, `ctx_breadth`, `ctx_factor_disp`) were all causal in v1.3. Their removal does not introduce leakage.

---

## 8. Model Input Specification

| Component | Shape | Data type | Notes |
|---|---|---|---|
| Stock tensor | \((B, N_{\max}^{\text{fixed}}, 60, 17)\) | `float32` | \(B\) = batch of dates, \(N_{\max}^{\text{fixed}} = 1{,}200\) |
| Stock mask | \((B, N_{\max}^{\text{fixed}})\) | `bool` | True for valid stocks |
| Context tensor | \((B, 60, 3)\) | `float32` | Shared across \(N_s\) |
| Target | \((B, N_{\max}^{\text{fixed}})\) | `float32` | \(y^* \in [-0.5, 0.5]\) |
| Target mask | \((B, N_{\max}^{\text{fixed}})\) | `bool` | True for valid targets |

The architecture is unchanged from v1.3 §8 (Architecture Y).

---

## 9. Ablation Protocol

### 9.1 Baselines

| ID | Feature set | Expected net Sharpe |
|---|---|---|
| **B0** | Full v1.3 (30 features: 19 stock + 11 context) | +0.169 (frozen baseline, Round 9) |
| **B1** | 29 features (B0 − `r_id`) | ~+0.165 |
| **B2** | 28 features (B1 − `vol_60d`) | ~+0.160 |
| **B3** | 20 features (B2 − 8 context features per §4.3–4.5) | **proposed** |
| **B4** | 22 features (B2 − 6 context features; keep `ctx_umd`, `ctx_breadth`) | alternative |

### 9.2 Ablation sequence

Run in this order to isolate each drop:

1. B0 → B1: Drop `r_id`. If Δnet Sharpe > −0.02, adopt.
2. B1 → B2: Drop `vol_60d`. If Δnet Sharpe > −0.02 (cumulative), adopt.
3. B2 → B3: Drop 8 context features. If Δnet Sharpe > −0.02 (cumulative), adopt B3.
4. If B3 fails, try B4 (keep `ctx_umd` and `ctx_breadth`). If B4 passes, adopt B4.

### 9.3 Metrics

**Primary:** Net LS Sharpe (HAC-adjusted, \(c = 5\) bps one-way).
**Secondary:** Rank IC, IC_IR, top-decile precision, bottom-decile precision, gross spread, turnover, max drawdown.
**Regime diagnostic:** net Sharpe per VIX quartile (from `ctx_mkt_vol_20`).

### 9.4 Acceptance criteria

The proposed compact set is accepted if **all** of:

1. **Cumulative net Sharpe ≥ B0 net Sharpe − 0.02** (i.e., ≥ +0.149).
2. **All 4 folds show net Sharpe ≥ −0.05** (no catastrophic fold collapse).
3. **Rank IC ≥ B0 Rank IC − 0.002** (i.e., ≥ +0.0197).
4. **Top-decile precision ≥ B0 top precision − 0.005** (i.e., ≥ 0.0858).
5. **Bottom-decile precision ≥ B0 bottom precision − 0.005** (i.e., ≥ 0.1855).

If any criterion fails, the compact set is rejected and B0 remains the production baseline.

### 9.5 Protocol

Same 5-fold NF-WFCV as Round 9 (purge 5, embargo 5, 4,928 test sessions). Same deterministic harness.

### 9.6 Do not

- Do not use single-fold results to select features.
- Do not use in-sample permutation importance to justify removal. Use out-of-sample net Sharpe.
- Do not add new features during this ablation. This is a **reduction** test only.
- Do not change the model architecture, loss, or training hyperparameters.

---

## 10. Deliverables

1. `docs/features_baseline_v1.md` — this document, updated with results after ablation.
2. `data/ablation_results/B{0-4}_{metric}.json` — the ablation results.
3. `docs/features_baseline_ablation.md` — the ablation report with all metrics, per-fold, per-regime.
4. Implementation: `src/data/feature_sets.py` — a registry mapping `{B0, B1, B2, B3, B4}` to their feature column lists.

---

## 11. Status

```
FEATURE SET: FEATURES_BASELINE_PROPOSED_v1.0
STATUS:      PROPOSED (pending ablation)
STOCK FEATURES:      17 (down from 19)
CONTEXT FEATURES:     3 (down from 11)
TOTAL INPUT FEATURES: 20 (down from 30)
MAX LOOKBACK:       332 sessions
FIRST VALID DATE:   ~2001-10
LAST VALID DATE:    2026-07-31 (factor data boundary)
```

**Freeze condition:** this document becomes FROZEN when the ablation in §9 confirms that the compact set meets all acceptance criteria in §9.4.

**Rejection condition:** if any acceptance criterion fails, the set reverts to `FEATURES_BASELINE_v1.0` (the frozen v1.3 30-feature set).

**Next action:** execute the ablation in §9 on Modal A10G. Report net Sharpe, Rank IC, top/bottom precision per fold, plus VIX-quartile breakdown. Compare B0 → B1 → B2 → B3 (and optionally B4). Adopt the largest set that passes all acceptance criteria.

---

**End of FEATURES_BASELINE_PROPOSED_v1.0**