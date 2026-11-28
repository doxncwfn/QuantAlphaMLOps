# Diagnosis Report 1.1: Validation vs. Test IC Time Series Gap

**Author:** Quantitative Research & MLOps Team  
**Date:** 2026-11-08  
**Experiment Reference:** Baseline 1A Multi-GPU Diagnostic Audit  
**Authoritative Artifact:** `data/cache/full_diagnosis_results.json`

---

## 1. Executive Summary

This diagnostic evaluates the divergence between Validation Rank Information Coefficient (IC) and Out-of-Sample Test Rank IC across the four expanding walk-forward folds of the Russell 1000 benchmark pipeline.

A severe performance compression was observed in Folds 1–3, where Validation IC exceeded Test IC by 3.4× to 51.6×. In contrast, Fold 4 exhibited an inverse dynamic, where Test IC (+0.0395) outperformed Validation IC (+0.0207) by nearly 2×.

```
Fold 1 (2006–2011): Val IC = +0.0393  -->  Test IC = +0.0114   (Ratio:  3.44x)
Fold 2 (2011–2016): Val IC = +0.0331  -->  Test IC = +0.0006   (Ratio: 51.64x)
Fold 3 (2016–2021): Val IC = +0.0233  -->  Test IC = +0.0036   (Ratio:  6.39x)
Fold 4 (2021–2026): Val IC = +0.0207  -->  Test IC = +0.0395   (Ratio:  0.52x)
```

The primary mechanisms driving this gap are:
1. **Validation Window Asymmetry & Sample Duration:** Validation windows are relatively brief (185 to 745 trading days) compared to the 1,245-day (5-year) test periods. Checkpoint selection (`argmax val_ic`) easily captures transient market regimes in short validation sets that fail over multi-year out-of-sample horizons.
2. **Structural Macroeconomic Regime Shifts:** Folds 1–3 evaluate across profound structural regime transitions (the 2008 Great Financial Crisis, the 2012–2016 quantitative easing zero-rate regime, and the 2020 pandemic dislocation) that differed markedly from their training and validation priors.
3. **Training Sample Depth in Expanding Windows:** Fold 4 trained on 4,217 dates spanning 16+ years of market history across multiple cycles. This expanded training depth enabled the architecture to generalize robustly into the 2021–2026 high-inflation / rate-hiking regime.

---

## 2. Quantitative Summary Statistics

For each fold, the per-date cross-sectional Spearman Rank IC series $IC_t$ was evaluated across both the validation split and the out-of-sample test split.

| Fold | Split | Dates Count | Date Range | Mean IC | Std IC | Median IC | Lag-1 Autocorr | Val $\to$ Test Ratio |
|---|---|---|---|---|---|---|---|---|
| **Fold 1** | Validation | 185 | 2005-12-30 to 2006-09-25 | **+0.0393** | 0.1689 | +0.0082 | 0.8152 | **3.44×** |
| | Test | 1,245 | 2006-10-09 to 2011-09-16 | **+0.0114** | 0.2083 | +0.0008 | 0.8017 | |
| **Fold 2** | Validation | 372 | 2010-03-17 to 2011-09-02 | **+0.0331** | 0.2371 | +0.0272 | 0.5115 | **51.64×** |
| | Test | 1,245 | 2011-09-19 to 2016-08-29 | **+0.0006** | 0.2167 | +0.0010 | 0.5140 | |
| **Fold 3** | Validation | 558 | 2014-06-02 to 2016-08-16 | **+0.0233** | 0.2074 | +0.0247 | 0.7889 | **6.39×** |
| | Test | 1,245 | 2016-08-30 to 2021-08-10 | **+0.0036** | 0.1989 | +0.0055 | 0.7856 | |
| **Fold 4** | Validation | 745 | 2018-08-13 to 2021-07-28 | **+0.0207** | 0.2458 | +0.0225 | 0.6490 | **0.52×** |
| | Test | 1,245 | 2021-08-11 to 2026-07-28 | **+0.0395** | 0.2341 | +0.0430 | 0.5544 | |

### Key Observations
- **High Autocorrelation in Daily IC:** In Folds 1, 3, and 4, the lag-1 autocorrelation of daily Rank IC ranges from 0.55 to 0.81. Because the predictive target is a 5-day cumulative forward return ($H=5$), consecutive daily returns overlap by 4 days, inducing high serial correlation in performance.
- **Dispersion of Daily Performance:** Across all folds, the daily standard deviation of Rank IC is approximately 0.20–0.24. This confirms that daily signal quality is inherently volatile and sensitive to cross-sectional return dispersion.

---

## 3. Detailed Per-Fold Dissection

### Fold 1: Pre-GFC Validation to Crisis Test Horizon (2006–2011)
- **Validation Period:** 185 dates (2005-12 to 2006-09). A calm, late-cycle expansion regime characterized by moderate market momentum and low cross-sectional volatility. Validation IC reached **+0.0393** at Epoch 5.
- **Test Period:** 1,245 dates (2006-10 to 2011-09). Encompasses the 2007 quantitative equity collapse ("Quant Meltdown"), the 2008 Lehman bankruptcy, the 2009 market bottom, and the initial recovery.
- **Failure Mode:** The model checkpoint selected at Epoch 5 overfitted to the calm pre-crisis environment. While the model retained positive mean IC (+0.0114), during the extreme deleveraging of 2007–2008, standard linear factor relationships inverted, causing significant drawdowns (-78.9% max drawdown) and dragging down the multi-year average.

### Fold 2: Post-Crisis Recovery Validation to Zero-Rate QE Test Horizon (2011–2016)
- **Validation Period:** 372 dates (2010-03 to 2011-09). Marked by high market volatility, sovereign debt concerns in Europe, and large cross-sectional price dispersions. The model achieved a peak Validation IC of **+0.0331** at Epoch 18.
- **Test Period:** 1,245 dates (2011-09 to 2016-08). A continuous 5-year secular bull market under quantitative easing (QE), with historically low volatility, depressed equity dispersion, and heavy central-bank liquidity suppression.
- **Failure Mode:** Total collapse of signal efficacy (Test IC dropped to **+0.0006**, ratio 51.64×). The model relied heavily on volatility and momentum signals learned during the volatile 2001–2010 training period. In a flat, low-dispersion regime, these signals generated near-zero predictive power.

### Fold 3: Mid-Decade Validation to COVID Dislocation Test Horizon (2016–2021)
- **Validation Period:** 558 dates (2014-06 to 2016-08). Transition from commodities drop into a low-volatility consolidation. Peak Validation IC was **+0.0233** at Epoch 5.
- **Test Period:** 1,245 dates (2016-08 to 2021-08). Covers the 2017 ultra-low vol grind, the 2018 trade war sell-off, and the violent March 2020 COVID shock and subsequent retail-driven tech rally.
- **Failure Mode:** The model stopped training early (Epoch 5), locking in parameters fitted to low-complexity features. It produced an anemic Test IC of **+0.0036** (ratio 6.39×).

### Fold 4: The Inverted Ratio — Why Post-COVID Generalizes Robustly (2021–2026)
- **Validation Period:** 745 dates (2018-08 to 2021-07). Spanned the turbulent trade war and the 2020 pandemic dislocation. Validation IC averaged **+0.0207**.
- **Test Period:** 1,245 dates (2021-08 to 2026-07). Characterized by high inflation, federal funds rate increases from 0% to >5%, valuation repricing, and high single-stock cross-sectional dispersion.
- **Efficacy Breakthrough:** Test IC surged to **+0.0395** (Ratio 0.52×). 
  - **Reason 1 (Data Volume):** Fold 4 benefited from 4,217 training dates (16+ years), allowing the transformer/LSTM attention architecture to learn stable multi-regime representations.
  - **Reason 2 (Regime Congruence):** Post-2021 markets featured high dispersion and distinct winner/loser separation across equities, creating optimal conditions for cross-sectional ranking architectures.

---

## 4. Root Causes & Actionable Directives

1. **Short Validation Slices Induce Premature Stopping:**  
   In Folds 1 and 3, validation sets of 185 and 558 days triggered early stopping at Epoch 5. The model never converged to multi-epoch representations.
   - *Directive:* Evaluate regularizing early stopping (e.g., minimum warm-up of 10 epochs or early-stopping based on trailing multi-epoch validation averages).
2. **Signal Decay Under Low Volatility:**  
   The model requires cross-sectional dispersion. When market volatility falls into Q1, signal Rank IC flips negative (-0.0127 in Fold 1, -0.0125 in Fold 2).
   - *Directive:* Implement regime gating or adaptive volatility weighting to protect capital during low-dispersion regimes.
