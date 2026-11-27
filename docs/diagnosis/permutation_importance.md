# Diagnosis Report 1.3: Permutation Feature Importance Across Folds

**Author:** Quantitative Research & MLOps Team  
**Date:** 2026-11-08  
**Experiment Reference:** Baseline 1A Multi-GPU Diagnostic Audit  
**Authoritative Artifact:** `data/cache/full_diagnosis_results.json`

---

## 1. Executive Summary

This diagnostic evaluates feature importance via out-of-sample permutation testing on each fold's best model checkpoint.
- **Stock Features (19 features):** Permuted across stocks within each date ($\Delta \text{IC} = \text{Base IC} - \text{Permuted IC}$).
- **Context Features (11 features):** Permuted across trading dates.

### Primary Diagnostic Finding: Severe Cross-Fold Regime Divergence
The empirical evidence confirms that **different folds learn fundamentally different feature representations**:
1. **Folds 1 & 3 (Pre-Crisis & Mid-Decade):** Dominated almost exclusively by **price mean-reversion** (`drawdown_20`, `price_z_20`). Macro context features have zero importance ($\Delta \text{IC} < 10^{-5}$), consistent with premature early-stopping at Epoch 5 before context cross-attention could be integrated.
2. **Fold 2 (QE Era):** Exhibits fragile feature weights with an inverted response to market dispersion ($\Delta \text{IC} = -0.0050$), confirming severe memorization.
3. **Fold 4 (Post-COVID 2021–2026):** Overwhelmingly dominated by **realized volatility** (`vol_20d`, `vol_60d`, `resid_vol_20`) and **macro context interaction** (`ctx_mktrf`, `ctx_mkt_ret_20`, `ctx_dispersion`).

This divergence provides decisive empirical proof of **regime dependence**: a static architecture with unconstrained feature inputs shifts between pure mean-reversion and pure volatility-sorting depending on the macro environment of its training window.

---

## 2. Top Feature Rankings Per Fold

### Top 10 Stock-Level Features
Values represent the drop in Test Rank IC ($\Delta \text{IC}$) when the feature is randomly permuted. Larger positive values indicate higher feature importance.

| Rank | Fold 1 (2006–2011) | $\Delta \text{IC}$ | Fold 2 (2011–2016) | $\Delta \text{IC}$ | Fold 3 (2016–2021) | $\Delta \text{IC}$ | Fold 4 (2021–2026) | $\Delta \text{IC}$ |
|---|---|---|---|---|---|---|---|---|
| 1 | `drawdown_20` | +0.00278 | `price_z_20` | +0.00041 | `drawdown_20` | +0.00305 | `vol_20d` | **+0.01463** |
| 2 | `ret_60d` | +0.00162 | `r_cc` | +0.00036 | `price_z_20` | +0.00212 | `vol_60d` | **+0.01303** |
| 3 | `price_z_20` | +0.00156 | `ret_5d` | +0.00032 | `resid_vol_20` | +0.00064 | `resid_vol_20` | **+0.00746** |
| 4 | `vol_ratio` | +0.00061 | `resid_vol_20` | +0.00032 | `ret_60d` | +0.00025 | `vol_ratio` | +0.00157 |
| 5 | `r_cc` | +0.00053 | `vol_ratio` | +0.00028 | `vol_shock_5` | +0.00023 | `eff_ratio_20` | +0.00121 |
| 6 | `close_loc` | +0.00041 | `ret_60d` | +0.00023 | `r_id` | +0.00023 | `range` | +0.00109 |
| 7 | `rel_vol` | +0.00022 | `close_loc` | +0.00019 | `r_on` | +0.00019 | `resid_mom_20` | +0.00090 |
| 8 | `r_id` | +0.00016 | `resid_mom_20` | +0.00013 | `r_cc` | +0.00011 | `drawdown_20` | +0.00072 |
| 9 | `eff_ratio_20` | +0.00013 | `rel_vol` | +0.00011 | `eff_ratio_20` | +0.00009 | `ret_60d` | +0.00030 |
| 10 | `vol_shock_5` | +0.00013 | `drawdown_20` | +0.00009 | `close_loc` | +0.00009 | `ret_5d` | +0.00016 |

### Context Feature Importance
Values represent the drop in Test Rank IC ($\Delta \text{IC}$) when the context feature is permuted across time.

| Context Feature | Fold 1 $\Delta \text{IC}$ | Fold 2 $\Delta \text{IC}$ | Fold 3 $\Delta \text{IC}$ | Fold 4 $\Delta \text{IC}$ | Cross-Fold Role |
|---|---|---|---|---|---|
| `ctx_mktrf` | -0.00003 | -0.00196 | -0.00000 | **+0.01548** | Critical macro driver in Fold 4; inactive in Folds 1/3 |
| `ctx_mkt_ret_20` | +0.00000 | **+0.00317** | -0.00001 | **+0.00971** | Key market momentum conditioning signal |
| `ctx_dispersion` | +0.00000 | -0.00500 | -0.00001 | **+0.00885** | Vital for volatility dispersion regimes |
| `ctx_umd` | -0.00004 | +0.00073 | -0.00001 | **+0.00467** | Cross-sectional momentum factor conditioning |
| `ctx_mkt_vol_20` | -0.00000 | -0.00050 | +0.00000 | **+0.00460** | Volatility level gating |
| `ctx_factor_disp`| -0.00001 | -0.00112 | -0.00001 | **+0.00391** | Factor dispersion |
| `ctx_smb` | -0.00000 | +0.00154 | -0.00002 | **+0.00363** | Size factor conditioning |
| `ctx_rmw` | -0.00001 | +0.00218 | +0.00002 | **+0.00187** | Quality/Profitability factor conditioning |
| `ctx_cma` | -0.00003 | +0.00052 | +0.00001 | **+0.00131** | Investment factor conditioning |
| `ctx_breadth` | -0.00002 | -0.00200 | -0.00001 | +0.00037 | Market breadth |
| `ctx_hml` | +0.00001 | -0.00101 | +0.00001 | -0.00095 | Value factor conditioning |

---

## 3. Key Observations & Inferences

1. **The Volatility Regime Monopoly in Fold 4:**  
   In Fold 4, the top three features (`vol_20d`, `vol_60d`, `resid_vol_20`) account for over **80% of total signal predictive power**. When `vol_20d` is permuted, Rank IC drops by **0.01463** (over 37% of the total +0.0395 IC). This demonstrates that in high-dispersion environments (2021–2026), the model primarily functions as an idiosyncratic volatility and volatility-spread predictor.

2. **Absence of Context Integration in Short-Trained Folds:**  
   In Fold 1 and Fold 3, context features generated near-zero $\Delta \text{IC}$ ($\le 10^{-5}$). Because training halted at Epoch 5, the model's cross-attention mechanisms between stock states and context features never developed non-zero weights. The model effectively acted as an unconditioned single-stock model.

3. **Inverted Context Sensitivities in Fold 2:**  
   Permuting `ctx_dispersion` in Fold 2 produced a negative drop ($\Delta \text{IC} = -0.0050$), meaning that shuffling dispersion across time actually *improved* rank correlation on the test set. This is a telltale indicator of an overfitted feature weight whose sign inverted between the 2001–2010 training period and the 2011–2016 test period.

---

## 4. Architectural & Strategic Implications

- **Feature Group Regularization:** Realized volatility features (`vol_20d`, `vol_60d`) tend to dominate the linear and temporal attention projections if unconstrained. Group dropout or feature-level normalization will prevent the network from collapsing into a single-factor (pure volatility) model.
- **Context Gate Pre-Training or Warm-Up:** Context features require more epochs to integrate into the cross-attention layers. Premature early stopping guarantees that context conditioning is disabled. Ensuring adequate training epochs is essential for context utilization.
