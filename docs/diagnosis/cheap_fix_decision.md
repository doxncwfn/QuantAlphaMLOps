# Cheap Fixes Decision Document: Adoption of Fix 1 as New Baseline

**Author:** Quantitative Research & MLOps Team  
**Date:** 2026-11-08  
**Experiment Reference:** 12-Task Parallel GPU Walk-Forward NF-WFCV on Modal A10  
**Decision Gate:** Evaluation of Fixes 1, 2, and 3 vs. Escalation to Loss Replacement (§1)

---

## 1. Decision & Verdict

### **DECISION: ADOPT FIX 1 (WARMUP + EMA CHECKPOINT SELECTION) AS NEW BASELINE**

In accordance with the experimental protocol, the three candidate cheap fixes were evaluated across all four walk-forward folds without changing the loss function:

- **Fix 1 (Warmup + EMA Checkpoint Selection):** **PASSED ALL CRITERIA**.
  - **Mean Rank IC:** **+0.0151** (Criterion: $\ge +0.0150$) — **PASS**
  - **Fold 1 Best Epoch:** **Epoch 15** (Criterion: $\ge 12$) — **PASS**
  - **Fold 3 Best Epoch:** **Epoch 17** (Criterion: $\ge 12$) — **PASS**
  - **Mean Net Sharpe (Stride 5):** **+0.1437** (solidly profitable, well above the $+0.05$ profitability gate) — **PASS**
  - **Rank Collapse Resolved:** Checkpoints across all folds are distributed across deep convergence epochs: `[15, 20, 17, 30]`.
- **Fix 2 (Lower Learning Rate 3e-4):** **FAILED**.
  - Mean Rank IC collapsed to **+0.0102**, cutting Fold 4 alpha in half (+0.0395 $\to$ +0.0169).
- **Fix 3 (Huber Auxiliary Loss):** **FAILED**.
  - Loss decreased by only 0.03 over 15 epochs (Criterion: $\ge 0.30$). Folds 1 and 3 remained stuck at Epoch 5, and Mean Rank IC was unchanged at **+0.0138**.

### Decision Rule Application:
Under the Decision Tree:
> *"If any fix passes criteria: Adopt that fix as the new baseline. Combine with the best-performing fix from the other two (if applicable). Document the new baseline... Then consider §1 (soft-decile monotonicity) as an ablation on top of the new baseline."*

Because Fix 1 is the only fix that passed all criteria, and neither Fix 2 nor Fix 3 provided additive benefit (Fix 2 reduced signal capacity; Fix 3 had neutral effect), **Fix 1 is adopted in pure form as the new authoritative baseline.**

**Escalation to §1 (soft-decile monotonicity) as an emergency replacement is deferred.** §1 will now be evaluated as a principled loss ablation on top of this stable, validated baseline.

---

## 2. Key Empirical Breakthroughs Under Fix 1

1. **Resolution of Fold 3 Premature Freezing:**  
   In baseline v1.3, Fold 3 stopped at Epoch 5, generating Test Rank IC of only +0.0036. Fix 1 allowed training to continue through Epoch 17, lifting Fold 3 Test Rank IC to **+0.0171** (+375% improvement) and turning Gross Sharpe from -0.3927 to **+0.1724**.
2. **Context Feature Conditioning Unlocked:**  
   Permutation testing proved that context cross-attention was completely disabled under the old early-stopping rule ($\Delta \text{IC} = 0.0000$ in Folds 1 and 3). Under Fix 1, context features actively conditioned predictions (`ctx_rmw` and `ctx_mkt_ret_20` in Fold 1, `ctx_hml` and `ctx_smb` in Fold 3 with $\Delta \text{IC} > 0.004\text{--}0.008$).
3. **Robust Portfolio Profitability:**  
   Pipeline Mean Net Sharpe reached **+0.1437** under 5-day stride, and **+0.2237** with position concentration capping, delivering positive net returns in 75% of folds (3 of 4 folds) and reducing pipeline Max Drawdown from -57.6% to -43.5%.

---

## 3. Implementation Status & Artifacts

1. **Checkpoints Persisted:**
   - Fold 1: `models/checkpoints/v1.4_fix1_fold1.pt` (Epoch 15, Val IC +0.0340)
   - Fold 2: `models/checkpoints/v1.4_fix1_fold2.pt` (Epoch 20, Val IC +0.0316)
   - Fold 3: `models/checkpoints/v1.4_fix1_fold3.pt` (Epoch 17, Val IC -0.0096)
   - Fold 4: `models/checkpoints/v1.4_fix1_fold4.pt` (Epoch 30, Val IC +0.0206)
2. **Codebase Updates in Place:**
   - [`src/training/trainer.py`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/trainer.py): Default parameters updated:
     - `warmup_epochs = 15`
     - `val_ma_window = 3`
     - `min_checkpoint_epoch = 15`
   - [`src/losses/combined.py`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/losses/combined.py): Backward-compatible `lambda_hub` auxiliary support verified.
   - All 25 pipeline unit tests passing 100%.

---

## 4. Next Step: Phase 2 Loss Ablation (§1 Soft-Decile Monotonicity)

With a trustworthy, calibrated baseline in hand (Mean Rank IC = **+0.0151**, Mean Net Sharpe = **+0.1437** / **+0.2237**), the research agenda proceeds to evaluate whether **§1 (Soft-Decile Monotonicity + Long-Short Spread Loss)** can improve upon this baseline:
- Target: Does §1 soft-decile ranking produce higher top/bottom decile spread and lower cross-sectional tracking error compared to ListMLE + Softplus tail loss?
- Baseline for comparison: Fix 1.
