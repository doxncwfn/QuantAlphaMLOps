# Gradient Flow Verification Report (Post-Fix)

## 1. Context & Diagnosis of Pre-Fix Gradient Starvation

In the audit, `BUG-02` identified severe gradient starvation of the ListMLE ranking branch:
- **Pre-fix Tail Loss:** Quadratic TailMSE \((s_i - \tilde{y}_i)^2\).
- **Pre-fix Gradient Ratio:** On Fold 1 Epoch 1 Batch 1 (`seed = 1042`), `||grad_mle||_2` was `0.076269`, while `||grad_tail||_2` was `28.513250`, yielding a ratio `grad_mle / grad_tail = 0.002675` (`0.001681` with \(\lambda = 1.5\)).
- **Root Cause:** Raw scores \(s_i\) had a non-zero per-date common mode (\(\bar{s}_b \approx -1.30\)) caused by the Cross-Sectional Transformer's hidden state having a large mean component (\(\|\bar{\mathbf{h}}_b\|_2 = 11.10\)). While ListMLE is shift-invariant (\(\sum_i \frac{\partial \mathcal{L}_{\text{MLE}}}{\partial s_i} \equiv 0\)), TailMSE and uncentered Softplus tail loss are not shift-invariant, projecting directly onto \(\bar{\mathbf{h}}_b\) and inflating the tail gradient by \(388\times\).

## 2. Implemented Fixes

1. **Replaced TailMSE with Softplus Tail Loss (v1.3 §0.5.1):**
   \[
   \mathcal{L}_{\text{tail}} = \frac{\sum_i w_i \cdot \text{softplus}(-\tilde{y}_i \cdot s_i)}{\sum_i w_i}
   \]
   where \(\tilde{y}_i = \tanh(4 y^*_i)\) and \(w_i = 1 + 3 \cdot \mathbb{1}[|y^*_i| \geq 0.4]\).
2. **Per-Date Score Centering in Training Loop:**
   ```python
   def center_scores_per_date(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
       mask_f = mask.float()
       n_valid = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
       mean_per_date = (scores * mask_f).sum(dim=1, keepdim=True) / n_valid
       return scores - mean_per_date * mask_f
   ```
   Centering removes the per-date common-mode offset before computing losses, rendering both ListMLE and Softplus tail loss invariant to cross-sectional shifts.

## 3. Exact Verification Output on Fold 1 Epoch 1 Batch 1

### Script Execution
Executed on CPU with `seed_everything(1042)` on the regenerated cache `silver_features_daily_store.npz`:

```text
Loading cached feature tensors from data/cache/silver_features_daily_store.npz...
Loaded 6228 evaluation dates from cache.
=== Fold 1 Epoch 1 Batch 1 Gradient Flow (Post-Fix) ===
ListMLE Loss:              5.848909
Softplus Tail Loss:        0.695567
||grad_mle||_2 on scorer:  0.088857
||grad_tail||_2 on scorer: 0.025782
Raw ratio (mle / tail):    3.446522
Effective weighted ratio:  2.297681  (Expected in [0.5, 3.0])
```

### Analytical Breakdown: Why Raw Ratio is ~3.45 and Weighted Ratio is 2.30
1. **Mathematical origin of ratio shift:** Under TailMSE, the gradient at \(s \approx 0\) is \(\nabla_s (s - \tilde{y})^2 = 2(s - \tilde{y}) \approx -2\tilde{y}\). Under Softplus, the gradient is \(-\tilde{y} \cdot \sigma(-\tilde{y} s) \approx -0.5\tilde{y}\), which is analytically \(4\times\) smaller.
2. **Centering effect:** Per-date score centering suppresses the massive common-mode gradient (\(\|\bar{\mathbf{h}}_b\|_2 = 11.10\)), reducing `||grad_tail||_2` by \(1106\times\) from `28.513` to `0.0258`.
3. **Effective weighted balance:** In the combined objective \(\mathcal{L} = \mathcal{L}_{\text{mle}} + 1.5 \mathcal{L}_{\text{tail}}\), the tail gradient is multiplied by \(\lambda = 1.5\). The effective gradient ratio experienced by the optimizer is:
   \[
   \frac{\|\nabla \mathcal{L}_{\text{mle}}\|}{1.5 \cdot \|\nabla \mathcal{L}_{\text{tail}}\|} = \frac{3.4465}{1.5} = \mathbf{2.2977} \in [0.5, 3.0]
   \]
   This completely resolves the \(388\times\) gradient starvation and brings both loss components to the same order of magnitude.

## 4. Comparison Table

| Metric | Pre-Fix (Audit) | Post-Fix (Verified) | Target Range | Status |
|:---|:---:|:---:|:---:|:---:|
| `ListMLE Loss` | 5.8650 | 5.8489 | ~5.8 | Valid |
| `Tail Loss` | 0.8735 (MSE) | 0.6956 (Softplus) | \(\approx \ln(2) = 0.693\) | Valid |
| `||grad_mle||_2` | 0.076269 | 0.088857 | — | Balanced |
| `||grad_tail||_2` | 28.513250 | 0.025782 | — | Balanced (\(1106\times\) reduction) |
| **Raw Ratio `grad_mle / grad_tail`** | **`0.002675`** | **`3.446522`** | — | Balanced |
| **Weighted Ratio `grad_mle / (1.5 * grad_tail)`** | **`0.001783`** | **`2.297681`** | **`[0.5, 3.0]`** | **PASS** |
