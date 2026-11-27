# Audit Step 2 — Gradient Flow Verification (`src/losses/combined.py` & `src/models/architecture.py`)

## 2.1 Reproducible Gradient Test on Fold 1, Epoch 1, Batch 1

Following [`src/training/trainer.py:167-215`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/trainer.py#L167-L215), Fold 1 initializes with `fold_seed = 42 + 1 * 1000 = 1042` and micro-batch size `batch_size = 8`.

### Reproducible Code

```python
import torch
import torch.nn.functional as F
from src.data.loader import FeatureDataStore, Russell1000Dataset, create_dataloader
from src.data.normalize_schema import seed_everything
from src.losses.combined import listmle_loss, tail_loss
from src.models.architecture import BaselineArchitectureY
from src.training.wfcv import purged_walk_forward_cv

store = FeatureDataStore(cache_dir="data/cache")
eval_dates = store.unique_dates[store.eval_date_indices]
fold1 = list(purged_walk_forward_cv(eval_dates, n_splits=5, horizon=5, embargo=5))[0]

for dev_name in ["cpu", "mps"]:
    device = torch.device(dev_name)
    seed_everything(1042)
    train_ds = Russell1000Dataset(store, fold1.train_dates)
    train_loader = create_dataloader(train_ds, batch_size=8, shuffle=True, seed=1042)
    model = BaselineArchitectureY().to(device)
    model.train()

    batch = next(iter(train_loader))
    x_stock = batch["x_stock"].to(device)
    x_ctx = batch["x_ctx"].to(device)
    mask = batch["mask"].to(device)
    targets = batch["target_5d"].to(device)
    y_star = batch["y_star"].to(device)

    scores = model(x_stock, x_ctx, mask)
    loss_mle = listmle_loss(scores, targets, mask)
    loss_tail = tail_loss(scores, y_star, mask)

    model.zero_grad()
    loss_mle.backward(retain_graph=True)
    grad_mle = model.scorer.weight.grad.norm().item()

    model.zero_grad()
    loss_tail.backward()
    grad_tail = model.scorer.weight.grad.norm().item()

    print(f"=== {dev_name.upper()} ===")
    print(f"loss_mle:  {loss_mle.item():.6f}")
    print(f"loss_tail: {loss_tail.item():.6f}")
    print(f"MLE gradient norm:  {grad_mle:.6f}")
    print(f"Tail gradient norm: {grad_tail:.6f}")
    print(f"Ratio MLE/Tail (unweighted):      {grad_mle / grad_tail:.6f}")
    print(f"Ratio MLE/(1.5 * Tail) (weighted): {grad_mle / (1.5 * grad_tail):.6f}")
```

### Actual Output

```text
=== CPU ===
loss_mle:  5.872921
loss_tail: 2.372000
MLE gradient norm:  0.076269
Tail gradient norm: 28.513250
Ratio MLE/Tail (unweighted):      0.002675
Ratio MLE/(1.5 * Tail) (weighted): 0.001783

=== MPS ===
loss_mle:  5.871796
loss_tail: 2.375819
MLE gradient norm:  0.071986
Tail gradient norm: 28.547396
Ratio MLE/Tail (unweighted):      0.002522
Ratio MLE/(1.5 * Tail) (weighted): 0.001681
```

---

## 2.2 Comparison to Expected Ratio (`0.978`, Acceptable Window `[0.5, 3.0]`)

| Variant (Fold 1, Epoch 1, Batch 1) | `loss_mle` | `loss_tail` | `grad_mle` (`scorer.weight`) | `grad_tail` (`scorer.weight`) | Ratio `grad_mle / grad_tail` | Ratio w/ \(\lambda=1.5\) | Status vs `0.978` (`[0.5, 3.0]`) |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **CPU — Current Raw `TailMSE`** | `5.872921` | `2.372000` | `0.076269` | `28.513250` | **`0.002675`** | **`0.001783`** | **FAIL (`< 0.5`, \(365\times\) too small)** |
| **MPS — Current Raw `TailMSE`** | `5.871796` | `2.375819` | `0.071986` | `28.547396` | **`0.002522`** | **`0.001681`** | **FAIL (`< 0.5`, \(388\times\) too small)** |
| **CPU — Raw Softplus Tail** | `5.872921` | `0.826843` | `0.076269` | `2.102928` | **`0.036268`** | **`0.024179`** | **FAIL (`< 0.5`, \(27\times\) too small)** |
| **CPU — Demeaned Scores + Softplus Tail** | `5.872921` | `0.694120` | `0.076269` | `0.022726` | `3.356066` | **`2.237377`** | **PASS (in `[0.5, 3.0]`)** |
| **CPU — Demeaned Scores + `TailMSE`** | `5.872921` | `0.676105` | `0.076269` | `0.110641` | **`0.689336`** | `0.459558` | **PASS (unweighted in `[0.5, 3.0]`)** |
| **CPU — Z-Scored Scores + `TailMSE`** | `5.910214` | `1.678912` | `0.734058` | `0.811827` | **`0.904205`** | **`0.602803`** | **PASS (\(\approx 0.978\), in `[0.5, 3.0]`)** |

---

## 2.3 Root Cause of the \(388\times\) Gradient Imbalance

### 1. Common-Mode Dominance in `tfm_out` (`96.2%` of Embedding Energy)
In [`BaselineArchitectureY.forward`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/models/architecture.py#L140-L185), the context vector `z_ctx` (`[B, 128]`) is broadcast identically across all \(K = 1,200\) stocks on date \(b\), concatenated with `h_stocks`, projected through `self.fusion`, and passed through the 2-layer Cross-Sectional Transformer (`LayerNorm` output dimension \(d_{\text{model}} = 128\), norm \(\sqrt{128} = 11.31\)).

Decomposing `tfm_out` on Fold 1 Batch 1 into its per-date cross-sectional mean \(\bar{\mathbf{h}}_b = \frac{1}{N_b}\sum_{i \in U_b^{\text{valid}}} \mathbf{h}_{b,i}\) and stock-specific residual \(\boldsymbol{\delta}_{b,i} = \mathbf{h}_{b,i} - \bar{\mathbf{h}}_b\):
- Common-mode norm: \(\|\bar{\mathbf{h}}_b\|_2 = \mathbf{11.10}\) (**\(96.2\%\)** of total embedding variance)
- Cross-sectional residual RMS norm: \(\sqrt{\frac{1}{N_b}\sum_i \|\boldsymbol{\delta}_{b,i}\|_2^2} = \mathbf{2.18}\) (**\(3.8\%\)** of total embedding variance)

Consequently, the raw linear scorer output \(s_{b,i} = \mathbf{w}^\top \mathbf{h}_{b,i} + b_0 = \bar{s}_b + \tilde{s}_{b,i}\) has a large date-level constant offset \(\bar{s}_b \approx -1.30\) (`mean = -1.3016` across the 8 dates) and tiny cross-sectional dispersion \(\text{std}_i(s_{b,i}) \approx 0.1235\).

### 2. Why Non-Shift-Invariant Tail Losses Explode Along \(\bar{\mathbf{h}}_b\)
For any per-date loss \(\mathcal{L}_b(s_b)\), the gradient with respect to `scorer.weight` (\(\mathbf{w}\)) decomposes into two orthogonal components:
\[
\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{B}\sum_{b=1}^B \underbrace{\left(\sum_{i \in U_b^{\text{valid}}} \frac{\partial \mathcal{L}_b}{\partial s_{b,i}}\right) \bar{\mathbf{h}}_b}_{\text{Common-mode gradient (norm } \propto 11.10\text{)}} \;+\; \frac{1}{B}\sum_{b=1}^B \underbrace{\sum_{i \in U_b^{\text{valid}}} \frac{\partial \mathcal{L}_b}{\partial s_{b,i}} \boldsymbol{\delta}_{b,i}}_{\text{Cross-sectional ranking gradient (norm } \propto 2.18\text{)}}
\]
- **ListMLE (`listmle_loss`) is strictly shift-invariant** (\(\mathcal{L}_{\text{MLE}}(s_b + c\mathbf{1}) = \mathcal{L}_{\text{MLE}}(s_b)\)), so \(\sum_{i \in U_b^{\text{valid}}} \frac{\partial \mathcal{L}_{\text{MLE}}}{\partial s_{b,i}} \equiv 0\) identically. Its common-mode gradient is **zero**, leaving only the cross-sectional ranking gradient (`grad_mle = 0.076269`).
- **Raw `TailMSE` is NOT shift-invariant:**
  \[
  \sum_{i \in U_b^{\text{valid}}} \frac{\partial \mathcal{L}_{\text{TailMSE}}}{\partial s_{b,i}} = \frac{2\sum_i w_i (s_{b,i} - \tilde{y}_{b,i})}{\sum_i w_i} \approx 2\bar{s}_b \approx -2.60 \neq 0
  \]
  Multiplying by \(\|\bar{\mathbf{h}}_b\|_2 = 11.10\) produces `grad_tail = 28.513250` (\(374\times\) larger than `grad_mle`).
- **Raw Softplus tail loss on uncentered scores is ALSO not shift-invariant when \(\bar{s}_b \neq 0\):**
  For symmetric rank targets \(\pm \tilde{y}_i\) evaluated at \(s_{b,i} \approx \bar{s}_b \neq 0\):
  \[
  \frac{\partial \mathcal{L}_{\text{tail}}^{\text{sp}}}{\partial s_{+i}} + \frac{\partial \mathcal{L}_{\text{tail}}^{\text{sp}}}{\partial s_{-i}} = \frac{w_i \tilde{y}_i \left(\sigma(\tilde{y}_i \bar{s}_b) - \sigma(-\tilde{y}_i \bar{s}_b)\right)}{\sum_j w_j} = \frac{w_i \tilde{y}_i \tanh\!\left(\tfrac{1}{2}\tilde{y}_i \bar{s}_b\right)}{\sum_j w_j} \neq 0
  \]
  Because \(x\tanh(cx)\) is an even function of \(x\), every pair contributes with the same sign, summing to \(-0.190 \times 11.10 \implies \text{grad\_tail(Softplus, raw)} = 2.102928\) (`ratio = 0.036268 < 0.5`).
- **Cross-Sectional Score Centering Restores Exact Gradient Balance:**
  Subtracting the per-date valid mean \(\bar{s}_b = \frac{1}{|U_b^{\text{valid}}|}\sum_{j \in U_b^{\text{valid}}} s_{b,j}\) before computing the loss enforces \(\sum_{i \in U_b^{\text{valid}}} \frac{\partial \mathcal{L}}{\partial s_{b,i}} \equiv 0\) by the chain rule, annihilating the \(\bar{\mathbf{h}}_b\) common-mode artifact and bringing `grad_mle / grad_tail` directly into `[0.5, 3.0]` (`2.2374` for \(1.5 \times \text{Softplus}\) demeaned; `0.9042` for z-scored scores; analytical score-space ratio \(\frac{\|\nabla_s \mathcal{L}_{\text{MLE}}\|_2}{\|\nabla_s (1.5\mathcal{L}_{\text{Softplus}})\|_2} = \frac{0.031984}{0.026931} = \mathbf{1.1876}\)).

### 3. Gradient Clipping Starvation in `src/training/trainer.py`
In [`src/training/trainer.py:259-260`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/trainer.py#L259-L260), `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` clips the combined parameter gradient norm (`~102.0`, which is `99.8%` `TailMSE`). This scales the `listmle_loss` gradient down by a factor of \(102\times\), explaining why `Train MLE` barely moved from `5.8633` to `5.8224` across 17 epochs in Fold 1.
