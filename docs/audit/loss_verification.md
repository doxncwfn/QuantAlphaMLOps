# Audit Step 1 — Loss Function Verification (`src/losses/combined.py`)

## 1.1 Exact Code of `src/losses/combined.py`

Source: [`src/losses/combined.py`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/losses/combined.py#L1-L163)

```python
"""Loss functions for v1.3 ranking model: ListMLE + Tail-weighted soft loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def listmle_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Listwise Maximum Likelihood Estimation (ListMLE) ranking loss per v1.3 §0.5.1.

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores of shape [B, K].
    targets : torch.Tensor
        Realized targets (target_5d or y_star) of shape [B, K].
    mask : torch.Tensor
        Boolean mask of shape [B, K] where True indicates valid constituent.
    eps : float
        Numerical stability epsilon.

    Returns
    -------
    torch.Tensor
        Scalar ListMLE loss averaged across valid dates.
    """
    batch_size = scores.shape[0]
    date_losses = []

    for b in range(batch_size):
        valid = mask[b]
        n_valid = int(valid.sum())
        if n_valid < 2:
            continue

        s = scores[b][valid]
        y = targets[b][valid]

        # Sort scores in descending order of true target (highest return first)
        order = torch.argsort(y, descending=True)
        s_sorted = s[order]

        # Numerically stable suffix logsumexp:
        # suffix_lse[k] = log sum_{j=k}^{n-1} exp(s_sorted[j])
        rev_s = torch.flip(s_sorted, dims=[0])
        rev_lse = torch.logcumsumexp(rev_s, dim=0)
        suffix_lse = torch.flip(rev_lse, dims=[0])

        # Negative log-likelihood per element: suffix_lse[k] - s_sorted[k]
        nll = suffix_lse - s_sorted
        date_losses.append(nll.mean())

    if not date_losses:
        return scores.sum() * 0.0

    return torch.stack(date_losses).mean()


def tail_loss(
    scores: torch.Tensor,
    y_star: torch.Tensor,
    mask: torch.Tensor,
    tail_threshold: float = 0.4,
    tail_multiplier: float = 3.0,
    soft_scale: float = 4.0,
) -> torch.Tensor:
    """Tail-weighted loss with anti-symmetric soft labels: y_tilde = tanh(4 * y_star).

    Weights:
        w_i = 1.0 + 3.0 * 1[|y_star_i| >= 0.4]

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores of shape [B, K].
    y_star : torch.Tensor
        Cross-sectional rank target in [-0.5, +0.5] of shape [B, K].
    mask : torch.Tensor
        Boolean mask of shape [B, K].
    tail_threshold : float
        Threshold for tail deciles (|y_star| >= 0.4).
    tail_multiplier : float
        Weight added to tail constituents (yielding weight 1 + 3 = 4).
    soft_scale : float
        Scale factor in soft label tanh(scale * y_star).

    Returns
    -------
    torch.Tensor
        Scalar weighted MSE loss averaged across valid dates.
    """
    batch_size = scores.shape[0]
    date_losses = []

    for b in range(batch_size):
        valid = mask[b]
        if int(valid.sum()) < 2:
            continue

        s = scores[b][valid]
        y = y_star[b][valid]

        # Soft labels: odd (anti-symmetric) function of y*
        y_soft = torch.tanh(soft_scale * y)

        # Symmetric tail weights: 4.0 for |y*| >= 0.4, 1.0 otherwise
        is_tail = (y.abs() >= tail_threshold).float()
        weights = 1.0 + tail_multiplier * is_tail

        # Weighted MSE on soft labels
        sq_err = (s - y_soft) ** 2
        w_loss = (weights * sq_err).sum() / (weights.sum() + 1e-8)
        date_losses.append(w_loss)

    if not date_losses:
        return scores.sum() * 0.0

    return torch.stack(date_losses).mean()


class CombinedRankingLoss(nn.Module):
    """Combined loss: L = L_ListMLE + 1.5 * L_tail per v1.3 Baseline specification."""

    def __init__(
        self,
        lambda_tail: float = 1.5,
        tail_threshold: float = 0.4,
        tail_multiplier: float = 3.0,
        soft_scale: float = 4.0,
    ):
        super().__init__()
        self.lambda_tail = lambda_tail
        self.tail_threshold = tail_threshold
        self.tail_multiplier = tail_multiplier
        self.soft_scale = soft_scale

    def forward(
        self,
        scores: torch.Tensor,
        target_5d: torch.Tensor,
        y_star: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss and return (total_loss, l_listmle, l_tail)."""
        l_mle = listmle_loss(scores, target_5d, mask)
        l_tail = tail_loss(
            scores,
            y_star,
            mask,
            tail_threshold=self.tail_threshold,
            tail_multiplier=self.tail_multiplier,
            soft_scale=self.soft_scale,
        )
        total_loss = l_mle + self.lambda_tail * l_tail
        return total_loss, l_mle, l_tail
```

### Classification of `src/losses/combined.py`

- **Implemented Formula:** **TailMSE** (`sq_err = (s - y_soft) ** 2`, `w_loss = (weights * sq_err).sum() / (weights.sum() + 1e-8)`):
  \[
  \mathcal{L}_{\text{tail}} = \frac{\sum_i w_i \cdot (\hat{y}_i - \tilde{y}_i)^2}{\sum_i w_i + 10^{-8}}
  \]
  with \(\tilde{y}_i = \tanh(4 y^*_i)\) and \(w_i = 1 + 3 \cdot \mathbb{1}[|y^*_i| \geq 0.4]\).
- **Deviation from Softplus Tail Loss:** It does **not** implement the Softplus margin tail loss:
  \[
  \mathcal{L}_{\text{tail}} = \frac{\sum_i w_i \cdot \text{softplus}(-\tilde{y}_i \cdot s_i)}{\sum_i w_i}
  \]
- **Deviation from [`specs/features_baseline.md` §0.5.1](file:///Users/doxncwfn/Project/QuantAlphaMLOps/specs/features_baseline.md#L74-L80):** The frozen baseline spec §0.5.1 defines pure Plackett-Luce ListMLE summed over \(k = 1 \dots |U_t^{\text{valid}}|\) without a quadratic MSE penalty clamping raw logits to \([-0.964, +0.964]\).

---

## 1.2 Verification of `listmle_loss`

1. **Suffix Softmax:** Computed over the **suffix** \(\sum_{j=k}^{N} \exp(\hat{y}_{\pi(j)})\) at each rank position \(k\) via `rev_s = torch.flip(s_sorted, dims=[0])`, `rev_lse = torch.logcumsumexp(rev_s, dim=0)`, `suffix_lse = torch.flip(rev_lse, dims=[0])`, and `nll = suffix_lse - s_sorted` ([lines 52–57](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/losses/combined.py#L52-L57)). Note that `nll.mean()` divides by \(N = |U_t^{\text{valid}}|\) rather than summing over \(k = 1 \dots N\).
2. **Sorting Target:** `CombinedRankingLoss.forward` ([line 152](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/losses/combined.py#L152)) passes `target_5d` (not `y_star`) to `listmle_loss`, which sorts descending via `order = torch.argsort(y, descending=True)`. Note that `torch.argsort` defaults to `stable=False`, so exact ties in `target_5d` have non-deterministic permutation ordering on GPU/MPS.
3. **Mask Application:** `valid = mask[b]` is applied consistently to both `s = scores[b][valid]` and `y = targets[b][valid]` before sorting and computing `logcumsumexp`.
4. **1D Input Shape Bug ([lines 34–41 & 99–106](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/losses/combined.py#L34-L41)):** Both `listmle_loss` and `tail_loss` assume 2D `[B, K]` inputs and loop `for b in range(scores.shape[0]): valid = mask[b]; if int(valid.sum()) < 2: continue`. When called with 1D tensors of shape `[5]`, `mask[b]` is a 0D scalar `True` (`int(valid.sum()) == 1 < 2`), causing every element to be skipped and returning `0.0` for all inputs.

---

## 1.3 Toy Test on `listmle_loss`

### Reproducible Command & Output

```python
import math
import torch
from src.losses.combined import listmle_loss

scores = torch.tensor([1.0, 0.5, 0.0, -0.5, -1.0])
targets = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])  # already sorted descending
mask = torch.ones(5, dtype=torch.bool)
scores_random = torch.tensor([0.5, -1.0, 1.0, -0.5, 0.0])
scores_bad = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])

print("=== 1D Input (as written in prompt) ===")
print("Test 1 (perfect):", listmle_loss(scores, targets, mask).item())
print("Test 2 (random):", listmle_loss(scores_random, targets, mask).item())
print("Test 3 (adversarial):", listmle_loss(scores_bad, targets, mask).item())

print("\n=== 2D Input (unsqueezed batch dim [1, 5]) ===")
print(
    "Test 1 (perfect):",
    listmle_loss(scores.unsqueeze(0), targets.unsqueeze(0), mask.unsqueeze(0)).item(),
)
print(
    "Test 2 (random):",
    listmle_loss(
        scores_random.unsqueeze(0), targets.unsqueeze(0), mask.unsqueeze(0)
    ).item(),
)
print(
    "Test 3 (adversarial):",
    listmle_loss(
        scores_bad.unsqueeze(0), targets.unsqueeze(0), mask.unsqueeze(0)
    ).item(),
)
print("Reference ln(5!)/5 (all-zero scores):", math.log(math.factorial(5)) / 5)
```

```text
=== 1D Input (as written in prompt) ===
Test 1 (perfect): 0.0
Test 2 (random): 0.0
Test 3 (adversarial): 0.0

=== 2D Input (unsqueezed batch dim [1, 5]) ===
Test 1 (perfect): 0.5577566623687744
Test 2 (random): 1.0663106441497803
Test 3 (adversarial): 1.5577566623687744
Reference ln(5!)/5 (all-zero scores): 0.9574983748394816
```

### Diagnosis of Toy Test 1.3
1. **1D Call:** Fails completely (`0.0` for all three cases) because `int(mask[b].sum()) == 1 < 2`.
2. **2D Call:**
   - **Test 1 (Perfect):** Returns **`0.557757`**, which fails the `< 0.01` ("near 0") check because `listmle_loss` has no temperature scaling \(\tau\) on raw scores; with score spacing \(\Delta s = 0.5\), \(\frac{1}{5}\sum_{k=1}^5 \ln\sum_{j=k}^5 e^{s_j - s_k} = 0.557757\). Furthermore, because `TailMSE` in `CombinedRankingLoss` forces \(s_i \approx \tilde{y}_i \in [-0.964, +0.964]\), `listmle_loss` can never approach `0.0` during training.
   - **Test 2 (Random):** Returns **`1.066311`** (note: for uniform/zero scores, \(\frac{\ln(5!)}{5} = \frac{\ln(120)}{5} = \mathbf{0.957498}\)).
   - **Test 3 (Adversarial):** Returns **`1.557757`** (strictly worse than random by `0.4914` and worse than perfect by exactly `1.0000`).

---

## 1.4 Toy Test on `tail_loss`

### Reproducible Command & Output

```python
import torch
import torch.nn.functional as F
from src.losses.combined import tail_loss

scores_perfect = torch.tensor([1.0, 0.5, 0.0, -0.5, -1.0])
y_star = torch.tensor([0.5, 0.25, 0.0, -0.25, -0.5])
mask = torch.ones(5, dtype=torch.bool)

print("=== 1D Input (as written in prompt) ===")
print(f"Perfect: {tail_loss(scores_perfect, y_star, mask).item()}")
print(f"Random (zeros): {tail_loss(torch.zeros(5), y_star, mask).item()}")
print(f"Bad (inverted): {tail_loss(-scores_perfect, y_star, mask).item()}")

print("\n=== 2D Input [1, 5] — Current TailMSE ===")
print(
    f"Perfect: {tail_loss(scores_perfect.unsqueeze(0), y_star.unsqueeze(0), mask.unsqueeze(0)).item():.6f}"
)
print(
    f"Random (zeros): {tail_loss(torch.zeros(1, 5), y_star.unsqueeze(0), mask.unsqueeze(0)).item():.6f}"
)
print(
    f"Bad (inverted): {tail_loss(-scores_perfect.unsqueeze(0), y_star.unsqueeze(0), mask.unsqueeze(0)).item():.6f}"
)


def softplus_tail_loss_ref(scores: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
    y_tilde = torch.tanh(4.0 * y_star)
    w = 1.0 + 3.0 * (y_star.abs() >= 0.4).float()
    return (w * F.softplus(-y_tilde * scores)).sum() / w.sum()


print("\n=== Reference v1.3 Softplus Tail Loss ===")
print(f"Perfect: {softplus_tail_loss_ref(scores_perfect, y_star).item():.6f}")
print(f"Random (zeros): {softplus_tail_loss_ref(torch.zeros(5), y_star).item():.6f}")
print(f"Bad (inverted): {softplus_tail_loss_ref(-scores_perfect, y_star).item():.6f}")
```

```text
=== 1D Input (as written in prompt) ===
Perfect: 0.0
Random (zeros): 0.0
Bad (inverted): 0.0

=== 2D Input [1, 5] — Current TailMSE ===
Perfect: 0.013383
Random (zeros): 0.781350
Bad (inverted): 3.094771

=== Reference v1.3 Softplus Tail Loss ===
Perfect: 0.392654
Random (zeros): 0.693147
Bad (inverted): 1.163001
```

### All-Zero Score Collapse Under `TailMSE` on Real Training Data
For a uniform rank target \(y^* \sim \text{Uniform}[-0.5, +0.5]\), when a model predicts identically zero scores (\(\mathbf{s} \equiv \mathbf{0}\)):
\[
\mathcal{L}_{\text{TailMSE}}(\mathbf{0}) = \frac{\int_{-0.5}^{0.5} (1 + 3\mathbb{1}[|u| \ge 0.4])\tanh^2(4u)\,du}{\int_{-0.5}^{0.5} (1 + 3\mathbb{1}[|u| \ge 0.4])\,du} = \mathbf{0.659432}
\]
\[
\mathcal{L}_{\text{ListMLE}}(\mathbf{0}) = \frac{\ln(N!)}{N} \approx \mathbf{5.865} \quad (N \approx 960), \qquad \mathcal{L}_{\text{Total}}(\mathbf{0}) = 5.865 + 1.5 \times 0.659432 = \mathbf{6.8541}
\]
In the failed 1A baseline run logs ([`output.txt`](file:///Users/doxncwfn/.gemini/antigravity/brain/497ba708-12c9-419b-9798-66e2da2d8914/.system_generated/steps/914/output.txt)), **every single fold collapsed to \(s_i \approx 0\) by Epoch 2**:
- Fold 1 Epoch 2: `Train Loss: 6.8598 (MLE: 5.8587, Tail: 0.6674)`
- Fold 2 Epoch 2: `Train Loss: 6.8548 (MLE: 5.8664, Tail: 0.6589)`
- Fold 3 Epoch 2: `Train Loss: 6.8554 (MLE: 5.8674, Tail: 0.6586)`
- Fold 4 Epoch 2: `Train Loss: 6.8597 (MLE: 5.8721, Tail: 0.6584)`

Because daily 5-day stock returns have low signal-to-noise ratio (\(\text{IC} \sim 0.02\text{–}0.04\)), the Bayes-optimal prediction under quadratic `TailMSE` is the conditional expectation \(s_i^* = \mathbb{E}[\tilde{y}_i \mid X_i] \approx \text{IC} \cdot \sigma(\tilde{y}) \cdot z_i \in [-0.05, +0.05]\). Any larger score incurs a quadratic MSE penalty \(w_i s_i^2\), compressing predicted score dispersion near zero and starving `listmle_loss`.

---

## 1.5 Conclusion

**`loss implementation does not match v1.3`**
- `src/losses/combined.py` implements **TailMSE** (`(s - y_soft) ** 2`) rather than the v1.3 Softplus tail loss (`softplus(-y_tilde * s)`) or pure ListMLE (§0.5.1).
- Both `listmle_loss` and `tail_loss` silently return `0.0` on 1D inputs.
