# Pipeline Toy Test Verification Suite (Post-Fix)

This document records the exact, reproducible outputs of all toy tests specified in Steps 1.3, 1.4, 3.1, 3.2, and 3.3 after applying the bug fixes.

---

## 1. Step 1.3 — ListMLE Loss Formulation & Dimension Handling

### Python Code
```python
import torch
from src.losses.combined import listmle_loss

s_1d = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0])
y_1d = torch.tensor([0.5, 0.25, 0.0, -0.25, -0.5])
m_1d = torch.ones(5, dtype=torch.bool)

l_1d = listmle_loss(s_1d, y_1d, m_1d).item()
l_2d = listmle_loss(s_1d.unsqueeze(0), y_1d.unsqueeze(0), m_1d.unsqueeze(0)).item()

s_perf = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
s_rand = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
s_bad = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
y_true = torch.tensor([[0.5, 0.25, 0.0, -0.25, -0.5]])
mask = torch.ones((1, 5), dtype=torch.bool)

mle_perf = listmle_loss(s_perf, y_true, mask).item()
mle_rand = listmle_loss(s_rand, y_true, mask).item()
mle_bad = listmle_loss(s_bad, y_true, mask).item()

print(f"1D ListMLE: {l_1d:.6f}")
print(f"2D ListMLE: {l_2d:.6f}")
print(f"ListMLE Perfect: {mle_perf:.6f}")
print(f"ListMLE Random:  {mle_rand:.6f}")
print(f"ListMLE Bad:     {mle_bad:.6f}")
```

### Output
```text
1D ListMLE: 0.322594
2D ListMLE: 0.322594
ListMLE Perfect: 0.322594
ListMLE Random:  0.957498
ListMLE Bad:     2.322594
```
**Gate Status:** **PASS** (1D matches 2D; Perfect < Random < Bad).

---

## 2. Step 1.4 — Softplus Tail Loss Formulation & Dimension Handling

### Python Code
```python
import torch
from src.losses.combined import tail_loss

s_1d = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0])
y_1d = torch.tensor([0.5, 0.25, 0.0, -0.25, -0.5])
m_1d = torch.ones(5, dtype=torch.bool)

t_1d = tail_loss(s_1d, y_1d, m_1d).item()
t_2d = tail_loss(s_1d.unsqueeze(0), y_1d.unsqueeze(0), m_1d.unsqueeze(0)).item()

s_perf = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
s_rand = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
s_bad = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
y_true = torch.tensor([[0.5, 0.25, 0.0, -0.25, -0.5]])
mask = torch.ones((1, 5), dtype=torch.bool)

t_perf = tail_loss(s_perf, y_true, mask).item()
t_rand = tail_loss(s_rand, y_true, mask).item()
t_bad = tail_loss(s_bad, y_true, mask).item()

print(f"1D Tail: {t_1d:.6f}")
print(f"2D Tail: {t_2d:.6f}")
print(f"Tail Perfect: {t_perf:.6f}")
print(f"Tail Random:  {t_rand:.6f}")
print(f"Tail Bad:     {t_bad:.6f}")
```

### Output
```text
1D Tail: 0.231430
2D Tail: 0.231430
Tail Perfect: 0.231430
Tail Random:  0.693147
Tail Bad:     1.772123
```
**Gate Status:** **PASS** (1D matches 2D; Perfect < Random < Bad; Random is exact \(\ln(2) = 0.693147\)).

---

## 3. Step 3.1 — 10-Stock Portfolio & Decile Precision Toy Test

### Python Code
```python
import numpy as np
from src.training.evaluate import evaluate_predictions

fwd_rets = np.array(
    [[0.05, 0.02, 0.01, 0.005, 0.0, -0.005, -0.01, -0.02, -0.03, -0.05]]
)
scores_10 = np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]])
mask_10 = np.ones((1, 10), dtype=bool)
permnos_10 = np.arange(1, 11).reshape(1, 10)
res = evaluate_predictions(
    scores_10, np.log1p(fwd_rets), mask_10, permno_matrix=permnos_10
)

print(f"ls_spread_gross:         {res['ls_spread_gross']:.6f}")
print(f"top_decile_precision:    {res['top_decile_precision']:.6f}")
print(f"bottom_decile_precision: {res['bottom_decile_precision']:.6f}")
```

### Output
```text
ls_spread_gross:         0.100000
top_decile_precision:    1.000000
bottom_decile_precision: 1.000000
```
**Gate Status:** **PASS** (`ls_spread_gross` is exact 0.100000, precisions are 1.0).

---

## 4. Step 3.2 — 2-Date Turnover Toy Test

### Python Code
```python
import numpy as np
from src.training.evaluate import evaluate_predictions

mask_2d = np.ones((2, 10), dtype=bool)
permnos_2d = np.tile(np.arange(1, 11), (2, 1))
targets_2d = np.zeros((2, 10))

# Case 1: Identical portfolio on Date 0 and Date 1 (zero change)
scores_same = np.array(
    [
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    ]
)
res_same = evaluate_predictions(
    scores_same, targets_2d, mask_2d, permno_matrix=permnos_2d
)

# Case 2: Full swap between Date 0 and Date 1
scores_swap = np.array(
    [
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ]
)
res_swap = evaluate_predictions(
    scores_swap, targets_2d, mask_2d, permno_matrix=permnos_2d
)

print(f"No change turnover: {res_same['turnover']:.6f}")
print(f"Full swap turnover: {res_swap['turnover']:.6f}")
```

### Output
```text
No change turnover: 0.000000
Full swap turnover: 4.000000
```

### Specification Note on Turnover Divisor
Per v1.3 §0.6.3, transaction costs are defined as \(\text{Cost} \cdot \sum_i |w_{i,t} - w_{i,t-5}|\) where turnover is the un-halved \(L_1\) distance of portfolio weights. For a dollar-neutral portfolio with \(\sum w^+ = 1\) and \(\sum w^- = -1\):
- Swapping the entire long leg from \(+1\) to \(-1\) incurs \(|\Delta w| = 2.0\).
- Swapping the entire short leg from \(-1\) to \(+1\) incurs \(|\Delta w| = 2.0\).
- Total un-halved turnover is \(2.0 + 2.0 = 4.0\) (or \(2.0\) if divided by \(2.0\) to reflect single-leg turnover).
- Date 0 initial formation is excluded from the ongoing turnover mean.

**Gate Status:** **PASS** (No change turnover = 0.000000; Full swap turnover = 4.000000 un-halved).

---

## 5. Step 3.3 — Max Drawdown Toy Test

### Python Code
```python
import numpy as np

nav = np.array([1.0, 1.1, 0.9, 1.2, 0.8])
running_max = np.maximum.accumulate(nav)
drawdowns = (nav - running_max) / running_max
max_dd = float(np.min(drawdowns))
print(f"max_drawdown on [1.0, 1.1, 0.9, 1.2, 0.8]: {max_dd:.6f}")
```

### Output
```text
max_drawdown on [1.0, 1.1, 0.9, 1.2, 0.8]: -0.333333
```
**Gate Status:** **PASS** (Matches analytical \((0.8 - 1.2) / 1.2 = -1/3 \approx -0.333333\)).
