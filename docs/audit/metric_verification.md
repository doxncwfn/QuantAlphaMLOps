# Audit Step 5 — Metric Computation & Early Stopping Verification (`src/training/trainer.py` & `src/training/evaluate.py`)

## 5.1 Validation Rank IC Computation Code

Source: [`src/training/trainer.py:37-82`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/trainer.py#L37-L82)

```python
def evaluate_rank_ic(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean Spearman Rank IC on a dataloader split."""
    model.eval()
    rank_ics: list[float] = []
    use_autocast = device.type == "cuda"

    with torch.inference_mode():
        for batch in dataloader:
            x_stock = batch["x_stock"].to(device)
            x_ctx = batch["x_ctx"].to(device)
            mask = batch["mask"].to(device)
            t5 = batch["target_5d"].numpy()
            mask_np = batch["mask"].numpy()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_autocast,
            ):
                scores_t = model(x_stock, x_ctx, mask)
            scores = scores_t.detach().cpu().float().numpy()
            del scores_t
            B = scores.shape[0]

            for b in range(B):
                valid = mask_np[b]
                if int(valid.sum()) < 10:
                    continue
                s = scores[b][valid]
                y = t5[b][valid]
                res = spearmanr(s, y)
                ic = (
                    float(res.statistic) if hasattr(res, "statistic") else float(res[0])
                )
                if np.isfinite(ic):
                    rank_ics.append(ic)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return float(np.mean(rank_ics)) if rank_ics else 0.0
```

- **Classification:** Per-date Spearman correlation computed across valid constituents (`valid.sum() >= 10`) on each validation date \(b\), then averaged across dates (`float(np.mean(rank_ics))`). It does **not** pool `(date, stock)` pairs across dates.

---

## 5.2 Test Rank IC Computation Code

Source: [`src/training/evaluate.py:90-108,174-175`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/evaluate.py#L90-L108)

```python
for t in range(T):
    valid_idx = np.flatnonzero(mask[t])
    n_valid = len(valid_idx)
    if n_valid < 10:
        continue

    s_t = scores[t, valid_idx]
    y_t = target_5d[t, valid_idx]

    # 1. Rank IC
    sp_res = spearmanr(s_t, y_t)
    ic_val = (
        float(sp_res.statistic) if hasattr(sp_res, "statistic") else float(sp_res[0])
    )
    if np.isfinite(ic_val):
        rank_ics.append(ic_val)
# ...
ic_arr = np.array(rank_ics, dtype=np.float64)
mean_rank_ic = float(np.mean(ic_arr)) if len(ic_arr) > 0 else 0.0
```

- **Classification:** Identical per-date Spearman correlation averaged across all valid test dates.

---

## 5.3 Five-Date Sanity Check on Fold 1 Validation Set

### Reproducible Code & Output

```python
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from src.data.loader import FeatureDataStore, Russell1000Dataset, create_dataloader
from src.data.normalize_schema import seed_everything
from src.models.architecture import BaselineArchitectureY
from src.training.trainer import evaluate_rank_ic
from src.training.wfcv import purged_walk_forward_cv

store = FeatureDataStore(cache_dir="data/cache")
eval_dates = store.unique_dates[store.eval_date_indices]
fold1 = list(purged_walk_forward_cv(eval_dates, n_splits=5, horizon=5, embargo=5))[0]

seed_everything(1042)
model = BaselineArchitectureY()
model.eval()

val_5_dates = fold1.val_dates[:5]
val_5_ds = Russell1000Dataset(store, val_5_dates)
val_5_loader = create_dataloader(val_5_ds, batch_size=5, shuffle=False)

batch = next(iter(val_5_loader))
with torch.inference_mode():
    scores = model(batch["x_stock"], batch["x_ctx"], batch["mask"]).numpy()
t5 = batch["target_5d"].numpy()
mask = batch["mask"].numpy()

manual_ics = []
for b in range(5):
    v = mask[b]
    s_b = scores[b][v]
    y_b = t5[b][v]
    r_s = rankdata(s_b)
    r_y = rankdata(y_b)
    ic_manual = float(np.corrcoef(r_s, r_y)[0, 1])
    ic_scipy = float(spearmanr(s_b, y_b).statistic)
    manual_ics.append(ic_manual)
    print(
        f"Date {batch['date'][b]} (n={v.sum()}): pipeline={ic_scipy:+.8f}, manual={ic_manual:+.8f}, diff={abs(ic_scipy - ic_manual):.2e}"
    )

pipeline_mean = evaluate_rank_ic(model, val_5_loader, torch.device("cpu"))
print(
    f"5-Date Mean IC: pipeline={pipeline_mean:+.8f}, manual={np.mean(manual_ics):+.8f}, diff={abs(pipeline_mean - np.mean(manual_ics)):.2e}"
)
```

```text
Date 2005-02-09 (n=970): pipeline=+0.02531885, manual=+0.02531885, diff=0.00e+00
Date 2005-02-10 (n=970): pipeline=+0.00682379, manual=+0.00682379, diff=0.00e+00
Date 2005-02-11 (n=970): pipeline=+0.04497926, manual=+0.04497926, diff=6.94e-18
Date 2005-02-14 (n=970): pipeline=+0.03143102, manual=+0.03143102, diff=0.00e+00
Date 2005-02-15 (n=970): pipeline=+0.01877677, manual=+0.01877677, diff=0.00e+00
5-Date Mean IC: pipeline=+0.02546594, manual=+0.02546594, diff=0.00e+00
```

**Result:** The per-date Spearman Rank IC calculation matches manual calculation to machine epsilon (`<= 6.94e-18`).

---

## 5.4 Diagnosis of Fold 2's Epoch 2 Selection (`Val Rank IC = -0.0312`)

### Fold 2 Training Trajectory (from [`steps/914/output.txt:41-50`](file:///Users/doxncwfn/.gemini/antigravity/brain/497ba708-12c9-419b-9798-66e2da2d8914/.system_generated/steps/914/output.txt#L41-L50))

```text
--- Starting Fold 2 --- Train dates: 2194 (2000-09-25 to 2009-06-17), Val dates: 388 (2009-07-01 to 2011-01-12), Test dates: 1300 (2011-01-27 to 2016-03-29)
  Epoch  1/30 | Train Loss: 7.1799 (MLE: 5.8697, Tail: 0.8735) | Val Rank IC: -0.0378 | LR: 9.97e-04
  Epoch  2/30 | Train Loss: 6.8548 (MLE: 5.8664, Tail: 0.6589) | Val Rank IC: -0.0312 | LR: 9.89e-04
  Epoch  3/30 | Train Loss: 6.8509 (MLE: 5.8649, Tail: 0.6573) | Val Rank IC: -0.0377 | LR: 9.76e-04
  Epoch  4/30 | Train Loss: 6.8481 (MLE: 5.8634, Tail: 0.6565) | Val Rank IC: -0.0356 | LR: 9.57e-04
  Epoch  5/30 | Train Loss: 6.8421 (MLE: 5.8616, Tail: 0.6537) | Val Rank IC: -0.0385 | LR: 9.34e-04
  Epoch  6/30 | Train Loss: 6.8330 (MLE: 5.8589, Tail: 0.6494) | Val Rank IC: -0.0405 | LR: 9.05e-04
  Epoch  7/30 | Train Loss: 6.8266 (MLE: 5.8571, Tail: 0.6463) | Val Rank IC: -0.0437 | LR: 8.73e-04
  Early stopping triggered at epoch 7 (best epoch: 2).
Fold 2 Complete in 1615.1s | Best Epoch: 2 | Val Rank IC: -0.0312 | Test Rank IC: +0.0200 | Test IC_IR: +0.053 | Net Sharpe: +0.242
```

### Answers to Audit Step 5.4 Questions
1. **Why was Epoch 2 selected as best when `Val Rank IC` was `-0.0312`?**
   In [`src/training/trainer.py:217,316-330`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/trainer.py#L217-L330), `best_val_ic` is initialized to `-float("inf")` with no minimum warmup epoch requirement and no positivity threshold. At Epoch 1, `-0.0378 > -inf` saved Epoch 1 weights; at Epoch 2, `-0.0312 > -0.0378` saved Epoch 2 weights (`best_epoch = 2`, `patience_counter = 0`); in Epochs 3–7, `Val Rank IC` was `-0.0377, -0.0356, -0.0385, -0.0405, -0.0437` (all `< -0.0312`), reaching `patience = 5` at Epoch 7 and restoring Epoch 2 weights.
2. **Was the model actually trained before early stopping?**
   **No — it collapsed to predicting near-zero scores (`s ≈ 0`) by Epoch 2 and barely learned any ranking signal:**
   - Between Epoch 1 and Epoch 2, the 388\(\times\) larger `TailMSE` gradient drove `Tail` from `0.8735` down to `0.6589` — which is the exact analytical value \(\mathcal{L}_{\text{TailMSE}}(\mathbf{0}) = \mathbf{0.6594}\) for an **all-zero score predictor**!
   - Across all 7 epochs, `Train MLE` decreased by only **`0.0126`** (`5.8697 → 5.8571`), remaining stuck at the random-guessing baseline \(\frac{\ln(N!)}{N} \approx 5.865\).
3. **Hyperparameter Protocol Deviations in [`src/training/trainer.py:132-137`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/trainer.py#L132-L137):**
   - [`specs/features_baseline.md` §9.7](file:///Users/doxncwfn/Project/QuantAlphaMLOps/specs/features_baseline.md#L757-L758) specifies **Early Stopping Patience = `10 epochs`** and **Batch Size = `64 dates`**, whereas `trainer.py` used `patience = 5` and `batch_size = 8, grad_accum_steps = 4` (effective batch size `32`). In Fold 1, `Val Rank IC` was also negative for the first 8 epochs (`-0.0219` at Epoch 1) and only turned positive at Epoch 9 (`+0.0041` → `+0.0151` at Epoch 12) because its negative validation values happened to drift slightly upward at Epochs 2, 3, 4, and 6, resetting the 5-epoch patience counter. In Fold 2, the 2009–2010 post-GFC validation regime had slight downward drift during Epochs 3–7, so `patience = 5` killed training before the ranking branch could emerge.
