# Audit Step 6 — Compute Target & Runtime Verification (`src/training/trainer.py` & `src/data/normalize_schema.py`)

## 6.1 Hardware & Determinism Configuration

1. **Hardware Used:** Local Apple Silicon **`mps`** (`device: mps` in [`wandb/run-20261106_042422-ggo4p5lk/files/config.yaml:8`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/wandb/run-20261106_042422-ggo4p5lk/files/config.yaml#L8)).
   - In [`src/training/trainer.py:28-34`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/trainer.py#L28-L34), `get_default_device()` checks `if torch.backends.mps.is_available(): return torch.device("mps")` first.
   - Cloud execution on Modal (`A10G` / `L4` / `L40S`) had failed earlier with `modal.exception.InvalidError: Please add a payment method to use <GPU> functions`.
2. **Was `torch.use_deterministic_algorithms(True)` set?**
   - **Relaxed with `warn_only=True`:** In [`src/data/normalize_schema.py:43`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/data/normalize_schema.py#L43), `seed_everything` calls `torch.use_deterministic_algorithms(True, warn_only=True)` instead of strict `torch.use_deterministic_algorithms(True)` mandated by [`specs/features_baseline.md` §10.6](file:///Users/doxncwfn/Project/QuantAlphaMLOps/specs/features_baseline.md#L899).
3. **Was the `index_put_with_accumulate_mps` warning addressed?**
   - **No — it was only downgraded from a `RuntimeError` to a `UserWarning` via `warn_only=True`:**
     ```text
     /Users/doxncwfn/Project/QuantAlphaMLOps/.venv/lib/python3.11/site-packages/torch/autograd/graph.py:1113:
     UserWarning: index_put_with_accumulate_mps does not have a deterministic implementation,
     but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'.
     ```
   - Where `index_put_with_accumulate_mps` originates:
     1. [`BaselineArchitectureY.forward`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/models/architecture.py#L145-L157) chunks `flat_stocks` in slices `flat_stocks[i : i + 2048]` (`view` slicing); during backward on MPS, accumulating gradients into the parent tensor invokes `index_put_with_accumulate_mps`.
     2. [`listmle_loss` and `tail_loss`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/losses/combined.py#L43-L48) boolean-index `scores[b][valid]` and permutation-index `s[order]`, whose backward pass on MPS also dispatches to `index_put_with_accumulate_mps`.

---

## 6.2 Per-Epoch & Per-Fold Runtime Breakdown

Extracting the exact per-epoch wall-clock timestamps and W&B active process timer (`_runtime`, which pauses when the OS process is suspended or the host machine sleeps) from `wandb/run-20261106_042422-ggo4p5lk`:

| Fold | Train Dates (Ratio) | Val Dates | Epochs Run | Active Epoch Time Range (`_runtime`) | Total Active Compute (`_runtime`) | Reported Wall-Clock (`time.time()`) |
|:---:|---:|---:|:---:|---|---:|---:|
| **Fold 1** | `1,089` (\(1.00\times\)) | `193` | `17` | `99.4s – 115.2s` (`~1.75 min/ep`) | `1,817.7s` (`30.3 min`) | `1,817.7s` (`30.3 min`) |
| **Fold 2** | `2,194` (\(2.01\times\)) | `388` | `7` | `216.9s – 234.4s` (`~3.77 min/ep`) | `1,615.1s` (`26.9 min`) | `1,615.1s` (`26.9 min`) |
| **Fold 3** | `3,299` (\(3.03\times\)) | `583` | `15` | `315.7s – 370.6s` (`~5.70 min/ep`) | `5,171.5s` (`86.2 min`) | `5,171.5s` (`86.2 min`) |
| **Fold 4** | `4,404` (\(4.04\times\)) | `778` | `18` | `437.7s – 715.6s` (`~7.3–11.9 min/ep`) | **`10,029.8s` (`167.2 min`)** | **`44,518.8s` (`742.0 min`)** |
| **Total** | — | — | `57` | — | **`18,634.0s` (`310.6 min`)** | **`53,129.0s` (`885.5 min`)** |

### Exact Per-Epoch Timing of Fold 4 (Wall-Clock vs. Active Compute)

```text
Fold 4 Start: 2026-11-06 06:48:32
  Epoch  1 (06:59:24): wall_delta=  652.0s ( 10.9m) | active_delta= 652.0s (10.9m)
  Epoch  2 (07:11:19): wall_delta=  715.6s ( 11.9m) | active_delta= 715.6s (11.9m)
  Epoch  3 (07:21:41): wall_delta=  621.8s ( 10.4m) | active_delta= 621.8s (10.4m)
  Epoch  4 (07:30:27): wall_delta=  526.0s (  8.8m) | active_delta= 526.0s ( 8.8m)
  Epoch  5 (07:39:52): wall_delta=  565.1s (  9.4m) | active_delta= 565.1s ( 9.4m)
  Epoch  6 (07:55:19): wall_delta=  926.9s ( 15.4m) | active_delta= 537.5s ( 9.0m)  [+6.5m sleep]
  Epoch  7 (10:28:31): wall_delta= 9191.8s (153.2m) | active_delta= 485.8s ( 8.1m)  [+145.1m macOS sleep]
  Epoch  8–9 (17:05:07): wall_delta=23796.0s (396.6m) | active_delta=1439.9s (24.0m) [+372.6m macOS sleep]
  Epoch 10 (17:13:04): wall_delta=  476.8s (  7.9m) | active_delta= 476.8s ( 7.9m)
  Epoch 11 (17:20:32): wall_delta=  447.8s (  7.5m) | active_delta= 447.8s ( 7.5m)
  Epoch 12 (17:28:01): wall_delta=  449.2s (  7.5m) | active_delta= 449.2s ( 7.5m)
  Epoch 13 (17:35:23): wall_delta=  441.5s (  7.4m) | active_delta= 441.5s ( 7.4m)
  Epoch 14 (17:42:40): wall_delta=  437.7s (  7.3m) | active_delta= 437.7s ( 7.3m)
  Epoch 15 (17:50:19): wall_delta=  458.5s (  7.6m) | active_delta= 458.5s ( 7.6m)
  Epoch 16 (18:05:17): wall_delta=  898.3s ( 15.0m) | active_delta= 496.5s ( 8.3m)  [+6.7m sleep]
  Epoch 17 (19:01:27): wall_delta= 3369.5s ( 56.2m) | active_delta= 431.9s ( 7.2m)  [+49.0m manual SIGSTOP]
  Epoch 18 + Test Eval (19:09:50): wall_delta=503.1s (8.4m) | active_delta=503.1s (8.4m)
```

---

## 6.3 Diagnosis of Fold 4's \(25\times\) Runtime Inflation (`742.0 min` vs `30.3 min`)

The `44,518.8s` (`742.0 min`) reported for Fold 4 is the sum of two distinct factors:

1. **OS Suspension & System Sleep Counted by `time.time()` (`34,489.0s = 574.9 min`, `77.5%` of Fold 4 wall-clock):**
   - [`src/training/trainer.py:224,339`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/trainer.py#L224) computes `elapsed = time.time() - t_start` using wall-clock epoch time rather than active process execution time.
   - Between `07:55` and `17:05`, the local macOS machine entered system sleep for **`524.2 minutes`** (`8.74 hours`), during which W&B emitted `Failed to export metrics batch due to timeout, max retries or shutdown` upon wake.
   - Between `18:07:54` and `18:56:04`, the process was manually paused via `kill -STOP 67362` for **`48.2 minutes`**.
2. **Expanding-Window Compute Scaling on Local MPS (`10,029.8s = 167.2 min` active compute, `22.5%` of Fold 4 wall-clock):**
   - Because NF-WFCV uses an **expanding** training window, Fold 4 has **$4.04\times$ more training dates** (`4,404` dates = `551` micro-batches/epoch) and **$4.02\times$ more validation dates** (`778` dates = `98` micro-batches/epoch) than Fold 1 (`1,089` train dates = `137` micro-batches/epoch; `193` val dates = `25` micro-batches/epoch).
   - At Fold 1's speed of `105s/epoch`, 18 epochs of a $4.04\times$ larger dataset take $18 \times 105 \times 4.04 = 7,636\text{s}$ (`127.3 min`), with the remaining `~40 min` caused by MPS sequential LSTM unrolling over `2,048`-stock chunks (`661,200` 60-step sequences per epoch) and calling `torch.mps.empty_cache()` every epoch.
