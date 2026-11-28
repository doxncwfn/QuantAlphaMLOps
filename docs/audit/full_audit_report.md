# Consolidated Pipeline Audit Report (v1.3 Baseline 1A)

**Audit Deliverables:**
1. **Step 1 (Loss Verification):** [`docs/audit/loss_verification.md`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/docs/audit/loss_verification.md)
2. **Step 2 (Gradient Verification):** [`docs/audit/gradient_verification.md`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/docs/audit/gradient_verification.md)
3. **Step 3 (Portfolio Verification):** [`docs/audit/portfolio_verification.md`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/docs/audit/portfolio_verification.md)
4. **Step 4 (Split & Loader Verification):** [`docs/audit/split_verification.md`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/docs/audit/split_verification.md)
5. **Step 5 (Metric Verification):** [`docs/audit/metric_verification.md`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/docs/audit/metric_verification.md)
6. **Step 6 (Compute Verification):** [`docs/audit/compute_verification.md`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/docs/audit/compute_verification.md)

---

## 1. Summary of Findings (One Sentence per Audit Step)

- **Step 0 (Package Integrity Check):** **FAIL** — Importing [`src/training/trainer.py`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/trainer.py#L24) currently throws `ImportError: cannot import name 'generate_baseline_1a_report' from 'src.training.evaluate'` because `evaluate.py` was modified on disk at `2026-11-06 17:53` while the training job was still running in memory.
- **Step 1 (Loss Function):** **FAIL** — [`src/losses/combined.py`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/losses/combined.py#L66-L126) implements quadratic **TailMSE** (`(s - y_soft) ** 2`) instead of the v1.3 Softplus tail loss (`softplus(-y_tilde * s)`) or pure ListMLE (§0.5.1), causing all-zero score collapse (`Tail = 0.6594`, `MLE = 5.865`), and silently returns `0.0` on 1D inputs.
- **Step 2 (Gradient Flow):** **FAIL** — On Fold 1 Epoch 1 Batch 1, `grad_mle / grad_tail` on `model.scorer.weight` is **`0.002522`** (MPS) / **`0.002675`** (CPU) (`0.001681` with \(\lambda = 1.5\)) vs. expected **`0.978`** (`[0.5, 3.0]`) because uncentered scores (\(\bar{s}_b \approx -1.30\)) project onto the Cross-Sectional Transformer's common-mode mean vector (\(\|\bar{\mathbf{h}}_b\|_2 = 11.10\), \(96.2\%\) of embedding norm), blowing up the non-shift-invariant tail gradient by \(388\times\) and starving ListMLE under gradient clipping.
- **Step 3 (Portfolio Construction):** **FAIL** — [`src/training/evaluate.py`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/evaluate.py#L90-L209) compounds 5-day forward returns **every single day** (`1,300` overlapping periods instead of `260` non-overlapping 5-day periods, causing `-99.99%` max drawdown), divides 1-day overlapping turnover by `2.0` (`0.3035` vs. expected `1.215`), and injects a hardcoded `1.0` turnover on date 0 (`1.50` on a 2-date full swap and `0.50` on zero change).
- **Step 4 (Train/Val/Test Partition & Data Loader):** **FAIL** — While [`purged_walk_forward_cv`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/wfcv.py#L45-L114) enforces exact 10-session (`horizon=5 + embargo=5`) non-overlapping gaps with zero date overlap, [`src/data/loader.py`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/data/loader.py#L107-L218) divides Blom ranks by total universe size `_n = pl.col("permno").count().over("date")` instead of per-feature non-null count `pl.col(c).count().over("date")` (truncating positive feature tails to `+1.53..+2.32`), starts evaluation on `2000-09-25` instead of after the 332-session warmup (`~2001-10`), and zero-fills early context (`2000-09-25..2000-12-18`) and missing stock window steps (`2.47%`).
- **Step 5 (Metric Computation & Early Stopping):** **CONDITIONAL FAIL** — Per-date Spearman Rank IC is exact to machine precision (`diff <= 6.94e-18`), but [`src/training/trainer.py`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/trainer.py#L132-L137) uses `patience = 5` (instead of `10`), effective `batch_size = 32` (instead of `64`), and no warmup guard on `best_val_ic = -inf`, causing Fold 2 to lock in Epoch 2 (`Val IC = -0.0312`) after collapsing to all-zero scores.
- **Step 6 (Compute Target):** **FAIL** — Executed on local Apple Silicon `MPS` with `torch.use_deterministic_algorithms(True, warn_only=True)` and non-deterministic `index_put_with_accumulate_mps` instead of deterministic `CUDA` (Modal A10G), with `574.9 min` (`77.5%`) of Fold 4's `742.0 min` wall-clock time caused by macOS system sleep (`526.7 min`) and manual `SIGSTOP` (`48.2 min`).

---

## 2. Root Cause of Each Failure (With Evidence)

1. **Why `Train MLE` Barely Moved (`5.8633 → 5.8224` in Fold 1; `5.8697 → 5.8571` in Fold 2):**
   - At Epoch 1 Batch 1, `tfm_out` has a date-level mean vector \(\bar{\mathbf{h}}_b\) of norm `11.10` (`96.2%` of total embedding energy) and cross-sectional stock dispersion \(\boldsymbol{\delta}_{b,i}\) of RMS norm `2.18`.
   - `listmle_loss` is shift-invariant (\(\sum_i \frac{\partial \mathcal{L}_{\text{MLE}}}{\partial s_{b,i}} \equiv 0\)), so its gradient with respect to `scorer.weight` is orthogonal to \(\bar{\mathbf{h}}_b\) (`grad_mle = 0.076269`).
   - Uncentered `TailMSE` has \(\sum_i \frac{\partial \mathcal{L}_{\text{TailMSE}}}{\partial s_{b,i}} \approx 2\bar{s}_b \approx -2.60 \neq 0\), which multiplies \(\bar{\mathbf{h}}_b\) to produce `grad_tail = 28.513250` (`ratio = 0.002675`). Even replacing `TailMSE` with `Softplus` on uncentered scores yields `grad_tail = 2.102928` (`ratio = 0.036268 < 0.5`) because \(\sum_i \frac{\partial \mathcal{L}_{\text{Softplus}}}{\partial s_{b,i}} \approx \frac{1}{\sum w_j}\sum_i w_i \tilde{y}_i \tanh(\frac{1}{2}\tilde{y}_i \bar{s}_b) \neq 0\) when \(\bar{s}_b \neq 0\).
   - Cross-sectionally centering scores per date (\(s_{b,i} \leftarrow s_{b,i} - \bar{s}_b\) over valid constituents) makes any downstream tail loss shift-invariant, eliminating the \(\bar{\mathbf{h}}_b\) common-mode explosion and restoring `grad_mle / grad_tail` to `[0.69, 2.24]` (`0.9042` with cross-sectional z-scoring, matching `0.978`).
   - Furthermore, under quadratic `TailMSE`, predicting identically zero scores \(\mathbf{s} \equiv \mathbf{0}\) achieves \(\mathcal{L}_{\text{TailMSE}}(\mathbf{0}) = 0.659432\) and \(\mathcal{L}_{\text{ListMLE}}(\mathbf{0}) \approx 5.865\) (`Total = 6.854`), which matches the Epoch 2 loss of every fold (`Tail = 0.6584–0.6674`, `MLE = 5.8587–5.8721`).

2. **Why `max_drawdown` Was `-99.99%` (`-0.9999`) and `turnover` Was `0.3035` (`4x` Too Low):**
   - [`src/training/evaluate.py:205`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/evaluate.py#L205) computes `nav = np.cumprod(1.0 + ls_net)` over all `1,300` daily test rows, where each `ls_net[t]` is a **5-trading-day return** \(R_{t \to t+5}^{\text{LS}}\). Compounding a 5-day return every day compounds `6,500` trading days (`25.8 years`) of return and variance into a 5.2-year test window.
   - [`src/training/evaluate.py:161`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/evaluate.py#L161) computes `turn_t = sum(|w_{t} - w_{t-1}|) / 2.0` on **1-day overlapping steps** and divides by `2.0`, cutting both reported turnover and transaction costs by \(4\times\) relative to a 5-day non-overlapping rebalance without the `/ 2.0` divisor. Line 165 also appends `1.0` on date `t = 0`.

3. **Why Fold 1 Had `top_decile_precision = +0.1139 > 0.10` But `ls_spread_gross = -0.0043 < 0`:**
   - Because `TailMSE` weights extreme ranks (\(|y^*| \ge 0.4\)) by \(4\times\), the model places high-volatility stocks into both predicted tails (`pred_top` and `pred_bot`). High-volatility stocks land in both `act_top` (\(>10\%\)) and `act_bot` (\(>10\%\)), inflating one-tail hit rate (`top_decile_precision = 0.1139`, `bottom_decile_precision = 0.1349`) while wrong-tail crashes and 2009 short-squeeze reversals make the mean spread negative (`-0.0043`). In Folds 2 & 3, the model collapsed onto `-vol_20d`/`-vol_60d`, producing `top_decile_precision ≈ 0.05`, `bottom_decile_precision ≈ 0.22–0.24`, and `turnover = 0.0445`.

4. **Why Blom Normal Scores Were Truncated in `src/data/loader.py`:**
   - [`src/data/loader.py:112-119`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/data/loader.py#L112-L119) divides `pl.col(c).rank("average").over("date")` by `_n = pl.col("permno").count().over("date")` (total universe count including nulls) instead of `pl.col(c).count().over("date")` (non-null count for feature `c`). For features with missing values (`resid_mom_20`, `resid_vol_20`, `resid_ret`, `drawdown_20`), the maximum quantile never reaches `1.0`, truncating the positive tail of `sp.ndtri` to `+1.53..+2.38` while the negative tail reaches `-3.22`.

---

## 3. Ranked List of Bugs to Fix (by Severity)

| Priority | ID | Component | Bug Summary |
|:---:|:---:|---|---|
| **P0 — Critical** | **BUG-01** | [`src/training/trainer.py:24,550`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/trainer.py#L24), [`src/training/evaluate.py`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/evaluate.py) | `ImportError: cannot import name 'generate_baseline_1a_report'` blocks all imports of `src.training.*`. |
| **P0 — Critical** | **BUG-02** | [`src/losses/combined.py:66-126`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/losses/combined.py#L66-L126), [`src/models/architecture.py:175-184`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/models/architecture.py#L175-L184) | `tail_loss` implements quadratic `TailMSE` instead of Softplus tail loss (`softplus(-y_tilde * s)`), and uncentered scores project onto `tfm_out`'s common mode (`||h_mean|| = 11.10`), causing a \(388\times\) gradient imbalance (`grad_mle / grad_tail = 0.0025`) and Epoch 2 all-zero score collapse. Both loss functions also return `0.0` on 1D inputs. |
| **P0 — Critical** | **BUG-03** | [`src/training/evaluate.py:90-209`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/evaluate.py#L90-L209) | `evaluate_predictions` compounds 5-day returns daily (`1,300` times instead of `260` non-overlapping 5-day periods, causing `-99.99%` max drawdown), computes turnover on 1-day steps divided by `2.0` (`4x` too low), and injects `1.0` turnover at `t = 0`. |
| **P1 — High** | **BUG-04** | [`src/data/loader.py:107-119`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/data/loader.py#L107-L119) | Blom normal score denominator uses total universe count `pl.col("permno").count().over("date")` instead of per-feature non-null count `pl.col(c).count().over("date")`, truncating positive feature tails (`+1.53..+2.38` vs `-3.22`). |
| **P1 — High** | **BUG-05** | [`src/data/loader.py:90-218`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/data/loader.py#L90-L218) | Missing 332-session warmup cutoff (`eval_date_indices` starts on `2000-09-25` instead of `~2001-10`), zero-filling of the first 60 context dates (`2000-09-25..2000-12-18`), and zero-filling of missing stock lookback steps (`2.47%` of Fold 1 valid windows). |
| **P1 — High** | **BUG-06** | [`src/training/trainer.py:132-137,316-330`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/trainer.py#L132-L137) | Hyperparameter deviations from v1.3 §9.7 (`patience = 5` instead of `10`, effective `batch_size = 32` instead of `64`) and no warmup guard before snapshotting negative `best_val_ic`. |
| **P1 — High** | **BUG-07** | [`src/training/trainer.py:28-34`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/training/trainer.py#L28-L34), [`src/data/normalize_schema.py:43`](file:///Users/doxncwfn/Project/QuantAlphaMLOps/src/data/normalize_schema.py#L43) | Non-deterministic `MPS` execution (`index_put_with_accumulate_mps` with `warn_only=True`) instead of strict deterministic `CUDA` execution, plus wall-clock `time.time()` timing that includes OS sleep/suspension. |

---

## 4. Recommended Fix Order

1. **Fix BUG-01 (`src/training/evaluate.py` / `src/training/trainer.py`):** Restore `generate_baseline_1a_report` (or clean up the import) so `src.training.*` imports cleanly.
2. **Fix BUG-02 (`src/losses/combined.py` & score centering/standardization):**
   - Implement the exact v1.3 Softplus tail loss \(\mathcal{L}_{\text{tail}} = \frac{\sum_i w_i \cdot \text{softplus}(-\tilde{y}_i \cdot s_i)}{\sum_i w_i}\), support 1D and 2D tensors, and cross-sectionally center/standardize valid scores per date so \(\sum_{i \in U_b^{\text{valid}}} \frac{\partial \mathcal{L}_{\text{tail}}}{\partial s_{b,i}} = 0\).
   - Re-run Audit Step 1.3 & 1.4 toy tests and Audit Step 2 gradient verification to confirm `grad_mle / grad_tail` is in `[0.5, 3.0]` (`≈ 0.978`).
3. **Fix BUG-04 & BUG-05 (`src/data/loader.py`):**
   - Replace `_n = pl.col("permno").count().over("date")` with per-feature non-null count `pl.col(c).count().over("date")` in Blom normal score computation.
   - Load pre-`2000-06-30` market factor history for the 252-day context rolling z-score, enforce the 332-session warmup cutoff (`~2001-10`) and full 60-session lookback validity mask without zero-filling, and rebuild `data/cache/silver_features_daily_store.npz`.
4. **Fix BUG-03 (`src/training/evaluate.py`):**
   - Compute portfolio returns, turnover, and cumulative NAV on the **non-overlapping 5-trading-session rebalance schedule** (§0.6.1), remove the `/ 2.0` division on turnover (§0.6.3), and remove the hardcoded `1.0` turnover on date 0.
   - Re-run Audit Step 3.1, 3.2, and 3.3 toy tests to confirm exact match.
5. **Fix BUG-06 & BUG-07 (`src/training/trainer.py`, `src/data/normalize_schema.py`, `src/modal/train.py`):**
   - Set `patience = 10` and effective `batch_size = 64` per v1.3 §9.7, persist fold checkpoints to disk, enforce strict `torch.use_deterministic_algorithms(True)`, and run on `CUDA` (Modal A10G).

---

## 5. Exit Criteria Before Training Can Resume

Training **must not resume** until every one of the following automated checks passes:

1. **Import Check:** `.venv/bin/python -c "import src.training.trainer"` exits with code `0`.
2. **Step 1 Loss Toy Tests:**
   - `listmle_loss` and `tail_loss` accept both 1D `[K]` and 2D `[B, K]` inputs.
   - `tail_loss` uses the v1.3 Softplus formulation (`Perfect < Random < Bad`).
3. **Step 2 Gradient Flow Gate:**
   - On Fold 1, Epoch 1, Batch 1 (`seed = 1042`), `grad_mle / grad_tail` on `model.scorer.weight` lies within `[0.5, 3.0]` (target `≈ 0.978`), with zero common-mode gradient explosion.
4. **Step 3 Portfolio & Turnover Gate:**
   - 10-stock toy test returns `ls_spread_gross = 0.10`, `top_decile_precision = 1.0`, `bottom_decile_precision = 1.0`.
   - 2-date turnover toy test returns `2.0` for a full swap and `0.0` for no change.
   - `nav` and `max_drawdown` are compounded strictly over non-overlapping 5-day rebalance periods.
5. **Step 4 Data & Split Gate:**
   - Every stock feature in `silver_features_daily_store.npz` has symmetric Blom normal scores (`|mean| < 1e-3`, `min ≈ -3.22`, `max ≈ +3.22`).
   - Zero context rows or valid stock lookback windows contain synthetic zero-filled missing values; `gap_tr_va >= 10` and `gap_va_te >= 10` on all 4 folds.
6. **Step 5 & 6 Protocol & Compute Gate:**
   - `patience = 10`, effective `batch_size = 64`, `torch.use_deterministic_algorithms(True)` (without `warn_only=True`), executing on `CUDA`.
