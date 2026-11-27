# Pre-Notebook Cloud Training & Evaluation Contract

**Status:** Pre-registered specification before cloud GPU execution (post-A2/post-audit warehouse).  
**Source commit baseline:** `fix/pipeline-audit-20261111` (requires live `store` `target_hash = cf61b95f73a5b458030965d99496df536c3c29afe55fd9e98befa3c2fbaddedc` computed by `scripts/verify_artifacts.py`).

---

## 6.1 Experimental Grid (24 Runs)

Every run uses `BaselineArchitectureY` (`596,546` parameters, `d_model=128`, `lstm_hidden=128`, `lstm_layers=2`, `context_hidden=32`, `n_heads=4`, `tfm_layers=2`, `ff_dim=256`, `dropout=0.1`) trained via `src.training.trainer.train_one_fold` across all 4 NF-WFCV folds (`purged_walk_forward_cv(..., n_splits=5, horizon=5, embargo=5, val_ratio=0.15)`):

| Run Group | `loss_type` | `loss_kwargs` / `lambda_hub` | Folds | Seeds | Runs |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **G1: Baseline** | `"combined"` | `lambda_hub=0.0` (defaults: `CombinedRankingLoss`) | `1, 2, 3, 4` | `42, 1042` | 8 |
| **G2: S1a** | `"decile_mono"` | `K=10, tau_sort=0.01, m_mono=0.02, lam_ls=0.5, lam_hub=0.0, beta=0.1` | `1, 2, 3, 4` | `42, 1042` | 8 |
| **G3: S1b** | `"decile_mono"` | `K=10, tau_sort=0.01, m_mono=0.02, lam_ls=0.5, lam_hub=0.2, beta=0.1` | `1, 2, 3, 4` | `42, 1042` | 8 |
| **Total** | | | **4** | **2** | **24** |

### Locked Training Protocol (Shared Across All 24 Runs)
- `batch_size = 8`, `grad_accum_steps = 8` (effective batch size `64` dates)
- `learning_rate = 1e-3`, `weight_decay = 1e-4`, `eta_min = 1e-5`
- `max_epochs = 30`, `patience = 10`, `warmup_epochs = 15`, `min_checkpoint_epoch = 15`, `val_ma_window = 3`
- `rebalance_stride = 5`, `horizon = 5` (primary non-overlapping evaluation reported via `res["metrics_stride5"]` alongside `res["metrics"]` and optional `evaluate_predictions(..., rebalance_stride=10)`)
- Legacy pre-audit artifacts (`models/checkpoints/v1.5_s1*.pt`, `data/cache/*.json` marked `LEGACY_PRE_AUDIT` by `scripts/verify_artifacts.py`) MUST NOT be mixed into any comparison table.

---

## 6.2 Primary Pre-Registered Decision Thresholds

These criteria determine (a) which loss wins and (b) whether the neural model earns its complexity over the linear 5-feature reversal composite (`mean Spearman IC = +0.016960`, `t = +9.94` across `6,228` eval dates; single best raw feature `ret_5d IC = -0.016155`, `t = -8.37`):

1. **Data & Provenance Gate (Mandatory Prerequisite):**
   - `python scripts/verify_artifacts.py --strict` passes on all 24 produced checkpoints (`INVALID == 0`, `LEGACY_PRE_AUDIT == 0` in the cloud run directory, matching live `target_hash = cf61b95f73a5b458...`).
2. **Model Beats Simple Reversal Baseline (Complexity Justification):**
   - Winning group (`G1`, `G2`, or `G3`) achieves pooled OOS non-overlapping `stride=5` **mean Test Rank IC >= +0.0220** (exceeding the 5-feature composite's `+0.01696` by at least `+0.0050`, ~30% relative uplift) **AND** fold-level paired comparison vs the composite clears statistical significance (`p < 0.05` across `n=4` folds).
   - **Statistical Power Amendment (Option B — 3 Conditions × 4 Folds × 2 Seeds):** Seed variance is NOT estimated. PRIMARY significance is tested at the fold level (`n=4` pairs). Minimum detectable effect is `~1.5σ_fold` at `α=0.05`, power `= 0.8`. A PRIMARY pass must clear this threshold; otherwise the win is reported as directional but not statistically significant. (To estimate seed variance post-hoc for the winning group, a follow-up wave may add seeds `[43, 44]` without altering protocol.)
3. **Portfolio & Monotonicity Quality (c_bps + Turnover Gate):**
   - Mean OOS Net Sharpe (`stride=5`, `c_bps=5.0`) **>= +0.80** across the 4 folds (with `>= 3/4` folds Net Sharpe `> 0.0` at both seeds), exceeding the 5-feature composite (`+0.3129` at `c_bps=5.0` on all-eval).
   - **Turnover Gate (NEW):** Mean OOS half-turnover per 5-day rebalance **< 1.0** (vs composite's `1.62`) on `>= 2/4` folds. This ensures the winning model is economically viable at `c_bps=10`, not only at `c_bps=5`. Composite's Net Sharpe at `c_bps=10` is `-0.1737` on all-eval; a model that clears both gates would improve this.
   - Mean OOS top-bottom decile spread `(D10 - D1)` `> 0` cleanly monotonically (at least `7/9` adjacent decile transitions positive on average across folds for the winner; Spearman decile monotonicity `> 0.7`).
   - Report alongside: Gross Sharpe, `c_bps=10` stress Net Sharpe, `c_bps=15` stress Net Sharpe, per-rebalance half-turnover, and Net Sharpe at `c_bps=10` computed post-hoc from reported half-turnover.
   - **Dry-Run Baseline Exclusion:** Single-epoch smoke-test Net Sharpe (`+0.190` `decile_mono`, `-0.316` `combined` on Fold 1) is NOT a baseline for the 24-run grid; comparisons must be trained-vs-trained (30-epoch/early-stopped) or vs the deterministic 5-feature composite only.
4. **Cross-Seed & Cross-Fold Stability:**
   - Seed spread per fold `|IC(seed=42) - IC(seed=1042)| <= 0.008` in at least `3/4` folds.
   - Mean `score_std >= 0.03` on test dates (zero rank collapse).
5. **Economic Viability Diagnostics (Reported, Not Gated):**
   For the winning group, additionally report per-fold:
   - Net Sharpe at `c_bps = [5, 10, 15, 20]`
   - Half-turnover distribution (mean, median, p10, p90)
   - Cumulative gross return vs cumulative cost over test window
   - Decile-level turnover (how much does each decile churn?)
   These are diagnostics for a post-notebook economic assessment, NOT decision gates. They inform whether the model is viable at realistic costs, which the composite alone is not.

---

## 6.3 Hard Kill Criteria (Abort Early)

Stop an individual run or halt the sweep immediately if ANY of the following trigger:

1. **Cell 2 Pre-Flight Parity Failure (Option B Split Gate):** `python scripts/check_env.py --check-cuda-parity` fails fp32 loss-kernel parity on identical scores (`max|ΔL_kernel| >= 1e-5`), end-to-end forward parity (`max|ΔL| >= 1e-5` for `combined`, `>= 1e-4` for `decile_mono` under `tau_sort=0.01`), OR backward global gradient-norm relative parity (`| ||g_cuda||_2 - ||g_cpu||_2 | / ||g_cpu||_2 >= 1e-3`) between `cpu` and `cuda`.
2. **Rank Collapse:** Any run ends an epoch post-warmup (`epoch >= 15`) with validation `score_std < 0.005` on two consecutive epochs.
3. **Optimization Pathology:** Two or more gradient skips (`NaN`/`Inf` gradient norms caught by D5 guard) within a single epoch, or `train_loss` fails to decrease from epoch 1 to epoch 5 on Fold 1.
4. **Dead Signal:** After completing `G1` + `G2` on **Fold 1** (`seed=42`, first 2 runs), if BOTH achieve Test Rank IC `< +0.010` AND Net Sharpe `< 0.0`, pause before spending compute on Folds 2–4 to inspect feature/label alignment.
5. **Budget Cap:** Projected total sweep wall-clock cost from Cell 2 calibration exceeds **\$50.00** or **3x** expected runtime.

---

## 6.4 Measured Cost & Memory Model

### Locally Measured Shapes, Memory & Timings (`batch_size=8, K_max=1200`)
- **Model size:** `596,546` float32 parameters (`2.28 MB` weights, `2.29 MB` `.pt` on disk).
- **Peak memory:** Loading `.npz` store (`435.9 MB`) + 1 real Fold-1 fwd+bwd batch on CPU showed `+2,073.5 MB` RSS delta (`2336.1 -> 4409.7 MB`); full Fold-1 MPS dry run stayed under `1.9 GB` process RSS (`1549.2 -> 1874.4 MB`). Require `>= 16 GB` GPU VRAM (e.g., T4 16GB / L4 24GB / A10G 24GB / A100 40GB) and `>= 16 GB` system RAM.
- **Exact batch counts per NF-WFCV fold (`batch_size=8`):**
  - Fold 1: `1,042` train (`131` batches) / `185` val (`24` batches) / `1,245` test (`156` batches)
  - Fold 2: `2,100` train (`263` batches) / `372` val (`47` batches) / `1,245` test (`156` batches)
  - Fold 3: `3,159` train (`395` batches) / `558` val (`70` batches) / `1,245` test (`156` batches)
  - Fold 4: `4,217` train (`528` batches) / `745` val (`94` batches) / `1,245` test (`156` batches)
  - **One 4-fold pass totals:** `1,317` train batches/epoch, `235` val batches/epoch, `624` test batches/fold-set.
- **Locally observed compute speed (Phase 4):**
  - `CPU`: `5.943 s/batch` (`combined`), `5.639 s/batch` (`decile_mono`) — `~13.0 min/epoch` on Fold 1 (`~360 h` for 24 runs at 25 epochs; infeasible locally).
  - `MPS` (Apple Silicon M-series): `0.703 s/batch` (`combined`), `0.674 s/batch` (`decile_mono`); full Fold-1 1-epoch dry run including val + 1,245-date test eval took `167.47 s` (`combined`) and `136.92 s` (`decile_mono`). Across 24 runs at ~25 epochs (`197,550` train batches), MPS projection is `~38.5–45 h` (and excluded from canonical runs due to Metal non-deterministic `index_put_with_accumulate_mps` / Transformer reduction drift `max|Δgrad| = 1.958e-4`).
- **CUDA Note (Unmeasured Locally):** Local macOS host has `cuda=False`. Exact CUDA throughput (`s/batch`) and total budget MUST be measured empirically in **Cell 2** on a single 1-epoch Fold-1 calibration before running Cells 5–7. Conservatively assuming L4/A10G (`~0.15–0.35 s/batch`, ~2x–4x MPS), the 24-run sweep requires **~8–22 GPU-hours** (`~$8–$26` on Modal A10G at ~$1.10/hr or Colab Pro+ A100/L4 compute units).

---

## 6.5 10-Cell Cloud Notebook Structure

| Cell | Purpose & Exact Action |
| :---: | :--- |
| **Cell 1** | **Clone & Pin Env:** Checkout pinned `git_sha`, install `requirements.lock` (matching `pyproject.toml` pins including `torch==2.15.0`, `torchsort==0.1.10`, `duckdb==2.0.0`, `numpy==2.4.6`). Print GPU SKU & VRAM. |
| **Cell 2** | **Pre-Flight & CUDA Parity Gate (Option B Split Gate):** Mount/fetch `russell1000.duckdb` + `silver_features_daily_store.npz`. Run `python scripts/check_env.py --check-cuda-parity` (asserts `ENV_CHECK=PASS`, zero IEEE NaNs, exact `10,745,931` rows, fp32 forward parity `max\|ΔL\| < 1e-5`, and backward L2 gradient-norm relative parity `\| \|\|g_cuda\|\|_2 - \|\|g_cpu\|\|_2 \| / \|\|g_cpu\|\|_2 < 1e-3`; logs element-wise `max\|Δgrad\|` and verifies `bfloat16+float32_loss` autocast finite smoke). |
| **Cell 3** | **Cost Calibration (1-Epoch Smoke):** Run `Fold 1` `max_epochs=1` for `combined` & `decile_mono`, record exact `CUDA s/batch`, print extrapolated 24-run wall-clock & dollar estimate, enforce budget kill check (`<= $50`). |
| **Cell 4** | **Stage Run Manifest & Idempotent Runner:** Compute live `target_hash` (`cf61b95f...`), verify `git status --short` clean (or record commit), define atomic per-fold save helper (`run_id = {group}_fold{f}_seed{s}`) skipping runs that already pass `verify_artifacts.py`. |
| **Cell 5** | **Execute G1 (`combined`, `lam_hub=0.0`, 8 runs):** Train Folds 1–4 × Seeds `[42, 1042]`; persist `{ckpt, metadata.json, fold_diagnostics.json}` after each fold completes. |
| **Cell 6** | **Execute G2 (`decile_mono`, `lam_hub=0.0` [S1a], 8 runs):** Train Folds 1–4 × Seeds `[42, 1042]`; enforce Fold-1 dead-signal kill check after first run. |
| **Cell 7** | **Execute G3 (`decile_mono`, `lam_hub=0.2` [S1b], 8 runs):** Train Folds 1–4 × Seeds `[42, 1042]`. |
| **Cell 8** | **Artifact Verification Gate:** Run `python scripts/verify_artifacts.py --strict --checkpoint-dir <out_ckpts> --results-dir <out_json>` asserting `24/24` `VALID`, `0` `INVALID`, `0` `LEGACY_PRE_AUDIT`. |
| **Cell 9** | **Comparison vs 5-Feature Composite & Monotonicity Audit:** Aggregate pooled + per-fold `stride=5` & `stride=10` Rank IC, IC_IR, HAC Net Sharpe, decile curves (`D1..D10`), `score_std`, paired t-tests vs the `+0.01696` reversal composite, and post-hoc regime slice (`2006–2011`, `2011–2016`, `2016–2021`, `2021–2026`). |
| **Cell 10** | **Decision Verdict & Artifact Export:** Evaluate §6.2 thresholds verbatim (`SHIP` vs `PIVOT`), bundle final manifest + 24 verified artifacts, push/save report to persistent volume. |

---

## 6.6 Pivot Condition & Fallback Plan

- **Trigger:** All three model groups (`G1`, `G2`, `G3`) fail §6.2 Criterion 2 (`mean OOS Rank IC < +0.0220` OR fail statistical dominance over the 5-feature reversal composite `IC = +0.01696, t = +9.94`), OR `G2/G3` exhibit negative/flat decile monotonicity despite clean convergence.
- **Pre-Registered Fallback (No Post-Hoc Hyperparameter Fishing on Test Sets):**
  1. Do **NOT** retune `Architecture Y` hyperparameters or modify test partitions on the same OOS folds.
  2. Benchmark a strictly cross-sectional shallow baseline using the **existing 19 + 11 features** (`LightGBM` / `Ridge` ranker evaluated under the exact same 4 NF-WFCV folds and `compute_turnover_and_cost` / `evaluate_predictions` harness) to test whether sequence-LSTM + Transformer attention hurts signal-to-noise ratio on short-term reversal.
  3. If a linear/tree ranker outperforms `BaselineArchitectureY` at `< 1%` compute cost, promote the simpler ranker or multi-horizon reversal/volatility ensemble before introducing any new feature set (e.g., fundamental/medium-horizon factors).

---

## 6.7 Reproducibility & Resume Contract

- **Pinned Runtime & Drift Correction:** Cloud builds MUST install from `requirements.lock` (matching `pyproject.toml` exact `==` pins, Python `3.11.x`). Existing `src/modal/train.py` and `src/modal/run_s1_ablation.py` contain stale package pins (`torch==2.2.2`, `numpy==1.26.4`, `duckdb==1.1.3`; `train.py` lacks `torchsort`; `run_s1_ablation.py:281` reads removed key `metrics_stride10`) and MUST NOT be invoked as-is for the 24-run sweep.
- **Provenance Enforcement:** Every checkpoint `<run>.pt` MUST be accompanied by `<run>.metadata.json` containing all 12 verified keys (`loss_type`, `lam_hub`, `device`, `dtype`, `lr`, `weight_decay`, `warmup_epochs`, `min_checkpoint_epoch`, `seed`, `batch_size`, `git_sha`, `target_hash`). Any result lacking matching `target_hash` or `git_sha` is rejected by `verify_artifacts.py --strict`.
- **Supported Resume Granularity (Fold-Level Only):** `train_one_fold` has **NO** mid-epoch batch-level checkpointing or optimizer/scheduler/RNG state persistence (verified via mid-epoch `SIGINT` test in Phase 5.4: process exits `rc=-2` cleanly without corrupting existing files or saving partial state). Therefore, cloud interruption recovery operates strictly at **fold/run granularity**: write each run output atomically on completion (`tmp -> rename` + `volume.commit()` on Modal), and on restart skip only `{group}_fold{f}_seed{s}` runs whose `.pt` + `.metadata.json` pass `verify_artifacts.verify_checkpoint`.
