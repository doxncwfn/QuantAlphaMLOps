# Russell 1000 Baseline v1.0 Reproduction Report (Universe 1A)

**Execution Date:** 2026-11-07 05:23:40
**Specification:** Frozen v1.3 Baseline v1 (`specs/features_baseline.md` §8.2, §9.7)
**Universe Selection:** Full Point-in-Time Russell 1000 (`silver.features_daily`)
- **Audit Log:** [model_pipeline_rebuild_audit.md](file:///Users/doxncwfn/Project/QuantAlphaMLOps/docs/model_pipeline_rebuild_audit.md)
- **Universe Decision:** [universe_decision.md](file:///Users/doxncwfn/Project/QuantAlphaMLOps/docs/universe_decision.md)
- **W&B Live Experiment Dashboard:** [https://forge.coreweave.com/wandb/doxncwfn/russell1000-v13-baseline/runs/8pr1zvrz](https://forge.coreweave.com/wandb/doxncwfn/russell1000-v13-baseline/runs/8pr1zvrz)

---

## 1. Acceptance Gate Checklist (§7)

| Criterion | Target Requirement | Realized Result | Status |
|---|---|---|---|
| **1. Unit Test Suite** | 100% pass across data, loss, padding, WFCV | Verified | **PASS** |
| **2. Training Runtime** | Fold 1 runtime < 4h | 6607.8s | **PASS** |
| **3. Rank IC Stability** | Rank IC > 0 on >= 3 of 4 folds | 4 of 4 folds positive | **PASS** |
| **4. Net Sharpe Profitability** | Net Sharpe > 0 on >= 2 of 4 folds | 2 of 4 folds positive | **PASS** |
| **5. Numerical Stability** | Zero NaN divergence across all folds | All folds complete, loss finite | **PASS** |
| **6. Reproducibility** | Determinism verified with seed 42 | Exact bitwise matching | **PASS** |
| **7. Report Generated** | Complete metrics table in docs/baseline_1a.md | Generated with 12 core metrics | **PASS** |

**Overall Acceptance Gate Status: **PASS****

---

## 2. Walk-Forward Cross-Validation Performance (NF-WFCV)

| Metric | Description | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `rank_ic` | Rank IC (Spearman) | +0.0114 | +0.0021 | +0.0033 | +0.0346 | +0.0129 | 0.0131 |
| `ic_ir` | IC IR (HAC lag 4) | +0.0303 | +0.0065 | +0.0088 | +0.0875 | +0.0333 | 0.0327 |
| `top_decile_precision` | Top Decile Precision | +0.0510 | +0.0943 | +0.0461 | +0.0839 | +0.0688 | 0.0207 |
| `bottom_decile_precision` | Bottom Decile Precision | +0.2077 | +0.1718 | +0.2207 | +0.1789 | +0.1948 | 0.0201 |
| `ls_spread_gross` | LS Spread Gross (mean daily) | -0.0016 | +0.0017 | -0.0017 | +0.0041 | +0.0006 | 0.0024 |
| `ls_spread_net` | LS Spread Net (after 5bps tc) | -0.0025 | +0.0005 | -0.0024 | +0.0030 | -0.0003 | 0.0023 |
| `ls_sharpe_gross` | LS Sharpe Gross (annualized) | -0.2407 | +0.3387 | -0.3670 | +0.7147 | +0.1114 | 0.4383 |
| `ls_sharpe_net` | LS Sharpe Net (annualized) | -0.3720 | +0.0943 | -0.5136 | +0.5361 | -0.0638 | 0.4130 |
| `turnover` | Turnover (2-sided daily) | 1.7813 | 2.5293 | 1.3756 | 2.0436 | +1.9325 | 0.4188 |
| `max_drawdown` | Max Drawdown (cumulative NAV) | -0.7890 | -0.5277 | -0.6725 | -0.3146 | -0.5760 | 0.1770 |
| `benchmark_return` | Benchmark Return (mean daily) | +0.0013 | +0.0028 | +0.0031 | +0.0014 | +0.0021 | 0.0008 |
| `active_return` | Active Return (LS Gross - Bench) | -0.0029 | -0.0010 | -0.0048 | +0.0027 | -0.0015 | 0.0028 |

### Fold Training Metadata
| Fold | Best Epoch | Best Val Rank IC | Test Dates Count | Runtime |
|---|---|---|---|---|
| Fold 1 | Epoch 5 | +0.0393 | 1,245 dates | 424.9s (7.1 min) |
| Fold 2 | Epoch 18 | +0.0336 | 1,245 dates | 1567.2s (26.1 min) |
| Fold 3 | Epoch 5 | +0.0234 | 1,245 dates | 1270.7s (21.2 min) |
| Fold 4 | Epoch 28 | +0.0356 | 1,245 dates | 3335.7s (55.6 min) |
| **Total Pipeline** | - | - | 4,980 out-of-sample dates | 6607.8s (110.1 min) |

