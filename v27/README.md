# v27 — Modular Quantitative Alpha & Portfolio Architecture

Clean, single-responsibility refactor of the frozen Semester 261 research stack ([SEMESTER_261_FINAL.md](file:///Users/doxncwfn/Project/QuantAlphaMLOps/outputs/QH_v7/v7_ensemble/SEMESTER_261_FINAL.md)).

## Package Layout (One Job Per Module)

| Module | Single Responsibility |
|---|---|
| `v27/config.py` | Immutable dataclasses (`PathConfig`, `DataConfig`, `ModelConfig`, `TrainConfig`, `PortfolioBacktestConfig`, `RiskOverlayConfig`) |
| `v27/eda/` | Warehouse and cached universe coverage, missingness, churn, and dispersion EDA |
| `v27/data/` | DuckDB `silver.v_prices` + Top-1000 hysteresis ingestion (`warehouse_loader.py`) and aligned OOS `V7DataBundle` loader (`bundle.py`) |
| `v27/features/` | 34 vectorized/GPU stock features (`stock_features.py`), RankGauss/MAD normalizer (`normalizer.py`), 28 causal market features (`market_context.py`) |
| `v27/targets/` | Zero-leakage `(t-1)` risk arrays + `raw_1d`/`risk_1d`/`raw_5d`/`risk_5d` targets (`target_builder.py`) and batch generator (`batch_sampler.py`) |
| `v27/models/` | `MarketFeatureGate`, `TemporalLstmAttention`, `CrossSectionalTransformerHead`, composite `HybridLstmTfm`, differentiable Spearman losses |
| `v27/training/` | Dynamic-purge 14-fold WF splits (`folds.py`), validation IC/TailIC/Hit (`metrics.py`), fold trainer + OOS gate exporter (`trainer.py`) |
| `v27/signal_analysis/` | 1d vs 5d horizon accounting audit (`horizon_audit.py`), `HOLD=1..5` alpha decay (`alpha_decay.py`), Sharpe persistence (`score_persistence.py`) |
| `v27/explainability/` | MASTER Gate 5-cluster weights, 3F/FF6 HAC & 12-sector attribution, 24-candidate correlation/overlap, MCR/CCR risk share, causal regimes |
| `v27/portfolio/` | Beta neutralization & leg renorm (`neutralization.py`), 8 constructors + Ortho-8 selector (`constructors.py`), position consolidation (`consolidation.py`) |
| `v27/optimization/` | Ledoit-Wolf / Bayes-Stein shrinkage (`estimators.py`), ERC/MinVar/MaxSharpe/CVaR LP solvers (`solvers.py`), causal WF schedules, Method D |
| `v27/risk/` | Single-name/gross/beta/sector/ADV constraints (`constraints.py`) and Method G causal vol-target/DD/CVaR overlays (`method_g_overlay.py`) |
| `v27/experiments/` | 24-candidate parity audit, Method A static, Method B score blends, Method E dynamic hysteresis, Method F hierarchical, Section 10 ablations |
| `v27/backtest/` | Daily MTM + linear TC/borrow/financing/impact (`simulator.py`), equity/DD/BRAIN metrics (`metrics.py`), LW/bootstrap/table exporter (`statistics.py`) |
| `v27/visualization/` | Explainability plots (`explainability_plots.py`), portfolio NAV/frontier/fold charts (`portfolio_plots.py`), `$20M` dollar PnL plots (`pnl_plots.py`) |
| `v27/pipelines/` | Clean CLI entrypoints: `run_eda.py`, `run_alpha.py`, `run_portfolio_suite.py` |

## Commands

```bash
.venv/bin/pytest -v v27/tests
.venv/bin/python -m v27.pipelines.run_eda --output-dir outputs/v27
.venv/bin/python -m v27.pipelines.run_portfolio_suite --qh-dir outputs/QH_v7 --output-dir outputs/v27
```
