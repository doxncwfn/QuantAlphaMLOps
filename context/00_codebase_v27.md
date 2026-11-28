# 14 · Kiến trúc Codebase [v27/](../v27/) & Lệnh chạy

Mỗi thư mục trong [v27/](../v27/) đảm nhiệm **đúng 1 trách nhiệm**, cấu hình tập trung tại [config/v27.yaml](../config/v27.yaml):
- [eda/](../v27/eda/) ([universe_eda.py](../v27/eda/universe_eda.py)): thống kê dữ liệu & churn universe.
- [data/](../v27/data/) ([warehouse_loader.py](../v27/data/warehouse_loader.py), [bundle.py](../v27/data/bundle.py)): nạp DuckDB, build cache, load `V7DataBundle` bằng Polars.
- [features/](../v27/features/) ([stock_features.py](../v27/features/stock_features.py), [normalizer.py](../v27/features/normalizer.py), [market_context.py](../v27/features/market_context.py)): 34 features, RankGauss, 28 macro features.
- [targets/](../v27/targets/) ([target_builder.py](../v27/targets/target_builder.py), [batch_sampler.py](../v27/targets/batch_sampler.py)): targets `1d/5d` + biến rủi ro trễ `(t-1)`.
- [models/](../v27/models/) & [training/](../v27/training/): `HybridLstmTfm`, Spearman loss, 14-fold WF trainer.
- [signal_analysis/](../v27/signal_analysis/) & [explainability/](../v27/explainability/): audit `1d vs 5d`, alpha decay, MASTER Gate, 3F/FF6, tương quan & regime.
- [portfolio/](../v27/portfolio/), [optimization/](../v27/optimization/), [risk/](../v27/risk/), [experiments/](../v27/experiments/), [backtest/](../v27/backtest/), [visualization/](../v27/visualization/): xây danh mục, tối ưu, overlay, backtest, biểu đồ.
- [cost_resilient/](../v27/cost_resilient/) ([signals.py](../v27/cost_resilient/signals.py), [optimizers.py](../v27/cost_resilient/optimizers.py), [pipeline.py](../v27/cost_resilient/pipeline.py), [reporting.py](../v27/cost_resilient/reporting.py)): làm mượt tín hiệu nhân quả (EMA/Kalman), cơ chế giảm turnover (NTB, Garleanu-Pedersen, throttle, `BetaNeu+HOLD`), bộ tối ưu Woodbury-ADMM phạt L1 trung hòa factor, và ensemble kháng chi phí trên `Net5`.
- [visualization/](../v27/visualization/) ([style.py](../v27/visualization/style.py), [explainability_plots.py](../v27/visualization/explainability_plots.py), [portfolio_plots.py](../v27/visualization/portfolio_plots.py), [pnl_plots.py](../v27/visualization/pnl_plots.py)): xuất 20 biểu đồ (`fig03_*`, `fig04_*`, `fig08_01..07`, `fig10_01`), tự động tô vùng 4 giai đoạn khủng hoảng (`CRISIS_ERAS`) trên mọi đồ thị chuỗi thời gian.

## Lệnh chuẩn
```bash
uvx ruff check v27 && uvx ruff format --check v27
.venv/bin/pytest -q v27/tests
.venv/bin/python -m v27.pipelines.run_eda --output-dir outputs/v27 --from-duckdb
.venv/bin/python -m v27.pipelines.run_portfolio_suite --qh-dir outputs/QH_v7 --output-dir outputs/v27
.venv/bin/python -m v27.pipelines.run_cost_resilient --qh-dir outputs/QH_v7 --output-dir outputs/QH_v7/v7_cost_resilient
```
