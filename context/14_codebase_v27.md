# 14 · Kiến trúc Codebase `v27/` & Lệnh chạy

Mỗi thư mục trong `v27/` đảm nhiệm **đúng 1 trách nhiệm**, cấu hình tập trung tại `config/v27.yaml`:
- `eda/` (`universe_eda.py`): thống kê dữ liệu & churn universe.
- `data/` (`warehouse_loader.py`, `bundle.py`): nạp DuckDB, build cache, load `V7DataBundle` bằng Polars.
- `features/` (`stock_features.py`, `normalizer.py`, `market_context.py`): 34 features, RankGauss, 28 macro features.
- `targets/` (`target_builder.py`, `batch_sampler.py`): targets `1d/5d` + biến rủi ro trễ `(t-1)`.
- `models/` & `training/`: `HybridLstmTfm`, Spearman loss, 14-fold WF trainer.
- `signal_analysis/` & `explainability/`: audit `1d vs 5d`, alpha decay, MASTER Gate, 3F/FF6, tương quan & regime.
- `portfolio/`, `optimization/`, `risk/`, `experiments/`, `backtest/`, `visualization/`: xây danh mục, tối ưu, overlay, backtest, biểu đồ.
- `visualization/` ([style.py](../v27/visualization/style.py), [explainability_plots.py](../v27/visualization/explainability_plots.py), [portfolio_plots.py](../v27/visualization/portfolio_plots.py), [pnl_plots.py](../v27/visualization/pnl_plots.py)): xuất 20 biểu đồ (`fig03_*`, `fig04_*`, `fig08_01..07`, `fig10_01`), tự động tô vùng 4 giai đoạn khủng hoảng (`CRISIS_ERAS`) trên mọi đồ thị chuỗi thời gian.

## Lệnh chuẩn
```bash
uvx ruff check v27 && uvx ruff format --check v27
.venv/bin/pytest -q v27/tests
.venv/bin/python -m v27.pipelines.run_eda --output-dir outputs/v27 --from-duckdb
.venv/bin/python -m v27.pipelines.run_portfolio_suite --qh-dir outputs/QH_v7 --output-dir outputs/v27
```
