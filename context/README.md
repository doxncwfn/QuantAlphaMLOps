# Context Hệ Thống (Semester 261 — [v27/](../v27/))

Tài liệu ngữ cảnh kỹ thuật cho hệ thống **Cross-Sectional Stock Ranking & Portfolio Optimization** (phiên bản chốt Semester 261, codebase [v27/](../v27/), cấu hình [v27.yaml](../config/v27.yaml)), sắp xếp theo đúng luồng dự án chuẩn:

| # | File | Chủ đề |
|---|---|---|
| 01 | [01_problem.md](01_problem.md) | Bài toán, phạm vi OOS & quy ước sổ `$20M` |
| 02 | [02_data.md](02_data.md) | Dữ liệu DuckDB `russell1000.duckdb`, cache `.pt` & Polars I/O |
| 03 | [03_universe_filter.md](03_universe_filter.md) | Bộ lọc Universe Top-1000 (`TTL=60`, `buffer=10%`) |
| 04 | [04_targets_and_formulation.md](04_targets_and_formulation.md) | Công thức hóa nhân quả & Targets `raw_1d/5d`, `risk_1d/5d` |
| 05 | [05_features.md](05_features.md) | 34 đặc trưng (5 cụm), RankGauss & 28 chỉ báo vĩ mô |
| 06 | [06_model.md](06_model.md) | Kiến trúc `HybridLstmTfm` (MASTER Gate + LSTM + CS-Transformer) |
| 07 | [07_objective_and_loss.md](07_objective_and_loss.md) | Hàm mất mát Spearman khả vi (`soft_rank`) & Early Stopping |
| 08 | [08_explainability_and_regimes.md](08_explainability_and_regimes.md) | Giải thích mô hình (MASTER Gate), bổ trợ, MCR & Chế độ thị trường |
| 09 | [09_strategies.md](09_strategies.md) | Hệ thống 25 chiến lược cơ sở (24 sleeves + Benchmark) & Tranches |
| 10 | [10_ensemble_and_combining.md](10_ensemble_and_combining.md) | Framework kết hợp, tối ưu (Methods A–G), Risk Overlays & Ablations |
| 11 | [11_backtest.md](11_backtest.md) | Mô phỏng Backtest MTM hàng ngày, netting & 3 tầng chi phí |
| 12 | [12_metrics.md](12_metrics.md) | Hệ thống Metric (Signal, Decile Monotonicity, Portfolio, HAC, LW, FF6) |
| 13 | [13_portfolio_results.md](13_portfolio_results.md) | Tổng kết kết quả OOS & khuyến nghị Flagship |
| 14 | [14_codebase_v27.md](14_codebase_v27.md) | Kiến trúc module [v27/](../v27/), [visualization/](../v27/visualization/) & lệnh chạy |
| 15 | [15_decisions_and_limitations.md](15_decisions_and_limitations.md) | Quyết định đã chốt & giới hạn thực nghiệm |
| 16 | [16_cost_resilient.md](16_cost_resilient.md) | Giảm Turnover & Tối ưu Kháng Chi phí ([v27/cost_resilient/](../v27/cost_resilient/)) |

## Ký hiệu chuẩn
- $t$: ngày ra quyết định (chỉ dùng thông tin đến hết đóng cửa $t - 1$).
- $U_t$: tập cổ phiếu hợp lệ tại ngày $t$ ($|U_t| \le K = 1000$).
- $L = 60$: cửa sổ lookback $[t - L, t)$; $F = 34$: số đặc trưng cổ phiếu; $m = 28$: số chỉ báo vĩ mô.
- $h \in \{1, 5\}$: chân trời dự báo (`1d` với `purge=2`, `5d` với `purge=6`).
