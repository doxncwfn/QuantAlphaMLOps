# QuantAlphaMLOps

Nền tảng định lượng xếp hạng cổ phiếu theo mặt cắt ngang (Cross-Sectional Stock Ranking) và tối ưu hóa danh mục Long–Short trung tính thị trường (Đồ án Tốt nghiệp HCMUT, phiên bản hoàn thiện Semester 261).

---

## 1. Bắt đầu nhanh (`v27` Modular Stack)

```bash
pip install -e '.[dev]'                 # Python >= 3.11 (Polars, PyTorch, DuckDB, Ruff)
.venv/bin/pytest -q v27/tests           # Chạy 16 unit & parity tests
uvx ruff check v27 && uvx ruff format --check v27

# 1. Thống kê khám phá dữ liệu (EDA từ cache & DuckDB)
.venv/bin/python -m v27.pipelines.run_eda --output-dir outputs/v27 --from-duckdb

# 2. Kiểm định tín hiệu, giải thích mô hình, tối ưu danh mục (Methods A–G), backtest & biểu đồ
.venv/bin/python -m v27.pipelines.run_portfolio_suite --qh-dir outputs/QH_v7 --output-dir outputs/v27
```

Toàn bộ tham số cấu hình của `v27/` được quản lý tập trung tại [config/v27.yaml](config/v27.yaml) (có thể ghi đè qua biến môi trường `V27_CONFIG`).

---

## 2. Kiến trúc hệ thống & Quy trình xử lý

Hệ thống tách thành 2 giai đoạn rõ ràng với nguyên tắc **một module — một trách nhiệm**:

1. **Sinh tín hiệu Alpha (`v27/eda`, `v27/data`, `v27/features`, `v27/targets`, `v27/models`, `v27/training`):**
   - Lọc động Top-1000 cổ phiếu thanh khoản nhất theo dollar volume 20 ngày (`TTL = 60`, hysteresis buffer `10%`) từ `russell1000.duckdb`.
   - Tính 34 đặc trưng cổ phiếu (5 cụm kinh tế), chuẩn hóa ngang `RankGauss`, kết hợp 28 chỉ báo bối cảnh thị trường trễ `shift(1)`.
   - Huấn luyện `HybridLstmTfm` (MASTER Gate + LSTM Attention + Cross-Sectional Transformer) trên 14 walk-forward folds cho 3 mục tiêu `raw_t1`, `risk_t1` (`purge=2`) và `risk_t5` (`purge=6`) bằng differentiable Spearman loss.
2. **Xây dựng, Tối ưu danh mục & Kiểm định (`v27/signal_analysis`, `v27/explainability`, `v27/portfolio`, `v27/optimization`, `v27/risk`, `v27/experiments`, `v27/backtest`, `v27/visualization`):**
   - Kiểm định tín hiệu (`RankIC`, `TailIC`, `Alpha Decay`, `Decile Monotonicity`) và giải thích (`MASTER Gate`, `3F/FF6 HAC`, tương quan & chế độ thị trường).
   - Xây dựng 24 chiến lược cơ sở (8 quy tắc × 3 mô hình), gộp vị thế ròng (`consolidate_positions`) để triệt tiêu churn nội bộ, và đánh giá các phương pháp kết hợp A–G cùng Risk Overlays (`VolTgt 9%`, `Gross <= 1.80`, DD brake, `CVaR95` cap, `|w_i| <= 2.5%`, `|beta_60| <= 0.08`, `|sector| <= 15%`, `<= 5% ADV`).

---

## 3. Kết quả Benchmark OOS (`2011-01-03` – `2024-12-31`, Sổ `$20M`)

| Vai trò | Chiến lược | CAGR (%) | AnnVol (%) | Net2 Sharpe | All-In Sharpe | Comp MaxDD (%) |
|---|---|---:|---:|---:|---:|---:|
| Thị trường chuẩn | `Bench_EW_Russell1000` | 12.52 | 20.13 | 0.687 | 0.687 | 41.83 |
| Trung bình đều 24 | `A1_EW_All24` | 14.86 | 14.94 | 1.002 | 0.865 | 27.01 |
| Trộn điểm dự báo | `B5_ScoreEW_Rank_All3__top10_BetaNeu_InvVol_Buf` | 15.65 | 10.56 | 1.430 | 1.096 | 16.69 |
| Tăng trưởng kiểm soát Vol | `G_ScoreBest_G1_VolTgt09` | 13.93 | 8.89 | 1.512 | 1.067 | 15.46 |
| **Flagship (Khuyến nghị)** | **`G_HierBest_G5_FullConstrainedOverlay`** | **7.76** | **4.97** | **1.531** | **1.329** | **8.24** |

---

## 4. Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `v27/` | Package hoàn chỉnh Semester 261 (`eda`, `data`, `features`, `targets`, `models`, `training`, `signal_analysis`, `explainability`, `portfolio`, `optimization`, `risk`, `experiments`, `backtest`, `visualization`, `pipelines`, `tests`) |
| `config/v27.yaml` | Cấu hình YAML tập trung cho toàn bộ pipeline `v27/` |
| `context/` | Tài liệu kỹ thuật chi tiết (công thức toán học, link code, định nghĩa metrics và backtest) |
| `outputs/QH_v7/` | Kết quả dự báo chuẩn (`raw_t1`, `risk_t1`, `risk_t5`) và báo cáo `v7_ensemble/` |
| `src/qam/`, `tests/` | Nền tảng thử nghiệm trước đó và bộ test kiểm toán kế toán |
| `reports/`, `legacy/` | Báo cáo đồ án và notebook tham chiếu |

---

## 5. Dữ liệu & Lưu ý bản quyền

Dữ liệu giá lịch sử được lưu tại `data/warehouse/russell1000.duckdb` (CRSP qua WRDS), không đưa lên git. Xem hướng dẫn tải tại [data/README.md](data/README.md). Tuyệt đối không commit `data/` hoặc `outputs/`.
