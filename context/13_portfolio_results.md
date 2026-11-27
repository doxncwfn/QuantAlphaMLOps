# 13 · Danh mục, Tối ưu hóa & Risk Overlays (`v27/`)

1. **8 quy tắc cơ sở × 3 mô hình = 24 ứng viên (`v27/portfolio/`):**
   - `rank_RAW`, `top10_RAW`, `top10_InvVol`, `top10_Buf10_20`, `top10_InvVol_Buf`, `top10_BetaNeu_InvVol_Buf`, `top10_HOLD5`, `top10_InvVol_HOLD5`.
   - `consolidate_positions` gộp vị thế ròng theo mã trước khi giao dịch để triệt tiêu churn nội bộ giữa các sleeve.
2. **Các phương pháp kết hợp (`v27/experiments/`, `v27/optimization/`, `v27/risk/`):**
   - **Method A:** Static Equal-Weight (`A1_EW_All24`), theo family, theo horizon.
   - **Method B:** Trộn điểm dự báo trước khi xây danh mục (`B5_ScoreEW_Rank_All3`).
   - **Method D:** Tối ưu nhân quả trên `Ortho-8` với Ledoit-Wolf & Bayes-Stein (`D5_ExpMaxSharpe_BS_Ortho`).
   - **Method E:** Phân bổ động rolling `63d/126d` có EMA + L1 hysteresis (`E3_RollMinVar_126d_Hyst`).
   - **Method F:** Phân cấp 2 tầng `1d` vs `5d` ERC (`F4_Hier_Dual1d5d_ERC`).
   - **Method G:** Risk Overlays (`VolTgt 9%`, `Gross <= 1.80`, DD brake, `CVaR95` cap, `|w_i| <= 2.5%`, `|beta_60| <= 0.08`, `|sector| <= 15%`, `<= 5% ADV`).

## Kết quả OOS chuẩn (`2011-01-03..2024-12-31`, `$20M` book, `2 bps`)
- **Flagship (`G_HierBest_G5_FullConstrainedOverlay`):** `Net2 Sharpe = 1.531`, `All-In Sharpe = 1.329`, `CAGR = 7.76%`, `AnnVol = 4.97%`, `Comp MaxDD = 8.24%`, `FF6 Alpha = 5.55%/năm (t=4.29)`.
- **Return-Oriented (`G_ScoreBest_G1_VolTgt09`):** `Net2 Sharpe = 1.512`, `CAGR = 13.93%`, `Comp MaxDD = 15.46%`.
- **Best Score Blend (`B5_ScoreEW_Rank_All3__top10_BetaNeu_InvVol_Buf`):** `Net2 Sharpe = 1.430`, `CAGR = 15.65%`, `PosYears = 100.0%`.
