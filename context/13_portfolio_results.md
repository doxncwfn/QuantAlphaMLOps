# 13 · Danh mục, Tối ưu hóa & Risk Overlays ([v27/](../v27/))

1. **8 quy tắc cơ sở × 3 mô hình = 24 ứng viên ([v27/portfolio/](../v27/portfolio/)):**
   - `rank_RAW`, `top10_RAW`, `top10_InvVol`, `top10_Buf10_20`, `top10_InvVol_Buf`, `top10_BetaNeu_InvVol_Buf`, `top10_HOLD5`, `top10_InvVol_HOLD5`.
   - `consolidate_positions` gộp vị thế ròng theo mã trước khi giao dịch để triệt tiêu churn nội bộ giữa các sleeve.
2. **Các phương pháp kết hợp ([v27/experiments/](../v27/experiments/), [v27/optimization/](../v27/optimization/), [v27/risk/](../v27/risk/)):**
   - **Method A:** Static Equal-Weight (`A1_EW_All24`), theo family, theo horizon.
   - **Method B:** Trộn điểm dự báo trước khi xây danh mục (`B5_ScoreEW_Rank_All3`).
   - **Method D:** Tối ưu nhân quả trên `Ortho-8` với Ledoit-Wolf & Bayes-Stein (`D5_ExpMaxSharpe_BS_Ortho`).
   - **Method E:** Phân bổ động rolling `63d/126d` có EMA + L1 hysteresis (`E3_RollMinVar_126d_Hyst`).
   - **Method F:** Phân cấp 2 tầng `1d` vs `5d` ERC (`F4_Hier_Dual1d5d_ERC`).
   - **Method G:** Risk Overlays (`VolTgt 9%`, `Gross <= 1.80`, DD brake, `CVaR95` cap, `|w_i| <= 2.5%`, `|beta_60| <= 0.08`, `|sector| <= 15%`, `<= 5% ADV`).

## Kết quả OOS chuẩn (`2011-01-03..2024-12-31`, `$20M` book, `2 bps` & `5 bps`)
- **Flagship Semester 261 (`G_HierBest_G5_FullConstrainedOverlay`):** `Net2 Sharpe = 1.531`, `Net5 = 0.559`, `All-In2 = 1.329`, `CAGR2 = 7.76%`, `AnnVol = 4.97%`, `Comp MaxDD = 8.24%`, `1-Way Turn = 31.94%/ngày`, `FF6 Alpha = 5.55%/năm (t=4.29)`.
- **Return-Oriented (`G_ScoreBest_G1_VolTgt09`):** `Net2 Sharpe = 1.512`, `Net5 = 0.197`, `CAGR2 = 13.93%`, `Comp MaxDD = 15.46%`, `1-Way Turn = 77.29%/ngày`.
- **Best Score Blend (`B5_ScoreEW_Rank_All3__top10_BetaNeu_InvVol_Buf`):** `Net2 Sharpe = 1.430`, `Net5 = 0.386`, `CAGR2 = 15.65%`, `PosYears = 100.0%`.

## Nâng cấp Giảm Turnover & Kháng Chi Phí ([v27/cost_resilient/](../v27/cost_resilient/))
- **Mục tiêu:** `1-Way Turn < 30%/ngày`, `Net2 > 1.50`, `Net5 > 1.00`, `Comp MaxDD <= 12–15%` (`23` chiến lược nhân quả đạt đồng thời toàn bộ mục tiêu).
- **Flagship Cost-Resilient (`P3C_ENS_Cross_Hier_DualParadigm`):** Hierarchical ERC tối ưu trên `Net5` kết hợp Woodbury-ADMM L1-QP (trung hòa dollar/beta/12-sector/vol) + `F4+G5Sm+NTB8bp`: `Net2 = 1.705`, `Net5 = 1.249` (`All-In5 = 1.114`, break-even `13.2 bps`, LW `z = 4.20, p < 0.0001` vs `G_HierBest_G5`), `CAGR2 = 7.38%`, `Comp MaxDD2 = 4.93%`, `1-Way Turn = 12.76%/ngày`, `FF6 Alpha2 = 4.86%/năm (t=4.49)`.
- **Return-Oriented Cost-Resilient (`P3C_ENS_ReturnOriented_150x`):** `Net2 = 1.704`, `Net5 = 1.249`, `CAGR2 = 11.19%`, `Comp MaxDD2 = 7.33%`, `1-Way Turn = 19.14%/ngày`.
- **Ultra-Low-Turnover (`P3C_ENS_UltraLowTurn_Sub10` / `P3A_Abl_NoDeadzone_dz0_L1_4bp`):** `10.01%/ngày` (`Net2 = 1.622`, `Net5 = 1.267`, `MaxDD2 = 4.48%`) và `5.17%/ngày` (`Net2 = 1.561`, `Net5 = 1.342`, `All-In5 = 1.236`, `MaxDD2 = 4.67%`).
- **Single-Stage ADMM (`P3A_ADMM_Kal3_L1_4bp`):** `Net2 = 1.568`, `Net5 = 1.254`, `1-Way Turn = 10.12%/ngày`, đạt **`14/14` fold dương ở 2 bps**.
