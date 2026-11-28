# 10 · Framework Kết hợp & Tối ưu Danh mục (Methods A–G)

**Code triển khai:** [consolidation.py](../v27/portfolio/consolidation.py), [estimators.py](../v27/optimization/estimators.py), [solvers.py](../v27/optimization/solvers.py), [method_d_opt.py](../v27/optimization/method_d_opt.py), [method_a_static.py](../v27/experiments/method_a_static.py), [method_b_score_blend.py](../v27/experiments/method_b_score_blend.py), [method_e_dynamic.py](../v27/experiments/method_e_dynamic.py), [method_f_hierarchical.py](../v27/experiments/method_f_hierarchical.py), [method_g_overlay.py](../v27/risk/method_g_overlay.py)

---

## 1. Nguyên tắc Nền tảng: Gộp Vị thế & Chọn Lõi Trực giao (`Ortho-8`)
1. **Consolidated Position Netting (`consolidate_positions`):**
   $$W_{p,i,t} = \sum_{k} a_{k,t}\, W_{k,i,t}$$
   Lệnh mua/bán ngược chiều giữa các sleeve tự triệt tiêu trước khi ra thị trường, giảm mạnh chi phí giao dịch thực tế.
2. **Lõi Trực giao Nhân quả (`Ortho-8`, `select_orthogonal_subset`):** Dùng duy nhất năm warmup `2010` (`252` ngày) lọc tham lam theo Sharpe giảm dần với ngưỡng tương quan $|\rho| < 0.88$, giữ tối thiểu 1 đại diện mỗi mô hình $\rightarrow$ chọn ra 8 sleeve ít trùng lặp nhất từ 24 ứng viên.

---

## 2. Các Phương pháp Kết hợp (A, B, D, E, F)

### Method A — Phân bổ Tĩnh ([method_a_static.py](../v27/experiments/method_a_static.py))
- `A1_EW_All24` ($a_k = 1/24$), `A2_EW_1d16`, `A3_EW_Exp_*`, `A4_EW_Fam_*`, các cặp bù trừ `A5..A8`, và `A9_Corr5dRebal_Pair_t5_t1` (tái cân bằng 5 ngày không gối đầu).

### Method B — Trộn Điểm Dự báo Trước Khi Xây Danh mục ([method_b_score_blend.py](../v27/experiments/method_b_score_blend.py))
- Trộn z-score hoặc rank chuẩn hóa của `raw_t1`, `risk_t1`, `risk_t5` rồi mới đưa qua 8 quy tắc danh mục:
  $$s^{\text{B5}}_{i,t} = \dfrac{1}{3}\Big(\text{pct\_rank}(s^{\text{raw\_t1}}_{i,t}) + \text{pct\_rank}(s^{\text{risk\_t1}}_{i,t}) + \text{pct\_rank}(s^{\text{risk\_t5}}_{i,t}) - 1.5\Big)$$
- **`B5_ScoreEW_Rank_All3__top10_BetaNeu_InvVol_Buf`**: `CAGR = 15.65%`, `Net2 Sharpe = 1.430`, `Comp MaxDD = 16.69%`, `100%` năm dương.

### Method D — Tối ưu Nhân quả Mở rộng ([estimators.py](../v27/optimization/estimators.py), [solvers.py](../v27/optimization/solvers.py), [method_d_opt.py](../v27/optimization/method_d_opt.py))
Ước lượng trên cửa sổ mở rộng với `purge = 6` ngày, tái cân bằng mỗi `21` ngày:
- **Ước lượng hiệp phương sai & kỳ vọng:** Ledoit-Wolf shrinkage $\hat{\Sigma}_{\text{LW}} = (1-\delta)S + \delta \mu I$ và Jorion Bayes-Stein $\hat{\mu}_{\text{BS}} = (1-\lambda)\hat{\mu} + \lambda \mu_0 \mathbf{1}$.
- **Bài toán giải ([solvers.py](../v27/optimization/solvers.py)):**
  - `InvVol`, `MinVar`: $\min_w w^\top \Sigma w + \gamma \|w\|_2^2$.
  - `ERC` (Equal Risk Contribution): $\min_w \sum_k \left(\dfrac{w_k (\Sigma w)_k}{w^\top \Sigma w} - \dfrac{1}{N}\right)^2$.
  - `MaxSharpe` (`D5_ExpMaxSharpe_BS_Ortho`): `Net2 Sharpe = 1.355`, `CAGR = 14.96%`, `100%` năm dương.
  - `Mean-CVaR95 LP`: quy hoạch tuyến tính Rockafellar–Uryasev qua `linprog(method="highs")`.

### Method E — Phân bổ Động với EMA & L1 Hysteresis ([method_e_dynamic.py](../v27/experiments/method_e_dynamic.py))
- Cập nhật trên cửa sổ trượt (`63d` hoặc `126d`), làm mượt $w_{\text{blend}} = \alpha w_{\text{target}} + (1-\alpha)w_{\text{curr}}$, chỉ giao dịch khi $\|w_{\text{blend}} - w_{\text{curr}}\|_1 \ge \tau_{\text{hyst}}$ (`1.5%`).
- `E3_RollMinVar_126d_Hyst`: `Net2 Sharpe = 1.299`, `Comp MaxDD = 18.91%`, turnover phân bổ chỉ `29.0%/năm`.

### Method F — Phân cấp 2 Tầng ([method_f_hierarchical.py](../v27/experiments/method_f_hierarchical.py))
- **`F4_Hier_Dual1d5d_ERC`:** Tầng 2 chạy `ERC` nội bộ cho 2 nhóm (`Horiz_1d_FastCore` 16 sleeves và `Horiz_5d_SlowDiv` 8 sleeves `risk_t5`), tầng 1 chạy `ERC` cân bằng rủi ro giữa hai chân trời (`Net2 Sharpe = 1.074` vs `1.002` của `EW-24`).

---

## 3. Method G — Risk Overlays & Ràng buộc Tổ chức ([method_g_overlay.py](../v27/risk/method_g_overlay.py), [constraints.py](../v27/risk/constraints.py))

Áp dụng tuần tự nhân quả (chỉ dùng dữ liệu đến $t-1$) lên danh mục đã gộp:
1. **Vol Targeting (`9%`, `G1`):** $s^{\text{vol}}_t = \text{clip}\!\left(\dfrac{0.09}{\max(\hat{\sigma}_{20/63,t-1},\ 0.025)},\ 0.35,\ 1.35\right)$.
2. **Gross Cap (`G2`):** giới hạn $\sum_i |W_{i,t}| \le 1.80$.
3. **Drawdown Brake (`G3`):** $s^{\text{dd}}_t = \text{clip}\!\left(1 - \dfrac{\max(\text{DD}_{t-1} - 0.035,\ 0)}{0.085},\ 0.45,\ 1.0\right)$.
4. **CVaR95 Cap (`G4`):** giữ Expected Shortfall 63 ngày $\le 1.25\%$/ngày.
5. **Ràng buộc danh mục đầy đủ (`G5`):** `|w_i| <= 2.5%`, `|beta_60| <= 0.08` (chiếu trực giao), `|sector_net| <= 15%`, và `|dw_i| <= 5% ADV / $20M`.

- **Flagship (`G_HierBest_G5_FullConstrainedOverlay`):** `Net2 Sharpe = 1.531` (vượt `EW-24` `+0.529`, Ledoit-Wolf $z = 2.69, p = 0.0072$), `All-In Sharpe = 1.329`, `Comp MaxDD = 8.24%`, `FF6 Alpha = 5.55%/năm (t = 4.29)`, `MKT = 0.011`.

### Nâng cấp Kháng Chi phí ([v27/cost_resilient/](../v27/cost_resilient/), chi tiết xem [16_cost_resilient.md](16_cost_resilient.md))
- **Smooth Method G + Post-NTB (`apply_smooth_method_g_overlay`):** Thêm hysteresis deadband `10%` và `EMA(0.30)` lên hệ số `VolTgt/DD/CVaR` + ngưỡng không giao dịch L1 `8 bps` (`P3B_F4_G5_Sm_NTB8bp`: `14.82%/ngày`, `Net5 = 1.041`).
- **Woodbury-ADMM Tail-Alpha + L1 QP (`solve_admm_tail_l1`):** Tối ưu trực tiếp ở mức cổ phiếu với trung hòa dollar/beta/12-sector/vol và phạt L1 tường minh $\lambda_1 \|z - W^{\text{drift}}_t\|_1$.
- **Net-5bps Dual-Paradigm Hierarchical ERC (`P3C_ENS_Cross_Hier_DualParadigm`):** Tối ưu ERC trên lợi suất `Net5`, hợp nhất lõi ADMM và lõi `F4` nâng cấp (`Net2 = 1.705`, `Net5 = 1.249`, `1-Way = 12.76%/ngày`, `MaxDD2 = 4.93%`).

---

## 4. Kiểm định Tiền đề Động & Ablation Study

1. **Dynamic Sharpe Persistence Pre-check ([score_persistence.py](../v27/signal_analysis/score_persistence.py)):** Đo tương quan Spearman giữa Sharpe quá khứ (`21d/63d/126d/252d`) và Sharpe tương lai (`21d/63d`, `purge = 6`) để xác nhận tính bền vững ngắn–trung hạn trước khi dùng Method E (`21d->21d RankIC = +0.0793`, `t = 2.44`).
2. **Section 10 Ablation Study ([ablations.py](../v27/experiments/ablations.py)):** So sánh với `Baseline_ERC_All24` bằng cách lần lượt loại bỏ từng thành phần:
   - Loại thành phần quy tắc: `Drop_BetaNeu`, `Drop_InvVol`, `Drop_Buffer`, `Drop_HOLD5`.
   - Loại mô hình dự báo: `Drop_raw_t1`, `Drop_risk_t1`, `Drop_risk_t5`.
   - Đổi kiến trúc: `Replace_CovOpt_With_EW` và `Final_Flagship_Full` (chứng minh giá trị gia tăng ròng của từng khối).
