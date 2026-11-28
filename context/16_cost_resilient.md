# 16 · Giảm Turnover & Tối ưu hóa Kháng Chi phí ([v27/cost_resilient/](../v27/cost_resilient/))

**Code triển khai:** [signals.py](../v27/cost_resilient/signals.py), [optimizers.py](../v27/cost_resilient/optimizers.py), [pipeline.py](../v27/cost_resilient/pipeline.py), [reporting.py](../v27/cost_resilient/reporting.py), [run_cost_resilient.py](../v27/pipelines/run_cost_resilient.py), [test_cost_resilient.py](../v27/tests/test_cost_resilient.py)

Mục tiêu: hạ **Daily 1-Way Turnover `< 30%/ngày`** trong khi đạt đồng thời **`Net2 Sharpe > 1.50`**, **`Net5 Sharpe > 1.00`**, và **`Comp MaxDD <= 12–15%`** trên cửa sổ OOS chuẩn `2011-01-03..2024-12-31` (`2010` warmup, `purge = 6`).

---

## 1. Quy luật Suy giảm Sharpe theo Chi phí
Với độ biến động năm hóa $\sigma$ và turnover 1 chiều hàng ngày $T_{1w}$ (`Daily1Way(%) = 0.5 * traded * 100`), độ dốc suy giảm Sharpe tuyến tính trên mỗi `1 bps/side` là:
$$\kappa = \dfrac{252 \times 2 \times 10^{-4}}{\sqrt{252}} \cdot \dfrac{T_{1w}}{\sigma} \approx 0.0504\,\dfrac{T_{1w}}{\sigma}$$
Do $T_{1w}/\sigma$ bất biến theo tỷ lệ đòn bẩy, việc giảm gross đơn thuần không cải thiện được `Net5 Sharpe`. Cần nâng tỷ số **alpha trên mỗi đơn vị turnover** qua 3 trụ cột.

---

## 2. Trụ cột 1 — Làm mượt Tín hiệu Nhân quả ([signals.py](../v27/cost_resilient/signals.py))
- **Chuẩn hóa giữ độ lớn đuôi (`xs_zscore`, `xs_rankgauss`):** Cắt ngưỡng `[-3.0, +3.0]` mỗi ngày, giữ cường độ tín hiệu cực trị thay vì làm phẳng thành `xs_rank`.
- **Per-stock Causal EMA (`causal_ema_panel`):** $\alpha = \dfrac{2}{\text{span} + 1}$ (`span = 3` hoặc `5` ngày); tự reset trạng thái khi mã rời khỏi $U_t$.
- **Scalar Adaptive Kalman Filter (`causal_kalman_panel`):** Tách alpha tiềm ẩn $x_{i,t}$ khỏi nhiễu ngày $\epsilon_{i,t}$ (`q_ratio = 0.30`, `R = 1.0`):
  $$x_{i,t} = \phi_t x_{i,t-1} + \eta_{i,t}, \quad z_{i,t} = x_{i,t} + \epsilon_{i,t}$$
  Khi `adaptive_phi=True`, $\phi_t \in [0.50, 0.95]$ được ước lượng nhân quả từ trung bình trượt 63 ngày của tự tương quan ngang bậc 1 đến hết $t - 1$.
- **Bằng chứng thực nghiệm ([tab01_signal_persistence_and_decay.csv](../outputs/QH_v7/v7_cost_resilient/tables/tab01_signal_persistence_and_decay.csv)):** `risk_t5` có tự tương quan rank bậc 1 `0.8897` (so với `0.6023` của `raw_t1` và `0.5963` của `risk_t1`). Kết hợp `z_all3_ema3` (`0.50 z_r5 + 0.35 z_r1_ema3 + 0.15 z_raw_ema3`) và `z_all3_kal` nâng `RankAC_1d` lên `0.8876`/`0.8817` và `Mean_IC_t0_t4` lên `0.0185`.

---

## 3. Trụ cột 2 — Cơ chế Tái cân bằng & Tiết chế Giao dịch ([optimizers.py](../v27/cost_resilient/optimizers.py))
1. **Asymmetric Hysteresis Buffers (`strat_top10_buf`):** Mở rộng dải thoát `10/25`, `10/30`, `15/30`.
2. **Beta-Neutral InvVol + Overlapping Tranches (`build_betaneu_invvol_hold`):** Áp dụng `neutralize_beta` + `strat_top10_invvol` rồi gộp `HOLD3` hoặc `HOLD5`.
3. **L1 No-Trade Band (`apply_l1_no_trade_band`):** Ngưỡng mềm quanh trọng số trôi $W^{\text{drift}}_t$:
   $$\Delta w_{i,t} = \operatorname{sign}(W^{\text{tar}}_{i,t} - W^{\text{drift}}_{i,t})\,\max\!\left(|W^{\text{tar}}_{i,t} - W^{\text{drift}}_{i,t}| - \text{band},\ 0\right)$$
4. **Garleanu–Pedersen Partial Rebalance (`apply_partial_rebalance`):** $w_t = (1 - \gamma) W^{\text{drift}}_t + \gamma W^{\text{tar}}_t$.
5. **Exact L1-Ball Turnover Throttle (`apply_turnover_throttle`):** Chiếu Duchi lên quả cầu $L_1$ bán kính $2 \cdot \text{max\_daily\_1way}$ (`prioritize_largest=True`), chỉ khớp các lệnh có độ tin cậy lớn nhất khi chạm trần turnover ngày.

*Kết quả Trụ cột 1+2 đơn lẻ:* hạ turnover xuống `17–29%/ngày` (`Net5 = 0.90–0.96`), nhưng chưa vượt đồng thời `Net2 > 1.50` và `Net5 > 1.00` nếu thiếu tối ưu hóa chi phí đồng thời với trung hòa nhân tố.

---

## 4. Trụ cột 3 — Woodbury-ADMM L1-Turnover QP & Ensemble Kháng Chi phí

### 4A. Bộ tối ưu Woodbury-ADMM Tail-Alpha + Phạt L1 (`solve_admm_tail_l1`)
Mỗi ngày $t$, giải bài toán lồi chặt trên tập mã hoạt động:
$$\min_{w, z}\; -\alpha_t^\top w + \dfrac{\gamma_t}{2}\sum_i \sigma_{i,t}^2 w_i^2 + \lambda_1 \|z - W^{\text{drift}}_t\|_1 + \dfrac{\lambda_2}{2}\|z - W^{\text{drift}}_t\|_2^2$$
$$\text{s.t.}\quad C_t w = 0,\quad w = z,\quad |z_i| \le w_{\max},\quad |z_i - W^{\text{drift}}_{i,t}| \le \dfrac{0.05\cdot\text{ADV}_{i,t}}{\$20\text{M}},\quad \|z\|_1 \le G_{\max}$$
- **Tail Deadzone & Hiệu chuẩn $\gamma_t$:** Đặt $\text{ts}_i = \operatorname{sign}(z_i)\max(|z_i| - \text{dz}, 0)$ (mặc định `dz = 0.80`), $\alpha_i = \text{IC}\cdot\sigma_i\cdot\text{ts}_i$ (`IC = 0.025`), và $\gamma_t = \max\!\left(\dfrac{\text{IC}\sum_i |\text{ts}_i|/\sigma_i}{G_{\text{target}}},\ 1.0\right)$ để nghiệm không ràng buộc khớp đúng `base_gross = 1.80`, giúp $\lambda_1 = \text{l1\_cost\_bps}/10^4$ mang đơn vị bps thực tế.
- **Trung hòa tuyến tính chính xác ($C_t w = 0$):** Giải bước $w$ bằng khai triển Woodbury–Cholesky ($K \le 15$) cho dollar-neutral, `beta_60`, 12 ngành SIC (`neutralize_sectors=True`), và biến động `sigma_prev` (`neutralize_vol=True`).
- **Bước $z$ (Proximal L1):** Soft-thresholding quanh $W^{\text{drift}}_t$ giữ chính xác các lệnh bằng 0 (no-trade region).

### 4B. Nâng cấp Method G mượt hóa & Chiếu ADMM trên `F4`
- **`apply_smooth_method_g_overlay`:** Thêm hysteresis deadband `10%` và `EMA(0.30)` lên hệ số `VolTgt/DD/CVaR` trước khi áp ràng buộc và `post_ntb = 8 bps`. Riêng bước này hạ turnover của `v27_G_HierBest_G5` từ `31.94%` xuống **`14.82%/ngày`** và tăng `Net5` từ `0.559` lên **`1.041`** (`P3B_F4_G5_Sm_NTB8bp`).
- **`solve_admm_turnover_projection`:** Chiếu danh mục mục tiêu lên $C_t w = 0$ kèm phạt $\|w - W^{\text{drift}}_t\|_1$ (`P3B_F4_ADMM_Proj_L1_15`).

### 4C. Ensemble Đa mô hình Tối ưu trên `Net-5bps` ([pipeline.py](../v27/cost_resilient/pipeline.py))
- Thay vì ước lượng ERC/MinVar trên lợi suất gross hoặc 2 bps, toàn bộ bộ trọng số meta nhân quả (`causal_wf_schedule`, `warmup = 252`, `purge = 6`) được fit trên chuỗi **`bt5["net"]`**.
- **Dual-Paradigm Hierarchical ERC (`P3C_ENS_Cross_Hier_DualParadigm`):** Tầng 2 chạy ERC nội bộ cho 4 sleeve ADMM (`Kal3_L1_4bp`, `All3Sm_L1_3bp`, `SLV_r5_L1_5bp`, `EW3_L1_4bp`) và 2 sleeve `F4` nâng cấp (`F4_G5_Sm_NTB8bp`, `F4_ADMM_Proj_L1_15`); tầng 1 chạy ERC giữa 2 nhóm.

---

## 5. Tổng hợp Kết quả Chính (`23` Chiến lược Đạt Toàn bộ Mục tiêu)

| Chiến lược | Net2 Sh | Net5 Sh | AllIn5 Sh | CAGR2 (%) | MaxDD2 (%) | 1-Way (%/d) | BE (bps) | PosFolds (2/5) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `P3C_ENS_Cross_Hier_DualParadigm` (**Flagship**) | **1.705** | **1.249** | 1.114 | 7.38 | **4.93** | **12.76** | 13.2 | 11/14, 11/14 |
| `P3C_ENS_Cross_ERC_ADMM_F4` | **1.700** | **1.275** | 1.132 | 7.70 | 5.65 | **12.46** | 14.0 | 12/14, 11/14 |
| `P3C_ENS_ReturnOriented_150x` | **1.704** | **1.249** | 1.096 | **11.19** | 7.33 | **19.14** | 13.2 | 11/14, 11/14 |
| `P3C_ENS_UltraLowTurn_Sub10` | **1.622** | **1.267** | 1.137 | 7.05 | **4.48** | **10.01** | 15.7 | 12/14, 11/14 |
| `P3A_ADMM_Kal3_L1_4bp` | **1.568** | **1.254** | 1.113 | 7.78 | 5.95 | **10.12** | 17.0 | **14/14**, 12/14 |
| `P3A_Abl_NoDeadzone_dz0_L1_4bp` | **1.561** | **1.342** | **1.236** | 5.62 | **4.67** | **5.17** | — | — |
| `P3B_F4_G5_Sm_NTB8bp` | **1.501** | **1.041** | 0.912 | 7.43 | 8.16 | **14.82** | 11.8 | 10/14, 9/14 |
| `v27_G_HierBest_G5` *(Baseline cũ)* | 1.531 | 0.559 | 0.356 | 7.76 | 8.24 | 31.94 | 6.7 | 12/14, 9/14 |

- **Kiểm định Ledoit-Wolf (`tab03`):** `P3C_ENS_Cross_Hier_DualParadigm` vượt `v27_G_HierBest_G5` `+0.690` Net5 Sharpe (`z = 4.20, p < 0.0001`), vượt `v27_G_ScoreBest_G1` `+1.052` (`z = 5.57`), và vượt `v27_risk_t5_BetaNeu_Buf` `+0.461` (`z = 2.83, p = 0.0046`).
- **Kết luận Ablation (`tab07`):** Trung hòa ngành thêm `+0.07` Net2 và giảm MaxDD `-3.25%`; trung hòa volatility xóa sạch fold âm tại 2 bps (`14/14` trên `All3Sm_L1_3bp` và `Kal3_L1_4bp`); trung hòa momentum làm giảm `-0.20` Net5 Sharpe (không nên ép về 0).
