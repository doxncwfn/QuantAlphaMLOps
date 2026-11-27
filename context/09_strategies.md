# 09 · Hệ thống 25 Chiến lược Cơ sở (24 Sleeves + 1 Benchmark)

**Code triển khai:** [constructors.py](../v27/portfolio/constructors.py), [neutralization.py](../v27/portfolio/neutralization.py), [parity_audit.py](../v27/experiments/parity_audit.py)

Hệ thống cơ sở gồm **3 mô hình dự báo** (`raw_t1`, `risk_t1`, `risk_t5`) × **8 quy tắc xây dựng danh mục** = **24 chiến lược Long–Short** (`gross = 2.0`, `long = +1.0`, `short = -1.0`), đối chiếu cùng **1 Benchmark thị trường** (`Bench_EW_Russell1000`).

---

## 1. Chuẩn hóa & Trung tính hóa Beta

- **Leg Renormalization (`renorm_legs`):** Tách $w^+ = \max(w, 0)$, $w^- = \max(-w, 0)$, chuẩn hóa:
  $$W_{i,t} = \dfrac{w^+_{i,t}}{\sum_j w^+_{j,t}} - \dfrac{w^-_{i,t}}{\sum_j w^-_{j,t}} \implies \sum_i W_{i,t} = 0,\quad \sum_i |W_{i,t}| = 2.0$$
- **Beta Neutralization (`neutralize_beta`):** Hồi quy ngang loại bỏ tương quan tuyến tính của điểm dự báo $s_t$ với beta trễ $\beta_{60, t-1}$:
  $$\tilde{s}_{i,t} = s_{i,t} - \dfrac{\text{Cov}_{U_t}(s_t, \beta_{\text{prev},t})}{\text{Var}_{U_t}(\beta_{\text{prev},t})}\left(\beta_{\text{prev},i,t} - \bar{\beta}_{\text{prev},t}\right)$$

---

## 2. 8 Quy tắc Xây dựng Danh mục

| # | Strategy ID | Cơ chế toán học | Đặc tính chính |
|---|---|---|---|
| 1 | `rank_RAW` | Trọng số tỷ lệ với $(\text{rank}(s_{i,t}) - \bar{r}_t)$, chuẩn hóa 2 chân | Phủ toàn bộ mặt cắt ngang, turnover thấp |
| 2 | `top10_RAW` | Long top $10\%$, Short bottom $10\%$, trọng số đều | Tập trung alpha đuôi, biến động cao |
| 3 | `top10_InvVol` | Long/Short top/bot $10\%$ nhân trọng số $1/\sigma_{\text{prev},i,t}$ | Giảm tỷ trọng mã biến động mạnh |
| 4 | `top10_Buf10_20` | Hysteresis buffer: vào khi vượt top/bot $10\%$, giữ đến khi rớt khỏi $20\%$ | Giảm mạnh giao dịch đảo chiều |
| 5 | `top10_InvVol_Buf` | Kết hợp Hysteresis $10/20$ + sizing $1/\sigma_{\text{prev}}$ | Cân bằng ổn định & chi phí thấp |
| 6 | `top10_BetaNeu_InvVol_Buf` | Áp dụng `neutralize_beta` trước `top10_InvVol_Buf` | Phá cụm beta âm (`Hold_Beta60 ~ 0`), Sharpe cao nhất |
| 7 | `top10_HOLD5` | Gộp trung bình 5 tranche gối đầu của `top10_RAW` (trôi theo $r_{1d}$) | Giảm turnover ~5x qua holding overlap |
| 8 | `top10_InvVol_HOLD5` | Gộp 5 tranche gối đầu của `top10_InvVol` | Mượt hóa lợi suất & chống nhiễu 1 ngày |

### Cơ chế Overlapping Tranches (`apply_overlapping_hold`, $H = 5$)
Mỗi ngày $t$, danh mục duy trì $H = 5$ tranche song song $T^{(0)}, \dots, T^{(H-1)}$:
1. **Trôi vị thế các tranche hiện hữu theo lợi suất 1 ngày $r_{1d, t-1}$:**
   $$T^{(k)}_t = \dfrac{T^{(k)}_{t-1} \odot (1 + r_{1d, t-1})}{\max\!\left(1 + \langle T^{(k)}_{t-1},\ r_{1d, t-1}\rangle,\ 0.25\right)}$$
2. **Thay thế tranche cũ nhất** ($k^* = t \bmod H$) bằng trọng số mục tiêu mới $W^{\text{target}}_t$: $T^{(k^*)}_t = W^{\text{target}}_t$.
3. **Tổng hợp & chuẩn hóa 2 chân:** $W^{\text{HOLD5}}_t = \text{renorm\_legs}\!\left(\dfrac{1}{\min(t + 1,\ H)} \sum_{k=0}^{\min(t+1, H)-1} T^{(k)}_t\right)$.

---

## 3. Tổng hợp 24 Chiến lược + Benchmark (OOS `2011-01-03..2024-12-31`, `2 bps`)

- **Benchmark (`Bench_EW_Russell1000`):** Danh mục trọng số đều toàn bộ $U_t$ (`CAGR = 12.52%`, `AnnVol = 20.13%`, `Sharpe = 0.687`, `Comp MaxDD = 41.83%`).
- **24 chiến lược cá nhân (`{raw_t1, risk_t1, risk_t5} × {8 quy tắc}`):**
  - **Nhóm `raw_t1` (8 sleeves):** Tín hiệu 1d mạnh nhất ở đầu kỳ nhưng turnover cao và nhạy cảm với chế độ thị trường; `raw_t1__top10_BetaNeu_InvVol_Buf` đạt `Net2 Sharpe = 0.784`.
  - **Nhóm `risk_t1` (8 sleeves):** Chuẩn hóa mục tiêu theo $\sigma_{\text{prev}}$ giúp ổn định xếp hạng; `risk_t1__top10_BetaNeu_InvVol_Buf` đạt `Net2 Sharpe = 1.276`.
  - **Nhóm `risk_t5` (8 sleeves):** Tín hiệu 5d bổ trợ cấu trúc rõ rệt (tương quan rank score chỉ `0.630`/`0.644` so với `1d`); **`risk_t5__top10_BetaNeu_InvVol_Buf`** là chiến lược đơn lẻ tốt nhất (`CAGR = 13.60%`, `Net2 Sharpe = 1.302`, `All-In Sharpe = 1.121`, `Comp MaxDD = 18.63%`, `1-Way Turn = 8,718.2%`).
- **Tính nhất quán kế toán:** Toàn bộ 24 chiến lược tái tạo từ cache khớp với `grand_summary_v7.csv` trong sai số tối đa $|\Delta\text{NetSh@2bps}| = 0.0036$.
