# 12 · Hệ thống Metric (Signals, Strategies, Portfolio & Backtest)

**Code triển khai:** [metrics.py](../v27/training/metrics.py), [horizon_audit.py](../v27/signal_analysis/horizon_audit.py), [decile_monotonicity.py](../v27/signal_analysis/decile_monotonicity.py), [metrics.py](../v27/backtest/metrics.py), [statistics.py](../v27/backtest/statistics.py), [factor_exposure.py](../v27/explainability/factor_exposure.py)

## 1. Signal Metrics (Chất lượng tín hiệu)
- **Daily Rank IC & ICIR:** $\text{IC}_t = \text{Spearman}(s_t, y_t)$, $\text{ICIR}_{1d} = \dfrac{\overline{\text{IC}}}{\sigma(\text{IC})}\sqrt{252}$ (với chân trời 5d điều chỉnh overlap: $\sqrt{252/5}$).
- **Tail IC (`5%`) & Top-Bottom Hit (`10%`):** đo độ chính xác ở hai đuôi danh mục nơi vị thế tập trung.
- **Decile Portfolio Monotonicity (`compute_decile_monotonicity`):** chia $U_t$ thành 10 decile $D_1 \dots D_{10}$ theo điểm dự báo, tính lợi suất năm $\bar{r}_{D_k}$:
  - $\rho_{\text{mono}} = \text{Spearman}\!\left((1,\dots,10),\ (\bar{r}_{D_1},\dots,\bar{r}_{D_{10}})\right)$
  - `Adj_Step_Rate(%)`: tỷ lệ bước tăng $\bar{r}_{D_{k+1}} > \bar{r}_{D_k}$
  - `D10 - D1 Spread`: lợi suất & Sharpe danh mục long $D_{10}$ / short $D_1$.

## 2. Portfolio & Risk Metrics (OOS `2011-01-03..2024-12-31`)
- **Compounded CAGR & MaxDD:** $\text{NAV}_t = \prod_{s \le t}(1 + r_s)$, $\text{DD}_t = \dfrac{\text{NAV}_t}{\max_{s \le t}\text{NAV}_s} - 1$, theo dõi độ dài underwater (`MaxDD_Dur`) và ngày hồi phục (`MaxRecov`).
- **Net2 Sharpe & All-In Sharpe:** $\text{Sharpe} = \dfrac{\bar{r}}{\sigma_r}\sqrt{252}$ trên lợi suất sau `2 bps` tuyến tính (`net`) và sau toàn bộ phí + borrow + impact (`net_allin`).
- **Lo (2002) HAC Sharpe (`lo_hac_sharpe`):** điều chỉnh tự tương quan bậc $q = 5$:
  $$\text{SR}_{\text{HAC}} = \dfrac{\text{SR}}{\sqrt{1 + 2\sum_{k=1}^{5}\left(1 - \dfrac{k}{6}\right)\hat{\rho}_k}}$$
- **Sortino, Calmar, Omega, VaR/CVaR (95%/99%), Skew, ExKurt, Worst 5d/20d.**
- **Turnover & Nồng độ:** `AnnTurn_1Way(%)` $= 0.5 \times \overline{\text{traded}_t} \times 252 \times 100$, `Daily1Way(%)` $= 0.5 \times \overline{\text{traded}_t} \times 100$, `EffN_Stock` $= \dfrac{(\sum_i |w_i|)^2}{\sum_i w_i^2}$.
- **Metric Kháng Chi phí ([reporting.py](../v27/cost_resilient/reporting.py)):**
  - `Net5_Sharpe` & `AllIn5_Sharpe`: Sharpe trên lợi suất sau `5 bps/chiều` tuyến tính và sau toàn bộ phí + borrow + impact ở mức `5 bps`.
  - `BreakEven_bps`: chi phí tuyến tính hòa vốn ước lượng từ độ dốc suy giảm giữa `0 bps` và `5 bps` ($\text{SR}_0 / \frac{\text{SR}_0 - \text{SR}_5}{5}$).
  - `GoalPass`: cờ đạt đồng thời `Daily1Way(%) < 30.0`, `Net2_Sharpe > 1.50`, `Net5_Sharpe > 1.00`, và `Comp_MaxDD2(%) <= 15.0`.

## 3. Kiểm định Thống kê & Phân rã Nhân tố
- **Ledoit-Wolf (2008) Sharpe Diff Test (`ledoit_wolf_sharpe_test`):** kiểm định $H_0: \text{SR}_A = \text{SR}_B$ bằng HAC sandwich trên 4 moment $(r_A, r_B, r_A^2, r_B^2)$, trả về $(\Delta\text{Sharpe}, z, p\text{-value})$.
- **Circular Block Bootstrap (`circular_block_bootstrap`):** `1,000` mẫu block `21` ngày -> khoảng tin cậy 95% cho Sharpe, CAGR, MaxDD.
- **FF6 Newey-West OLS (`newey_west_ols`):** hồi quy $(r_t - r_{f,t})$ lên `[MKT, SMB, HML, RMW, CMA, UMD]` với sai số chuẩn HAC lag 5.
