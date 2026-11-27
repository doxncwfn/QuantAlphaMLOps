# 05 · Đặc trưng (Features Pipeline)

**Code triển khai:** [stock_features.py](../v27/features/stock_features.py), [normalizer.py](../v27/features/normalizer.py), [market_context.py](../v27/features/market_context.py)

## 1. Công thức 34 Đặc trưng cổ phiếu theo 5 cụm kinh tế

### Cụm 1: `Price/ShortMom` (12 đặc trưng)
- `log_ret`: $r^{\log}_t = \text{clip}(\ln(1 + r_t),\ -0.2231,\ 0.1823)$
- `mom_n` ($n \in \{5, 10, 20, 60\}$): $\dfrac{P^{\text{adj}}_t}{P^{\text{adj}}_{t-n}} - 1$
- `dist_sma_n` ($n \in \{5, 10, 20, 60\}$): $\dfrac{P^{\text{adj}}_t}{\text{SMA}_n(P^{\text{adj}})_t} - 1$
- `dist_high_20` / `dist_low_20`: $\dfrac{P^{\text{adj}}_t}{\max_{20}(P^{\text{adj}})_t} - 1$ và $\dfrac{P^{\text{adj}}_t}{\min_{20}(P^{\text{adj}})_t} - 1$
- `rsi_14`: $100 \cdot \dfrac{\overline{\Delta P^+}_{14}}{\overline{\Delta P^+}_{14} + \overline{\Delta P^-}_{14} + 10^{-12}}$

### Cụm 2: `Volatility` (5 đặc trưng)
- `vol_n` ($n \in \{5, 20, 60\}$): $\text{std}_n(r^{\log})_t$ (sample std)
- `vol_ratio_5_60`: $\dfrac{\text{vol}_{5,t}}{\text{vol}_{60,t} + 10^{-8}}$
- `parkinson_vol_20`: $\sqrt{\dfrac{1}{4\ln 2}\,\overline{\ln^2(H/L)}_{20,t}}$

### Cụm 3: `Price-Volume` (6 đặc trưng)
- `vol_ratio_20`: $\dfrac{V_t}{\overline{V}_{20,t} + 10^{-9}}$
- `log_dollar_volume_20`: $\ln(1 + \overline{C \cdot V}_{20,t})$
- `amihud_20`: $\ln\!\left(1 + \overline{\dfrac{|r^{\log}|}{\text{DV}/10^6}}_{20,t}\right)$
- `corr_pv_20`: $\text{Corr}_{20}(P^{\text{adj}}, \ln(1 + V))_t$
- `clv` (Close Location Value): $\dfrac{(C_t - L_t) - (H_t - C_t)}{H_t - L_t + 10^{-8}}$
- `cmf_20` (Chaikin Money Flow): $\text{clip}\!\left(\dfrac{\overline{\text{clv}\cdot V}_{20,t}}{\overline{V}_{20,t} + 10^{-8}},\ -1,\ 1\right)$

### Cụm 4: `SlowMom/Quality` (3 đặc trưng)
- `mom_120_20`: $\dfrac{P^{\text{adj}}_{t-20}}{P^{\text{adj}}_{t-120}} - 1$
- `mom_252_20`: $\dfrac{P^{\text{adj}}_{t-20}}{P^{\text{adj}}_{t-252}} - 1$
- `sharpe_60`: $\dfrac{\overline{r^{\log}}_{60,t}}{\text{vol}_{60,t} + 10^{-8}}$

### Cụm 5: `Risk/Tail/Micro` (8 đặc trưng)
- `beta_60`: $\text{clip}\!\left(\dfrac{\text{Cov}_{60}(r^{\log}_i, r^{\log}_{\text{mkt}})}{\text{Var}_{60}(r^{\log}_{\text{mkt}}) + 10^{-12}},\ -5,\ 5\right)$
- `idio_vol_20`: $\text{std}_{20}(r^{\log}_i - \beta_{60} r^{\log}_{\text{mkt}})$
- `max_ret_20`: $\max_{20}(r^{\log})_t$
- `skew_20`: $\text{clip}\!\left(\dfrac{m_{3,20}}{\text{Var}_{20}^{3/2} + 10^{-12}},\ -10,\ 10\right)$
- `hl_range`: $\dfrac{H_t - L_t}{C_t + 10^{-8}}$, `close_open`: $\dfrac{C_t}{O_t + 10^{-8}} - 1$, `gap`: $\text{clip}\!\left(\dfrac{O_t}{C_{t-1} + 10^{-8}} - 1,\ -0.5,\ 0.5\right)$
- `overnight_vs_intraday_20`: $\overline{\text{gap}}_{20,t} - \overline{\text{close\_open}}_{20,t}$

## 2. Chuẩn hóa Cross-Sectional RankGauss (`cs_normalize`)
Với mỗi cặp `(ngày t, đặc trưng f)` trên tập thành viên hợp lệ $U_t$ ($n_t = |U_t|$):

$$u_{i,t} = \text{clip}\!\left(\dfrac{r_{\text{lo}, i, t} + r_{\text{hi}, i, t}}{2\,n_t},\ \dfrac{0.5}{n_t},\ 1 - \dfrac{0.5}{n_t}\right), \qquad z_{i,t} = \sqrt{2}\,\text{erf}^{-1}(2 u_{i,t} - 1)$$

Các mã ngoài $U_t$ được gán `0.0` và lưu `float16`.

## 3. 28 Đặc trưng bối cảnh thị trường (`build_market_features`)
7 chỉ báo ngang trên $U_t$ (`mkt_ret`, `disp`, `breadth`, `vol20`, `vratio`, `mom20`, `hl`) × 4 cửa sổ `(1, 5, 20, 60)` dịch `shift(1)`, chuẩn hóa expanding z-score `shift(1)` (cắt `[-5, 5]`).
