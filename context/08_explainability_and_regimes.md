# 08 · Giải thích Mô hình, Bổ trợ & Phân tích Chế độ Thị trường

**Code triển khai:** [gate_attribution.py](../v27/explainability/gate_attribution.py), [complementarity.py](../v27/explainability/complementarity.py), [risk_contribution.py](../v27/explainability/risk_contribution.py), [regime_analysis.py](../v27/explainability/regime_analysis.py), [factor_exposure.py](../v27/explainability/factor_exposure.py)

---

## 1. MASTER Gate Feature-Cluster Attribution ([gate_attribution.py](../v27/explainability/gate_attribution.py))
- Trọng số cổng $\alpha_{t,f}$ từ `MarketFeatureGate` được gom trung bình theo 5 cụm kinh tế (`Price/ShortMom`, `Volatility`, `Price-Volume`, `SlowMom/Quality`, `Risk/Tail/Micro`) để theo dõi cách mô hình tự động dịch chuyển trọng tâm đặc trưng qua từng giai đoạn thị trường.

## 2. Phân tích Bổ trợ 24 Chiến lược ([complementarity.py](../v27/explainability/complementarity.py))
- **Tương quan điều kiện:** Tính Pearson, Spearman, **Downside Correlation** (trên các ngày $r_i < 0$ hoặc $r_j < 0$), **High-Vol Correlation** (top 20% biến động thị trường) và **Large-DD Correlation** (top 15% drawdown sâu nhất của `EW-24`).
- **Trùng lặp vị thế:** Weight Cosine Similarity và **Jaccard Overlap** cho Long, Short, và toàn danh mục.
- **Lower-Tail Dependence ($q = 5\%$):**
  $$\lambda_L(A, B) = \dfrac{1}{0.05\,T}\sum_{t=1}^{T} \mathbb{I}\!\left(r_{A,t} \le q_{0.05}(r_A)\ \land\ r_{B,t} \le q_{0.05}(r_B)\right)$$

## 3. Đóng góp Rủi ro ([risk_contribution.py](../v27/explainability/risk_contribution.py))
Với danh mục `EW-24` ($w_k = 1/24$, ma trận hiệp phương sai năm hóa $\Sigma_{\text{ann}}$):

$$\text{MCR}_k = \dfrac{(\Sigma_{\text{ann}} w)_k}{\sqrt{w^\top \Sigma_{\text{ann}} w}}, \qquad \text{RiskShare}_k(\%) = \dfrac{w_k \cdot \text{MCR}_k}{\sqrt{w^\top \Sigma_{\text{ann}} w}} \times 100$$

## 4. Phân loại Chế độ Thị trường Nhân quả ([regime_analysis.py](../v27/explainability/regime_analysis.py))
Tất cả cờ chế độ đều dịch `shift(1)` (không look-ahead):
- **Bull / Bear:** Bear khi $\text{NAV}_{\text{mkt}, t-1} < \text{SMA}_{200}(\text{NAV}_{\text{mkt}})_{t-1}$ và lợi suất 63 ngày $< 0$.
- **Vol / Breadth / Momentum:** chia theo phân vị 33%/67% của `vol20` và tỷ lệ mã tăng (`breadth`), hoặc dấu tổng nhân tố `MOM` 63 ngày.
- **5 giai đoạn sự kiện:** `Euro_2011`, `ChinaOil_15_16`, `COVID_2020`, `RateHike_2022`, `AI_Bull_23_24`.
