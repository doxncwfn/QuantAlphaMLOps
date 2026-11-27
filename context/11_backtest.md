# 11 · Mô phỏng Backtest & Kế toán Giao dịch

**Code triển khai:** [simulator.py](../v27/backtest/simulator.py), [consolidation.py](../v27/portfolio/consolidation.py), [constraints.py](../v27/risk/constraints.py)

## 1. Hợp nhất vị thế & Triệt tiêu churn nội bộ (`consolidate_positions`)
Thay vì giao dịch từng chiến lược con riêng lẻ, các phân bổ $a_{k,t}$ được gộp ở mức cổ phiếu trước khi tính lệnh:

$$W_{p, i, t} = \sum_{k} a_{k, t}\, W_{k, i, t}$$

## 2. Kế toán Mark-to-Market hàng ngày (`backtest_weights_daily`)
- **Lợi suất gộp:** $g_t = \sum_i W_{i,t}\, r_{1d, i, t}$.
- **Trọng số trôi (drifted weight) từ phiên $t - 1$:**
  $$W^{\text{drift}}_{i,t} = \dfrac{W_{i,t-1}\,(1 + r_{1d, i, t-1})}{\max(1 + g_{t-1},\ 10^{-6})}$$
- **Khối lượng giao dịch 2 chiều:** $\text{traded}_t = \sum_i |W_{i,t} - W^{\text{drift}}_{i,t}|$.

## 3. Ba tầng chi phí thực tế
1. **Linear TC (`net`):** $\text{tc}_t = \text{traded}_t \times c_{\text{bps}} \times 10^{-4}$ (chuẩn $c = 2\text{ bps}$, lưới $0..15\text{ bps}$).
2. **Short Borrow & Financing:** $\text{borrow}_t = L^{\text{short}}_t \cdot \dfrac{0.005}{252}$, $\text{fin}_t = \max(L^{\text{long}}_t - 1, 0)\cdot(r_{f,t} + \dfrac{0.005}{252})$.
3. **Square-Root Market Impact (`net_allin`):** với $\eta = 0.10$, quy mô sổ $B = \$20\text{M}$:
   $$\text{impact}_t = \sum_i |\Delta W_{i,t}|\,\eta\,\sigma_{i,t}\,\sqrt{\min\!\left(\dfrac{|\Delta W_{i,t}|\,B}{\text{ADV}_{20,i,t}},\ 1\right)}$$

```python
# v27/backtest/simulator.py
gross = np.einsum("ti,ti->t", W_c, R)
drifted[1:] = (W_c[:-1] * (1.0 + R[:-1])) / np.maximum(1.0 + gross[:-1, None], 1e-6)
delta_w = np.abs(W_c - drifted)
traded = delta_w.sum(axis=1)
net_allin = gross - tc_linear - borrow - fin - impact
```
