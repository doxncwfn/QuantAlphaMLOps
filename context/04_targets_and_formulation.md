# 04 · Công thức hóa & Luồng nhân quả

**Code triển khai:** [target_builder.py](../v27/targets/target_builder.py), [batch_sampler.py](../v27/targets/batch_sampler.py)

## 1. Biến rủi ro trễ `(t - 1)` (Zero Look-Ahead)
Mọi biến chuẩn hóa rủi ro đều dịch `shift(1)` để đảm bảo tại ngày $t$ chỉ dùng thông tin đến $t - 1$:

$$\sigma_{\text{prev}, i, t} = \max\left(\max(\text{vol}_{20, i, t-1},\ \text{vol}_{60, i, t-1}),\ 0.005\right), \quad \beta_{\text{prev}, i, t} = \beta_{60, i, t-1}$$

## 2. Mục tiêu dự báo (`raw_1d`, `risk_1d`, `raw_5d`, `risk_5d`)
Với lợi suất tổng 1 ngày $R_{1d, i, t}$:

$$R_{5d, i, t} = \prod_{k=0}^{4} (1 + R_{1d, i, t+k}) - 1$$

$$Y_{\text{risk}, 1d, i, t} = \dfrac{R_{1d, i, t}}{\sigma_{\text{prev}, i, t}}, \qquad Y_{\text{risk}, 5d, i, t} = \dfrac{R_{5d, i, t}}{\sigma_{\text{prev}, i, t}\sqrt{5}}$$

| Experiment | Training target | Evaluation target | Horizon $h$ | Purge |
|---|---|---|---:|---:|
| `raw_t1` | `raw_1d` | `raw_1d` | 1 | 2 |
| `risk_t1` | `risk_1d` | `raw_1d` | 1 | 2 |
| `risk_t5` | `risk_5d` | `raw_5d` | 5 | 6 |

```python
# v27/targets/target_builder.py
sigma_raw = np.fmax(X[:, :, fi["vol_20"]], X[:, :, fi["vol_60"]])
sigma_prev = np.vstack([nan_row, sigma_raw[:-1]])
sigma_prev = np.where(np.isnan(sigma_prev), np.nan, np.maximum(sigma_prev, vol_floor))
y_risk_1d = np.where(np.isfinite(Y) & np.isfinite(sigma_prev), Y / sigma_prev, np.nan)
y_risk_5d = np.where(np.isfinite(y_raw_5d) & np.isfinite(sigma_prev), y_raw_5d / (sigma_prev * math.sqrt(5.0)), np.nan)
```
