# 06 · Kiến trúc mô hình `HybridLstmTfm`

**Code triển khai:** [hybrid_lstm_tfm.py](../v27/models/hybrid_lstm_tfm.py), [master_gate.py](../v27/models/master_gate.py), [temporal_encoder.py](../v27/models/temporal_encoder.py), [cross_sectional_tfm.py](../v27/models/cross_sectional_tfm.py), [folds.py](../v27/training/folds.py), [trainer.py](../v27/training/trainer.py)

## 1. Luồng xử lý theo từng ngày $t$
1. **MASTER Feature Gate (`MarketFeatureGate`):** điều biến 34 đặc trưng theo trạng thái vĩ mô $M_t \in \mathbb{R}^{28}$ ($\tau_g = 5.0$):
   $$\alpha_t = F \cdot \text{softmax}\!\left(\dfrac{W_g M_t + b_g}{\tau_g}\right), \qquad \tilde{X}_t = X_t \odot \alpha_t$$
   Tổng $\sum_{f=1}^{34} \alpha_{t,f} = 34$ (giá trị trung tính $= 1.0$).
2. **Temporal Encoder (`TemporalLstmAttention`):** qua `Linear(34, 64) + GELU`, 2 lớp `LSTM(64, dropout=0.2)`, attention pooling theo hidden state cuối $h_L$:
   $$a_\ell = \text{softmax}_\ell\!\left(\dfrac{h_\ell^\top h_L}{\sqrt{d}}\right), \qquad e_{i,t} = \text{LN}\!\left(W_p\,[h_L;\ \sum_\ell a_\ell h_\ell]\right)$$
3. **Cross-Sectional Transformer (`CrossSectionalTransformerHead`):** 2 lớp `TransformerEncoderLayer(d=64, heads=4, ff=128)` tương tác giữa $K$ mã trong cùng ngày $t$ (có mask mã rỗng), nối residual $[e_{i,t}; z_{i,t}]$ qua MLP head ra điểm $s_{i,t}$, sau đó chuẩn hóa z-score theo mặt cắt ngang.

## 2. Walk-Forward Cross-Validation (14 Folds)
- Chia đều đoạn test `2010-01-01` – `2024-12-31` thành 14 fold.
- Khoảng cách an toàn: `val_days = 400`, `embargo = 5`, `purge = 2` (1d) hoặc `purge = 6` (5d).
- Optimizer: `AdamW(lr=3e-4, weight_decay=1e-4)`, `ReduceLROnPlateau(factor=0.5, patience=2)`.
