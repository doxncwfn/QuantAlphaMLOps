# 07 · Hàm mục tiêu & Đánh giá xếp hạng

**Code triển khai:** [losses.py](../v27/models/losses.py), [metrics.py](../v27/training/metrics.py)

## 1. Differentiable Spearman Loss (`soft_rank`)
Với điểm dự báo $p_t \in \mathbb{R}^{K}$ và mặt nạ hợp lệ $m_t \in \{0,1\}^K$, chuẩn hóa $z_t = (p_t - \mu_{p,t}) / \sigma_{p,t}$ rồi tính hạng mềm khả vi với nhiệt độ $\tau = 0.1$:

$$\tilde{r}_{i,t} = \sum_{j \in U_t} \sigma\!\left(\dfrac{z_{i,t} - z_{j,t}}{\tau}\right) - 0.5$$

Hàm mất mát trên mini-batch các ngày có $|U_t| \ge 20$, trọng số suy giảm theo thời gian $w_t = 0.5^{(t_{\max} - t)/1500}$:

$$\mathcal{L} = \dfrac{\sum_t w_t \cdot \left(1 - \text{Corr}\left(\tilde{r}_t,\ \text{rank}(y_t)\right)\right)}{\sum_t w_t}$$

```python
# v27/models/losses.py
def compute_spearman_loss(scores, y, valid, day_weights, min_stocks=20, tau=0.1):
    wt = day_weights * (valid.sum(1) >= min_stocks).float()
    target_r = torch.where(valid, masked_rank(y, valid), torch.zeros_like(y))
    per_day = 1.0 - masked_corr(soft_rank(scores, valid, tau=tau), target_r, valid)
    return (per_day * wt).sum() / wt.sum()
```

## 2. Tiêu chí Early Stopping (`ValScore`)
$$\text{ValScore} = \overline{\text{RankIC}} + 0.50 \cdot \overline{\text{TailIC}_{5\%}}$$
- `TailIC_5%`: tương quan hạng Spearman chỉ tính trên top 5% và bottom 5% điểm dự báo.
- `Hit@10%`: tỷ lệ cổ phiếu top 10% (bottom 10%) dự báo thực tế nằm đúng top 10% (bottom 10%) lợi suất.
