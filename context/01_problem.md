# 01 · Bài toán & Phạm vi

- **Mục tiêu:** Xếp hạng cổ phiếu theo mặt cắt ngang (cross-sectional) để xây dựng danh mục Long-Short trung tính thị trường (`gross = 2.0`, `long = +1.0`, `short = -1.0`), tối ưu Sharpe sau phí và kiểm soát drawdown.
- **Sổ giao dịch chuẩn:** `$20,000,000` (`book_size`), phí tuyến tính cơ sở `2 bps/chiều` (kiểm tra thêm lưới `0..15 bps` và All-In gồm short borrow `50 bps/năm`, financing, market impact $\propto \sigma\sqrt{|trade|/ADV}$).
- **Cửa sổ đánh giá:**
  - Warmup nhân quả: `2010-01-04` – `2010-12-31` (252 ngày giao dịch).
  - Out-of-sample (OOS) chuẩn: `2011-01-03` – `2024-12-31` (3,522 ngày, 14 walk-forward folds, 1,894 mã).
