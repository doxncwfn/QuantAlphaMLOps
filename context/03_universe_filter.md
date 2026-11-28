# 03 · Bộ lọc Universe

- **Nguồn:** `silver.v_prices` trong `data/warehouse/russell1000.duckdb`.
- **Quy tắc chọn lọc ([warehouse_loader.py](../v27/data/warehouse_loader.py)):**
  - Cập nhật mỗi `TTL = 60` ngày giao dịch.
  - Điều kiện: có giá tại `t - 1` và ít nhất `20` ngày có dữ liệu trong `30` ngày gần nhất.
  - Xếp hạng theo tổng dollar volume `20` ngày (`close * volume`), lấy `K = 1000` mã với **hysteresis buffer `10%`** (giữ mã đang nằm trong top `1100` trước khi lấp đầy từ top ngoài).
  - Thiếu dữ liệu lookback > `max_missing = 2` ngày bị mask (`mask = False`).
