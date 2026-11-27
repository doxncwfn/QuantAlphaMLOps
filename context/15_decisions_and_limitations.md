# 15 · Tổng kết Quyết định & Giới hạn

Tất cả câu hỏi mở trước đây (filter, target `raw`/`risk`, `soft_rank`, kiến trúc `HybridLstmTfm`, 34 features, tầng danh mục) **đã chốt và kiểm chứng** trong Semester 261 (`v27/`).

**Giới hạn kỹ thuật ghi nhận:**
1. **Ngưỡng chi phí giao dịch:** Tín hiệu `1d` có turnover cao; trên `6.5–9.5 bps/chiều` phần lớn chiến lược suy giảm mạnh, yêu cầu khớp lệnh thực tế `<= 2–4 bps/chiều`.
2. **Chế độ đảo chiều đa nhân tố (`fold_07` 2017 & `fold_08` 2018):** Hầu hết ensemble đi ngang hoặc âm nhẹ khi dispersion nhân tố đảo chiều.
3. **Dữ liệu mở rộng:** Kho hiện dừng tại `2024-12-31`; nếu nối `2025+`, dùng return-based stitching với CRSP.
