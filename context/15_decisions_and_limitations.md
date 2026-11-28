# 15 · Tổng kết Quyết định & Giới hạn

Tất cả câu hỏi mở trước đây (filter, target `raw`/`risk`, `soft_rank`, kiến trúc `HybridLstmTfm`, 34 features, tầng danh mục) **đã chốt và kiểm chứng** trong Semester 261 ([v27/](../v27/)).

**Giới hạn kỹ thuật ghi nhận:**
1. **Ngưỡng chi phí giao dịch (đã giải quyết bởi [v27/cost_resilient/](../v27/cost_resilient/)):** Phiên bản gốc `v27` (`G_HierBest_G5`) có turnover `31.94%/ngày`, suy giảm mạnh trên `6.7 bps/chiều` (`Net5 = 0.559`). Module [v27/cost_resilient/](../v27/cost_resilient/) (Woodbury-ADMM L1-QP + Smooth Method G NTB + Net-5bps Hierarchical ERC) hạ turnover xuống `5.17–12.76%/ngày`, nâng `Net5` lên `1.249–1.342` và ngưỡng hòa vốn lên `13.2–19.7 bps/chiều`.
2. **Chế độ đảo chiều đa nhân tố (`fold_07` 2017 & `fold_08` 2018):** Các ensemble gốc đi ngang hoặc âm nhẹ khi dispersion nhân tố đảo chiều; trung hòa volatility-factor trong [v27/cost_resilient/](../v27/cost_resilient/) (`P3A_ADMM_Kal3_L1_4bp`, `P3A_ADMM_All3Sm_L1_3bp`) giúp đạt `14/14` fold dương ở `2 bps`.
3. **Dữ liệu mở rộng:** Kho hiện dừng tại `2024-12-31`; nếu nối `2025+`, dùng return-based stitching với CRSP.
