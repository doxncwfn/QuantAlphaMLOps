# 02 · Dữ liệu, Cache & EDA ([data/](../v27/data/) & [eda/](../v27/eda/))

**Code triển khai:** [warehouse_loader.py](../v27/data/warehouse_loader.py), [bundle.py](../v27/data/bundle.py), [universe_eda.py](../v27/eda/universe_eda.py), [run_eda.py](../v27/pipelines/run_eda.py)

## 1. Nguồn dữ liệu DuckDB (`russell1000.duckdb`)
- **`silver.v_prices`:** `10,160,300` dòng, `2,927` mã, `6,163` ngày (`2000-07-03` – `2024-12-31`), tỷ lệ null `close = 0.60%`, `ret_total = 0.61%`.
- **`silver.market_factor`:** Fama-French 5 factors + Momentum (`mktrf`, `smb`, `hml`, `rmw`, `cma`, `umd`, `rf`).
- **`silver.security_history`:** ánh xạ mã `siccd` sang 12 nhóm ngành (`sic_to_sector12`).

## 2. Tăng tốc nạp dữ liệu bằng Polars ([bundle.py](../v27/data/bundle.py))
Thay vì đọc parquet bằng pandas và gọi `pd.pivot()` 4 lần trên bảng 3.7M dòng, `load_v7_data_bundle` đọc trực tiếp bằng `polars.read_parquet` và scatter vào ma trận `(T, N)` qua `np.searchsorted`:

```python
# v27/data/bundle.py
preds_map = {
    e: pl.read_parquet(qh_dir / f"predictions_{e}.parquet")
    .drop_nulls(subset=["score", "label"])
    .filter(pl.col("score").is_not_nan() & pl.col("label").is_not_nan())
    .with_columns(pl.col("date").cast(pl.Datetime("ns")))
    for e in EXPERIMENTS_V7
}
```
- Truy vấn DuckDB sử dụng `con.sql(...).pl()` (DuckDB 2.0 Relation -> Polars zero-copy).
- Xuất `weights_*.parquet` và các bảng thống kê CSV bằng `polars.DataFrame.write_parquet` / `.write_csv`.
