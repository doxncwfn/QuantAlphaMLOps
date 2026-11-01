# 08 · Open questions

Collected from the other files. Remove an item once it is decided, and record the decision in the topic file.

| # | Question | Topic | Owner |
|---|---|---|---|
| 2 | Filter parameter values for experiments (k, buffer_ratio, x, y, ttl, agg); align ttl with Russell reconstitution? | [03_filter](03_filter.md) | Hào |
| 3 | A cached stock stops trading within the TTL: keep its slot until the next refresh, or refill it from the ranking? (Objective handling is decided.) | [03_filter](03_filter.md) | Hào |
| 4 | Which return defines r_i: raw, excess over market, or factor residual | [04_objective](04_objective.md) | Hiếu |
| 6 | Differentiable surrogate for Spearman during training | [04_objective](04_objective.md) | Hiếu |
| 7 | Horizon h (example: 5 days), lookback L, and label offset (default 0 = legacy: enter at the close the features end on; 1 = enter one day later, more conservative) | [02_formulation](02_formulation.md) | team |
| 8 | Shared or separate LSTM for stocks and indicators | [05_model](05_model.md) | Cường |
| 9 | Where stock-level non-time-series features go | [05_model](05_model.md) | Cường |
| 10 | Feature list and economic indicators (m, m') | [06_features](06_features.md) | Cường, Hào |
| 11 | Join 2025–2026 data with CRSP (return-based stitching recommended) | [07_data](07_data.md) | Hào |
| 12 | Long–short decision layer | [01_problem](01_problem.md) | planned |
