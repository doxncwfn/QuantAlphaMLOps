# Known Pipeline Limitations

This document records verified data and boundary limitations that remain intentionally unchanged after pre-notebook verification.

---

## A4 — CRSP-to-Yahoo Boundary Stitch on `2025-01-02`

- **Finding ID:** `A4`
- **Source:** `src/data/normalize_schema.py` (`load_warehouse_data`, boundary stitch at `2025-01-02`)
- **Observed Behavior:**
  - Across all `1,642` equity rows on `2025-01-02` (`999` with `in_universe = True`), the boundary stitch sets `ret_total = ret_price` (`max|ret_price - ret_total| = 0.0`, `0` nonzero differences), omitting any overnight ex-dividend component on that single transition session.
  - For securities whose `|log(Open_2025-01-02 / Close_2024-12-31)| > 0.20` (~35 stocks with backward-adjusted 2025–2026 splits in Yahoo relative to `2024-12-31` CRSP), `2025-01-02` falls back to intraday Yahoo return so rolling 20-session price-level features remain continuous across split-basis changes.
- **Impact Bound:**
  - Confined strictly to `2025-01-02` (1 trading day out of `6,559` sessions in `silver.features_daily`, impacting only forward windows crossing `2025-01-02` in Fold 4's out-of-sample test partition).
  - Bounded `<= 0.05%/yr` impact on annualized return and zero cross-sectional ranking bias outside ex-dividend stocks on that single session.
- **Status:** Accepted as a documented boundary limitation; do not modify `load_warehouse_data` without re-validating split continuity across `2024-12-31 -> 2025-01-02`.

---

## Yahoo-Era (`2025–2026`) Terminal Delistings & CRSP Uncoded Collapse

- **`stopped_2526` (`109` entries in `bronze.stopped_2526`):** `silver.delisting` covers CRSP through `2024-12-31`. Yahoo-era stopped/acquired tickers in 2025–2026 do not carry CRSP `DLSTCD`/`DLRET`; when `< 5` forward trading sessions remain in Yahoo, `target_5d` is left `NULL` rather than fabricating settlement returns.
- **`PERMNO 81593` (Washington Mutual, `WM`):** Collapses `-90.51%` (`ret_total = -2.354816`) on `2008-09-26` (reflected in `2008-09-19` `target_5d = -3.277006`), then moves to OTC without a CRSP `delist_code` row before re-entering CRSP years later as `WMIH` (`2015-09-28`, `COOP`). Consequently `2008-09-22..2008-09-26` have `target_5d = NULL` across that gap.
- **`PERMNO 64565` (Countrywide Financial, `CFC`):** Delisted on `2008-06-30` via Bank of America acquisition (`delist_code = 231`, `dlret = +0.020861`, `target_5d = +0.020646`), not a bankruptcy delisting (`delist_code >= 500`).

---

## Cross-Sectional Coverage Gap (`store.date_masks` Post-July Reconstitution)

- **Source:** `src/data/loader.py` (`FeatureDataStore` `60`-session contiguous lookback validity gate)
- **Observed Behavior:**
  - While `in_universe` maintains `[948, 1037]` members per date (`median = 992.0`, only `3/6,228` dates `< 950`, `0 < 900`), `store.date_masks` has `< 900` valid stocks on `582 / 6,228` evaluation dates (`9.34%`).
  - Minimum valid cross-section is `692` on `2005-09-02` (`~69%` of universe); `~70%` of under-covered dates occur in July–September immediately after annual Russell 1000 reconstitution because newly added constituents require `60` clean contiguous trailing sessions with non-missing features/prices (`w_count == 60`).
- **Impact Bound & Modeling Consequence:**
  - During July–September windows on under-covered dates, `DecileMonotonicityLoss` (`K = 10`) assigns `~69` stocks per soft-rank decile instead of `~100`, increasing bucket return sampling noise (`~20%` wider decile SE).
  - Fold 1's training partition (`2001-10-29` to `2005-12-15`, `1,042` dates) spans `3` full Jul–Sep post-reconstitution windows (including the `2005-09-02` minimum of `692`), leaving early-fold training slightly under-covered relative to later expanding folds.

---

## Single-Epoch Dry-Run Net Sharpe Is a Smoke Test Only

- **Observed Dry-Run Values:** Full Fold-1 1-epoch MPS smoke tests produced `Test Rank IC = -0.0158, Net Sharpe = +0.190` (`decile_mono`) and `Test Rank IC = +0.0124, Net Sharpe = -0.316` (`combined`).
- **Diagnostic Verification (B.2):** Evaluating the deterministic 5-feature reversal composite directly via `evaluate_predictions` on the exact same Fold-1 test set (`1,245` dates, `rebalance_stride=5`) yielded `IC = +0.020967, Net Sharpe = +0.2348 (c_bps=10) / +0.6351 (c_bps=5), decile mono = +0.9273` (`resid_vol_20` variant) and `IC = +0.016296, Net Sharpe = +0.2827 (c_bps=10) / +0.6107 (c_bps=5), decile mono = +0.8667` (`r_id` variant), proving the backtest sign is correctly oriented. Across all `6,228` dates (`half_turn = 1.62`), high 5-day turnover makes `c_bps=10` cost (`32.5 bps/reb`) exceed raw reversal gross spread (`26.7 bps/reb`, `Gross Sharpe = +0.7996`, `c_bps=10 Net Sharpe = -0.1737`, `c_bps=5 Net Sharpe = +0.3129`).
- **Interpretation Constraint:** The 1-epoch dry-run Net Sharpe reflects an essentially randomly initialized model after ~16 gradient steps and unstable tail assignments; it MUST NOT be used as a baseline for the 24-run cloud sweep. Comparisons must be strictly trained-vs-trained (converged/early-stopped) or against the deterministic 5-feature composite.
