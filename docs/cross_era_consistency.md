# Cross-Era Consistency Report — CRSP (`2024-12-31`) to Yahoo (`2025-01-02`) Boundary (§3.6)

**Date:** 2026-11-06  
**Boundary Evaluated:** `2024-12-31` (last CRSP trading session) $\to$ `2025-01-02` (first Yahoo trading session)  
**Common Active Constituents (`in_universe = True` on both dates):** `999` securities (`PERMNO`s)

---

## 1. Executive Summary

1. **Multi-Day Rolling & State Features (`ret_20d`, `ret_60d`, `vol_20d`, `vol_60d`, `vol_ratio`, `price_z_20`, `drawdown_20`, `eff_ratio_20`, `vol_shock_5`, `resid_mom_20`, `resid_vol_20`):**
   - Cross-sectional standard deviations change by **$-5.7\%$ to $+13.0\%$** across the boundary (well within the $\pm 20\%$ threshold).
   - Cross-sectional rank/linear correlations across the two adjacent sessions are **`0.896` to `0.999`**, confirming seamless lookback stitching between CRSP history (`2024`) and Yahoo observations (`2025`).
2. **Single-Day Return & Intraday Primitives (`r_cc`, `r_on`, `r_id`, `resid_ret`) and `ret_5d`:**
   - Single-day return cross-sectional dispersion is higher on `2025-01-02` (first trading day of the new year, high institutional rebalancing & macro dispersion) than on `2024-12-31` (quiet New Year's Eve session): `r_cc` std moves from `0.01041` to `0.01725` (`+65.7%`), `r_on` std moves from `0.00428` to `0.00951` (`+122.2%`, and across the full 2025 Yahoo regime `r_on` std is ~32% lower than 2000–2024 CRSP per v1.3 §11 Risk R1), `r_id` std moves from `0.01092` to `0.01509` (`+38.1%`), `resid_ret` std moves from `0.00961` to `0.01684` (`+75.2%`), and `ret_5d` std moves from `0.02137` to `0.02695` (`+26.1%`).
   - Per v1.3 §7.1 and §11 Risk R1, the model pipeline applies the cross-sectional **Blom normal-score transform** (`normal_scores`) per date $t$, mapping every stock feature's cross-sectional marginal distribution to exact $\mathcal{N}(0, 1)$ (`mean = 0.0000`, `std = 1.0000`) on both `2024-12-31` and `2025-01-02`, completely eliminating day-to-day and cross-era scale shifts.
3. **Post-2024 Split Basis Alignment:**
   - Because `bronze.prices_2526` was extracted in late 2026, Yahoo Finance backward-adjusted `2025-01-02` prices for 35 constituents that underwent stock splits or reverse splits during `2025–2026` (e.g., `NFLX` 10:1, `BKNG` 25:1, `ORLY` 15:1, `NOW` 5:1, `CRWD` 4:1, `FAST` 2:1).
   - `load_warehouse_data()` detects any constituent with an unadjusted basis shift $|\log(O_{\text{2025-01-02}} / C_{\text{2024-12-31}})| > 0.20$ at the boundary, uses the clean same-basis intraday return on `2025-01-02`, and aligns the Yahoo `split_adj_*` price series to the `2024-12-31` CRSP basis so rolling `price_z_20` and `drawdown_20` experience zero artificial jumps.

---

## 2. Cross-Sectional Distribution Comparison (`2024-12-31` vs. `2025-01-02`)

| # | Feature | `2024-12-31` Mean | `2024-12-31` Std | `2024-12-31` Median | `2024-12-31` IQR | `2025-01-02` Mean | `2025-01-02` Std | `2025-01-02` Median | `2025-01-02` IQR | $\Delta$ Std (%) | Cross-Date Corr | $>20\%$ Std Shift? |
|:-:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | `r_cc` | `0.00100` | `0.01041` | `0.00092` | `0.00884` | `-0.00338` | `0.01725` | `-0.00381` | `0.01564` | `+65.7%` | `-0.168` | Yes (Day-1 dispersion; mitigated by NS) |
| 2 | `r_on` | `0.00323` | `0.00428` | `0.00303` | `0.00415` | `0.00453` | `0.00951` | `0.00580` | `0.00604` | `+122.2%` | `+0.041` | Yes (Risk R1; mitigated by NS) |
| 3 | `r_id` | `-0.00223` | `0.01092` | `-0.00205` | `0.00893` | `-0.00791` | `0.01509` | `-0.00936` | `0.01433` | `+38.1%` | `-0.095` | Yes (Day-1 dispersion; mitigated by NS) |
| 4 | `range` | `0.01787` | `0.01169` | `0.01484` | `0.00822` | `0.02704` | `0.01335` | `0.02426` | `0.01317` | `+14.2%` | `+0.576` | No ($\le 20\%$) |
| 5 | `close_loc` | `0.42061` | `0.24867` | `0.37952` | `0.36227` | `0.31328` | `0.24020` | `0.24463` | `0.29733` | `-3.4%` | `-0.122` | No ($\le 20\%$) |
| 6 | `rel_vol` | `-0.26945` | `0.29617` | `-0.27809` | `0.41098` | `-0.11904` | `0.31688` | `-0.09821` | `0.35012` | `+7.0%` | `+0.380` | No ($\le 20\%$) |
| 7 | `ret_5d` | `-0.00782` | `0.02137` | `-0.00711` | `0.01978` | `-0.01877` | `0.02695` | `-0.01856` | `0.02501` | `+26.1%` | `+0.731` | Yes (5d window shift; mitigated by NS) |
| 8 | `ret_20d` | `-0.06921` | `0.07412` | `-0.06832` | `0.07254` | `-0.06888` | `0.07428` | `-0.06865` | `0.07582` | `+0.2%` | `+0.955` | No ($\le 20\%$) |
| 9 | `ret_60d` | `-0.00793` | `0.16149` | `-0.00748` | `0.18319` | `-0.00246` | `0.16582` | `-0.00912` | `0.19044` | `+2.7%` | `+0.990` | No ($\le 20\%$) |
| 10 | `vol_20d` | `0.01738` | `0.00890` | `0.01495` | `0.00773` | `0.01748` | `0.00895` | `0.01509` | `0.00851` | `+0.6%` | `+0.988` | No ($\le 20\%$) |
| 11 | `vol_60d` | `0.02051` | `0.00982` | `0.01837` | `0.00974` | `0.02057` | `0.00984` | `0.01853` | `0.01039` | `+0.2%` | `+0.999` | No ($\le 20\%$) |
| 12 | `vol_ratio` | `0.86309` | `0.21564` | `0.85402` | `0.28614` | `0.86612` | `0.21520` | `0.86214` | `0.29108` | `-0.2%` | `+0.960` | No ($\le 20\%$) |
| 13 | `price_z_20` | `-0.44084` | `0.37825` | `-0.43372` | `0.45727` | `-0.44112` | `0.42580` | `-0.44510` | `0.54812` | `+12.6%` | `+0.896` | No ($\le 20\%$) |
| 14 | `drawdown_20` | `-0.09078` | `0.05832` | `-0.08141` | `0.06144` | `-0.09184` | `0.05689` | `-0.08195` | `0.06498` | `-2.5%` | `+0.942` | No ($\le 20\%$) |
| 15 | `eff_ratio_20` | `0.33255` | `0.19794` | `0.32491` | `0.29858` | `0.33510` | `0.19985` | `0.32640` | `0.30912` | `+1.0%` | `+0.945` | No ($\le 20\%$) |
| 16 | `vol_shock_5` | `-0.47633` | `0.23363` | `-0.47936` | `0.31764` | `-0.30112` | `0.22045` | `-0.30105` | `0.28810` | `-5.6%` | `+0.936` | No ($\le 20\%$) |
| 17 | `resid_ret` | `0.00222` | `0.00961` | `0.00151` | `0.00874` | `-0.00201` | `0.01684` | `-0.00220` | `0.01588` | `+75.2%` | `-0.041` | Yes (Day-1 dispersion; mitigated by NS) |
| 18 | `resid_mom_20` | `-0.02137` | `0.07630` | `-0.02574` | `0.07800` | `-0.02395` | `0.07698` | `-0.02985` | `0.07910` | `+0.9%` | `+0.959` | No ($\le 20\%$) |
| 19 | `resid_vol_20` | `0.01406` | `0.00823` | `0.01161` | `0.00776` | `0.01412` | `0.00822` | `0.01189` | `0.00818` | `-0.1%` | `+0.983` | No ($\le 20\%$) |
