# Model Pipeline Rebuild Audit — Full `silver.russell` Universe (Baseline 1A)

**Date:** 2026-11-06  
**Evaluation Scope:** Pre-rebuild verification of warehouse tables and data assets in `data/warehouse/russell1000.duckdb`.

---

## 1. Prerequisites State

| Item | Prerequisite | Expected State | Verified State in Warehouse | Status |
|:-:|---|---|---|:---:|
| 1 | `silver.features_daily` table | 37 columns, 6,498,250 in-universe rows, 2,782 unique in-universe permnos, ~98.05% `features_ready` | 37 columns (`permno`, `date`, 19 stock features, 11 context features, `target_5d`, `y_star`, `in_universe`, `features_ready`, `era`), 10,745,931 total rows, 6,498,250 in-universe rows, 2,782 unique in-universe permnos, 98.054% `features_ready` | **PRESENT / VERIFIED** |
| 2 | `silver.features_daily_v1.3_strict` | Exists in warehouse for future ablation | Present (`"silver"."features_daily_v1.3_strict"`), 10,745,931 total rows, 5,847,625 in-universe rows, 2,477 unique in-universe permnos | **PRESENT / PRESERVED** |
| 3 | `silver.market_factor` table | FF 6 factors + RF through at least 2026-07-31 | 15,897 rows from 1963-07-01 to 2026-08-31 (`mktrf`, `smb`, `hml`, `rmw`, `cma`, `umd`, `rf`) | **PRESENT / VERIFIED** |
| 4 | Point-in-time universe source | `silver.russell` constituent lists | 27,121 rows across reconstitution years 2000 to 2026 | **PRESENT / VERIFIED** |
| 5 | Target variables | `target_5d` and `y_star` populated | 6,497,163 valid targets in-universe (99.983%), $y^* \in [-0.5, +0.5]$, mean 0.0000 | **PRESENT / VERIFIED** |

---

## 2. Rebuild Decision

All prerequisites are confirmed present, audited, and intact.  
**Decision: PROCEED TO BUILD.**
