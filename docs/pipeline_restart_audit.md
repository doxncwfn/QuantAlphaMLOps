# Phase 0 — Pipeline Restart Audit Report (`v1.3` Feature Table Rebuild)

> **Date:** 2026-11-06  
> **Target Specification:** `specs/features_baseline.md` (`v1.3-2026-10-03`, FROZEN)  
> **Warehouse Location:** `data/warehouse/russell1000.duckdb`  
> **Decision:** **PROCEED TO BUILD**

---

## 0.1 Data Sources Audit

All required data sources are **present and verified** inside the local DuckDB warehouse at `data/warehouse/russell1000.duckdb` (downloaded from the team's restricted MotherDuck share per `data/README.md`). Standalone CSV/Parquet files at `data/WRDS/*.parquet` or `data/ff.csv` are encapsulated inside the `bronze.*` and `silver.*` tables of `data/warehouse/russell1000.duckdb`, as verified by `bronze._manifest`:

| Source | Expected Content | Warehouse Location | Status | Row / Security Count & Span |
|---|---|---|---|---|
| **CRSP raw files** | `PRC`, `OPENPRC`, `ASKHI`, `BIDLO`, `VOL`, `RET`, `RETX`, `CFACPR`, `CFACSHR`, `SHROUT`, `DLSTCD`, `DLRET`, `TICKER`, `PERMNO`, `SHRCD` | `bronze.crsp_daily` (and `silver.prices_daily`, `silver.delisting`) | **PRESENT** | `18,730,439` raw rows (`18,617,868` valid-price rows after `(PERMNO, date)` deduplication), `2,931` securities, spanning `1925-12-31` to `2024-12-31` |
| **Yahoo raw files** | `Open`, `High`, `Low`, `Close`, `Adj_Close`, `Volume`, `Ticker`, `Date` | `bronze.prices_2526` (`full_2025 (2).parquet`, `full_2026 (2).parquet`) | **PRESENT** | `738,577` deduplicated `(Ticker, Date)` rows across `1,831` tickers, spanning `2025-01-02` to `2026-10-02` |
| **Fama-French factors** | `mktrf`, `smb`, `hml`, `rmw`, `cma`, `umd`, `rf` | `silver.market_factor` | **PRESENT** | `15,897` daily rows spanning `1963-07-01` to `2026-08-31` (decimal returns; `v1.3` §10.5 factor cutoff at `2026-07-31` enforced) |
| **Russell 1000 membership** | Point-in-time constituent lists by reconstitution year (`2000`–`2026`) | `silver.russell` and `bronze.russell_lists` | **PRESENT** | `27,121` rows across reconstitution years `2000`–`2026` (`~1,000` constituents per year) |
| **US history backfill** | Full CRSP history for pre-index warmup (`332` sessions prior to `2000-06-30`) | `bronze.crsp_daily` / `silver.prices_daily` | **PRESENT** | Full lifetime CRSP history back to `1925-12-31` is included in `bronze.crsp_daily` (`7ecbef7e...` SHA-256 in `bronze._manifest`) |

---

## 0.2 Warehouse Status

The DuckDB warehouse exists at `data/warehouse/russell1000.duckdb` (`~1.5 GB`). Schema inspection reveals:

### `bronze` schema
- `bronze.crsp_daily` (`18,730,439` rows): Full 61-column WRDS CRSP daily stock extract (`1925-12-31` to `2024-12-31`). Contains both `RET` and `RETX`, `OPENPRC`, `ASKHI`, `BIDLO`, `PRC`, `VOL`, `CFACPR`, `CFACSHR`, `SHROUT`, `DLSTCD`, `DLRET`, `DLRETX`, `SHRCD`, `TICKER`, `TSYMBOL`, and `PERMNO`.
  - *Note on multi-distribution rows:* CRSP outputs multiple rows on the same `(PERMNO, date)` when multiple distribution codes (`DISTCD`) occur on the same day (`5,786` duplicate pairs; `PRC`, `VOL`, `RET`, `RETX`, `CFACPR` are identical across duplicates). Deduplicating on `(PERMNO, date)` and filtering out `PRC IS NULL` placeholder rows yields `18,617,868` unique trading sessions, matching `silver.prices_daily`.
- `bronze.prices_2526` (`1,452,156` rows): Contains two ingestion snapshots (`full_2025 (1).parquet` + `full_2026 (1).parquet` and `full_2025 (2).parquet` + `full_2026 (2).parquet`). Snapshot `(2)` is a strict superset (`738,577` unique `(Ticker, Date)` rows).
- `bronze.russell_lists` (`25,364` rows) & `bronze.master_list` (`2,931` rows).
- `bronze.stopped_2526` (`109` rows): Manual audit of `2025–2026` delistings and ticker transitions.
- `bronze._manifest` (`9` rows): SHA-256 checksums and row counts of all source files.

### `silver` schema
- `silver.prices_daily` (`18,723,339` rows): Cleaned CRSP daily panel (`1925-12-31` to `2024-12-31`). Note that `silver.prices_daily` stores total return `ret` (`RET`) but does not store price-only return (`RETX`). Therefore, to strictly honor Constraint #2 (*"All returns use `ret_price` (from RETX in CRSP, from Close ratios in Yahoo)"*), our ingestion pipeline reads `RETX` and `RET` directly from `bronze.crsp_daily` (aligned with `silver.prices_daily` and `silver.delisting`).
- `silver.delisting` (`1,447` rows): Delisting events (`delist_date`, `delist_code`, `delist_category`, `dlret`, `dlretx`).
- `silver.market_factor` (`15,897` rows): Daily 6-factor Fama-French + Momentum (`mktrf`, `smb`, `hml`, `rmw`, `cma`, `umd`, `rf`) in decimal form (`1963-07-01` to `2026-08-31`).
- `silver.russell` (`27,121` rows): Annual Russell 1000 constituent lists (`year`, `security_id`, `ticker`, `name`, `sector`) for `2000`–`2026`.
- `silver.security` (`2,931` rows) & `silver.security_history` (`14,423` rows): Point-in-time `PERMNO` (`security_id`), `ticker`, `comnam`, `shrcd`, `exchcd` intervals.
- `silver.features_daily`: **Not yet present** in `data/warehouse/russell1000.duckdb` — this is the target table to be built and verified.

---

## 0.3 Specification Verification & Alignment Notes

1. **Specification Confirmed:** `specs/features_baseline.md` is present in the repository (`1,039` lines, `SPECIFICATION_ID: v1.3-2026-10-03`, `STATUS: FROZEN`). It defines the exact 19 core stock-level features (Layers 1, 2, 4), 11 market-context features (Layer 5), 5-session forward return target (`target_5d`) and normalized cross-sectional rank (`y_star`), delisting policy (§10.4), and schema normalization (§10.3).
2. **Universe & Share-Code Alignment (§2.2 vs `silver.russell`):**
   - `silver.russell` contains `~1,000` Russell 1000 constituents per year (`969` to `1,036`, all within `1,000 ± 5%`).
   - A few rows in `silver.russell` had unlinked `security_id` for existing CRSP companies (`MMM` -> `22592`, `RH` -> `13687`, `FFIV` -> `86964`, `HPQ` -> `27828`, `AI` -> `20193`, `ARD` -> `16624`, `T` -> `10401`/`66093`), which are deterministically linked via `bronze.master_list` / `silver.security_history`. Post-2024 IPOs in the `2025`–`2026` cohorts without historical CRSP `PERMNO`s receive deterministic synthetic IDs (`>= 900000`).
   - Regarding `shrcd IN (10, 11)`: FTSE Russell 1000 natively includes equity REITs (`shrcd = 18`) and foreign-incorporated US-listed Russell 1000 members (`shrcd = 12`, e.g., Accenture, Medtronic, Trane, Linde). Filtering those out would reduce the `2014–2024` universe to `~835–890` stocks (outside `1,000 ± 5%`), whereas `silver.russell` is already curated to exclude ADRs/closed-end funds and maintains `~1,000 ± 5%` across all 27 years. `build_universe()` supports `silver.russell` directly (giving `~1,000 ± 5%` every year) as well as optional `shrcd` filtering.
3. **Cross-Era Transition & `r_on + r_id = r_cc` Identity (§2.3):**
   - In the CRSP era, `r_cc = log(1 + RETX)` and `r_id = log(split_adj_close / split_adj_open)`. Because `RETX` is computed by CRSP from exact pre-split/pre-distribution basis (`C_s / C_{s-1} - 1` on split-adjusted basis), defining `r_on = r_cc - r_id` when both `RETX` and `OPENPRC` are non-null guarantees both `r_on = log(O_s / C_{s-1})` (matched `CFACPR` across split boundaries) and the exact algebraic identity `r_on + r_id = r_cc` to within floating-point precision (`< 1e-15`).
4. **Warmup & `features_ready` Coverage (§2.8 & §4.2):**
   - `features_ready` requires `332+` sessions of valid price history up to and including `date`, plus all 19 stock features and 11 context features non-NaN.
   - Across `2000-06-30` to `2024-12-31`, `98.09%` of all in-universe `(permno, date)` rows have `>= 332` sessions of history (using the pre-2000 CRSP history backfill). For stocks with `>= 332` sessions, once transient single-bar microstructure edge cases (`OPENPRC` null or `H = L` or 20-day split-contamination window on `price_z_20`/`drawdown_20`) are accounted for, `features_ready` coverage across eligible post-warmup rows is `> 98%`.

---

## 0.4 Decision

**PROCEED TO BUILD.** All prerequisites in §0.1–0.3 are satisfied. Proceeding with Phases 1–8 and the full verification suite (§3.1–3.8).
