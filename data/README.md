# Data Architecture & Documentation

This directory houses data specifications, dictionaries, and directory structures for the Quantitative Alpha MLOps Platform.

> **Note on Version Control:** Large raw archives, intermediate parquet partitions, and continuous panels are excluded from Git via `.gitignore`. This document outlines the expected data layout and reproduction workflow.

---

## 1. Directory Layout

```text
data/
├── raw/                      # Raw FTSE Russell constituent snapshot files (2000–2026)
│                             # 19 PDFs, 2 JSONs, 1 CSV, 2 XML XLS, 3 recovered lists
├── WRDS/                     # Annual CRSP US Daily Stock Parquet files (2000.parquet ... 2026.parquet)
│                             # Standard CRSP variables (PERMNO, date, PRC, VOL, RET, CFACPR, etc.)
├── Russell1000/              # Merged annual panels (russell1000_by_permno.parquet, by_year.parquet)
├── US_history.parquet        # (Optional) CRSP 1925–2024 security master for pre-index backfilling
├── ff.csv                    # Fama-French benchmark factor daily series (Mkt-RF, SMB, HML, RF)
├── dictionary.md             # Complete CRSP/WRDS data dictionary and variable definitions
└── processed/
    ├── continuous_panel.parquet     # Unified cross-sectional continuous panel (1925–2026)
    ├── universe_mask.parquet        # Point-in-time constituent membership mask
    ├── russell1000_all_years.csv    # Consolidated constituent membership list
    ├── 2000.txt ... 2026.txt        # Standardized constituent ticker lists per year
    └── panel_validation_report.md   # Deduplication, coverage, and continuity validation report
```

---

## 2. Primary Data Sources

1. **FTSE Russell 1000 Reconstitution Snapshots (2000–2026)**
   - Source: Official FTSE Russell index membership releases and historical reconstitution files.
   - Frequency: Annual reconstitution cycle ($t_{\text{June}} \rightarrow t_{\text{June}+1}$).
   - Parsed by: `tools/extract_russell_tickers.py`.

2. **CRSP US Daily Stock Database (WRDS, 2000–2024)**
   - Source: Wharton Research Data Services (WRDS) CRSP Daily Stock file.
   - Primary Key: `(PERMNO, date)`.
   - Coverage: 2,836 unique PERMNOs across Russell 1000 constituents.
   - Full variable specifications: See [`dictionary.md`](./dictionary.md).

3. **Yahoo Finance / Extended Market Data (2025–2026)**
   - Used to extend daily OHLCV continuity beyond the WRDS 2024-12-31 cutoff.
   - Reconciled with corporate action adjustments and cross-source boundary checks.

4. **CRSP Security Master (`US_history.parquet`)**
   - Deep historical CRSP database spanning 1925–2024.
   - Used by `tools/build_continuous_panel.py` to backfill pre-index trading history for constituents entering the index, ensuring uninterrupted 40-day (and up to 332-day) feature lookback windows.

---

## 3. Data Processing & Panel Reconstruction Pipeline

To rebuild the standardized datasets from raw files:

1. **Extract Constituent Lists**:
   ```bash
   python tools/extract_russell_tickers.py
   ```
   Extracts tickers from `data/raw/` (PDFs, JSONs, XLS) into `data/processed/{year}.txt` and `data/processed/russell1000_all_years.csv`.

2. **Construct Continuous Panel & PiT Mask**:
   ```bash
   python tools/build_continuous_panel.py
   ```
   - Ingests WRDS daily parquets and 2025–2026 continuation series.
   - Backfills pre-index history from `data/US_history.parquet`.
   - Produces `data/processed/continuous_panel.parquet` and `data/processed/universe_mask.parquet`.
   - Generates the full validation report at [`processed/panel_validation_report.md`](./processed/panel_validation_report.md).

3. **Execute Comprehensive Data Quality Audit**:
   ```bash
   python tools/run_audit.py
   ```
   Validates mathematical price bounds, reconstitution period semantics, lookback eligibility, and cross-source consistency.

---

## 4. Key Documentation Links

- [CRSP/WRDS Data Dictionary](./dictionary.md): Field-level descriptions, CRSP negative price conventions, split factors, and return calculations.
- [Continuous Panel Validation Report](./processed/panel_validation_report.md): Multi-distribution deduplication, overlap audits, and pre-index history coverage statistics.
