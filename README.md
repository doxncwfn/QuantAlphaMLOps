# Quantitative Alpha MLOps Platform

**STATUS:** 
- Ingestion layer = FROZEN
- PIT universe construction = NOT STARTED

This is the main repository for the Quantitative Alpha MLOps project. It features an integrated historical US Equity data pipeline alongside machine learning and evaluation infrastructure.

## 1. Project Purpose
To provide a best-effort, survivorship-aware US equity daily OHLCV dataset by rigorously reconstructing historical ticker-to-CIK mappings from SEC filings, then joining them with Yahoo Finance prices, preserving deep provenance. This data serves as the foundation for downstream PIT universe construction, feature engineering, and model training.

## 2. Architecture & Data Flow

```text
src/ingestion/
    SEC -> ticker history -> Yahoo -> canonical OHLCV
```

The ingestion subsystem extracts ticker validity intervals from SEC filings and orchestrates batch downloads from Yahoo Finance, producing canonical `.parquet` files for downstream tasks.

## 3. Directory Structure
- `src/`: Core logic.
  - `ingestion/`: SEC/Yahoo data downloading and historical reconstruction (FROZEN).
  - `data/`, `factors/`, `evaluation/`, `broker/`: Downstream research and ML modules.
- `tests/`: Reusable validation scripts, including `tests/ingestion/`.
- `scripts/`: Maintenance, migration, and audit scripts (`scripts/ingestion/audit/`).
- `docs/`: Audits, reports, and detailed structural documentation (e.g. `docs/ingestion/`).
- `data/`: The shared data root.
  - `raw/`: Raw downloaded SEC JSON and Yahoo Parquet batches.
  - `curated/`: Frozen canonical datasets and intermediate audit results.
- `logs/`: Preserved build and download logs.

## 4. Canonical Data Artifacts
Located in `data/curated/`:
- `ticker_observations.parquet` (SEC extractions)
- `ticker_history.parquet` (Reconstructed intervals)
- `yahoo_download_status.parquet` (Coverage tracking)
- `security_day_ohlcv.parquet` (Final compiled daily prices)

## 5. Raw Data Locations
- `data/raw/sec/`: Raw SEC submission JSONs.
- `data/raw/yahoo/`: Raw interval parquet files from `yfinance`.
- `data/raw/yahoo_quarantine/`: Anomalous data separated from standard processing.

## 6. How the Pipeline Works
1. Identifies SEC tickers/CIKs/exchanges.
2. Extracts observation events from corporate filings.
3. Reconstructs unbroken ticker-validity intervals for each security.
4. Downloads Yahoo Finance data mapping to those historically verified intervals.
5. Aggregates all valid Yahoo batches into a single daily OHLCV dataset.

## 7. How to Validate
Run the validation suite to assert data integrity for the ingestion layer:
```bash
python -m ingestion.pipeline validate
```

## 8. Rebuilding Canonical OHLCV from Cached Data
Since the ingestion is frozen, you can safely rebuild the final `security_day_ohlcv.parquet` from existing raw cached downloads without hitting external APIs:
```bash
python -m ingestion.pipeline build-ohlcv
```

## 9. Known Limitations
- The dataset is not full CRSP/Norgate-grade survivorship-free data.
- **SEC Corpus Limitation**: Some long-lived securities have late first observations because the corresponding older SEC filings are absent from the locally cached corpus (e.g., AMD, AAL, PEP, ORCL, ADP).
- **Structural Anomalies**: There are 277 OHLCV structural anomalies intentionally preserved in the raw canonical dataset, tracked separately for downstream handling.

## 10. Provenance / Audit Documentation
Significant audits have been conducted to trace anomalies and repair fragmentation. These critical historical investigations are preserved under `docs/ingestion/` (or `docs/audits/ingestion/`), including SEC historical recall, Yahoo anomalies, and ingestion freeze logs.

## Setup Instructions

```bash
uv sync
cp .env.example .env
```
*(Do NOT run `sec-submissions` or `yahoo` download commands as the dataset is frozen.)*
