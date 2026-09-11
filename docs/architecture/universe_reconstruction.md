# US-Market Universe Reconstruction Methodology

## 1. Lifecycle Overview

The universe reconstruction pipeline reconstructs a survivorship-bias-free historical US equity universe from point-in-time exchange and vendor snapshots:

```text
Daily Active Symbol Snapshots (5,699 trading days, 2003–2025)
    ↓
Data Cleaning & Corruption Exclusion (Excluding 3 corrupted dates: 2009-10-29, 2010-03-30, 2010-03-31)
    ↓
Contiguous Ticker Availability Spells (43,757 spells, 36,843 unique tickers)
    ↓
Point-in-Time Security Identity Resolution (Massive PIT + OpenFIGI + SEC EDGAR)
    ↓
Security Availability Episodes (Bridging non-material snapshot dropouts)
    ↓
Canonical Daily Research Universe (Common-stock only, tradable, survivorship-bias free)
```

---

## 2. Key Data Artifacts

| Dataset | Location | Role |
| :--- | :--- | :--- |
| `spells.csv` | `data/universe/spells.csv` | Immutable input manifest (SHA-256: `5fc79a37cdc341cf...`). 43,757 records. |
| `trading_sessions.parquet` | `data/identity/experiments/v2/` | Full NYSE/NASDAQ trading calendar (5,699 sessions). |
| `massive_manifest.parquet` | `data/manifests/v3/` | Merged point-in-time resolution results. |
| `security_master.parquet` | `data/identity/candidates/v3/` | Master table of unique canonical and provisional securities. |
| `ticker_history.parquet` | `data/identity/candidates/v3/` | Mapping from each ticker spell to its assigned security_id. |
| `availability_episodes.parquet` | `data/universe/candidates/v3/` | Security-level contiguous trading intervals. |
| `daily_universe.parquet` | `data/universe/candidates/v3/` | Daily cross-sectional eligible asset membership. |

---

## 3. Data Safety & Invariant Guarantees

1. **Bit-for-Bit Immutability**: `data/universe/spells.csv` is strictly read-only.
2. **Promotion Gate**: No candidate artifact is promoted to `data/identity/security_master.parquet` until passing all 31 automated architectural invariants (`src/identity/validation/invariant_suite.py`) and whole-dataset quality audits (`src/identity/validation/quality_suite.py`).
3. **Resumable Cloud Backfill**: Checkpoint chunks are written every $N$ spells and committed to persistent cloud volumes, enabling interrupted jobs to resume without duplicate API requests.
