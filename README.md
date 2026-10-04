# Quantitative Alpha MLOps Platform

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Testing](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)

An institutional-grade quantitative equity research and MLOps platform designed for survivorship-aware historical backtesting, Point-in-Time (PiT) universe reconstruction, and deep cross-sectional alpha modeling across the **Russell 1000** universe (2000–2026).

---

## 1. Project Overview

Empirical quantitative research often suffers from subtle data biases—most notably survivorship bias, look-ahead leakage, and artificial data truncation at annual cohort boundaries. This platform addresses these challenges through a unified data engineering and modeling pipeline:

- **Survivorship-Aware Universe Reconstruction**: Combines 27 annual FTSE Russell 1000 reconstitution snapshots (2000–2026) with CRSP/WRDS daily stock data and deep security-master backfills (1925–2024), recovering over 1.05M observations previously discarded by naive annual file chunking.
- **Strict Point-in-Time (PiT) Semantics**: Decouples membership status, price availability, model input eligibility, and forward target availability, strictly preventing forward-looking information contamination.
- **Deep Neural Alpha Architecture**: Hybrid model combining LSTM sequence encoders, temporal multi-head attention, and cross-sectional Transformer layers to predict relative asset rankings.
- **Production MLOps Infrastructure**: Modular pipeline supporting distributed experiment tracking (Weights & Biases), hyperparameter optimization (Optuna), explainable AI (Captum Integrated Gradients), and serverless cloud GPU scaling (Modal).

---

## 2. Project Status

| Subsystem / Component | Status | Description |
| :--- | :---: | :--- |
| **Constituent Extraction (2000–2026)** | Completed | Multi-format parser for 27 annual Russell 1000 reconstitution lists (PDF, JSON, XML XLS). |
| **Data Quality & Integrity Audit** | Completed | Rigorous validation of price bounds, negative price quotes, split adjustments, and source boundaries. |
| **Continuous Panel Construction** | Completed | Unified cross-stock panel with continuous 40-day lookback stitching and pre-index history backfills. |
| **PiT Universe & Feature Specs** | Locked | Formal engineering specifications for input tensors, masks, missingness handling, and target returns. |
| **Deep Alpha Model Architecture** | Implemented | PyTorch Lightning implementation of LSTM + Temporal Attention + Cross-Sectional Transformer. |
| **Explainable AI (XAI)** | Implemented | Integrated Gradients attribution and LightGBM baseline comparators. |
| **Portfolio TCA & Optimization** | In Progress | Covariance shrinkage and transaction cost modeling stubs; convex optimization in development. |
| **Live Execution Client** | Planned | Alpaca paper-trading broker integration stubbed for downstream deployment. |

---

## 3. Repository Structure

```text
├── config/                 # Central configuration files (audit parameters, model training configs)
│   ├── audit.yaml          # Data quality audit parameters, thresholds, and cohort specifications
│   └── config.yaml         # Deep learning hyperparameters, paths, and training settings
├── src/                    # Reusable core library source code
│   ├── audit/              # Data quality audit engine, integrity checks, and reporting pipelines
│   ├── broker/             # Brokerage integration and execution clients (Alpaca)
│   ├── data/               # Market residualization, feature preprocessing, and purged walk-forward splitters
│   ├── evaluation/         # Performance metrics (IC, Rank IC, Sharpe) and feature attribution (XAI)
│   ├── models/             # Deep neural architectures (Hybrid LSTM-Transformer, ranking losses)
│   ├── portfolio/          # Portfolio optimization and transaction cost analysis (TCA)
│   ├── training/           # PyTorch Lightning trainers, callbacks, and seed orchestration
│   ├── modal_run.py        # Remote serverless GPU training entry point (Modal)
│   ├── modal_upload.py     # Modal data volume batch synchronizer
│   └── train.py            # Local model training and Optuna study entry point
├── tools/                  # Executable utility scripts and data pipelines
│   ├── build_continuous_panel.py   # Assembles continuous historical panel with pre-index backfilling
│   ├── run_audit.py                # CLI runner for the data quality and PiT readiness audit
│   ├── extract_russell_tickers.py  # Ingestion parser for annual constituent source files
│   ├── merge_annual_parquets.py    # Merges annual WRDS parquets into unified datasets
│   ├── csv_to_parquet.py           # Efficient tabular format converter
│   └── patch_wrds_missing_securities.py # Reconciles missing security anomalies
├── data/                   # Data specifications, dictionaries, and documentation (raw data excluded)
│   ├── README.md           # Dataset architecture, sources, and layout documentation
│   ├── dictionary.md       # Comprehensive 63-variable WRDS/CRSP daily stock data dictionary
│   └── processed/          # Houses validation reports and constituent lists
├── analysis/               # Forensic analysis suite, anomaly audits, and empirical artifacts
│   ├── README.md           # Guide to forensic audit suite and analytical outputs
│   ├── forensics/          # Modular forensic audit engine & research CLI package
│   │   ├── __init__.py     # Programmatic API for audit runners
│   │   ├── __main__.py     # Unified CLI dispatcher (python -m analysis.forensics)
│   │   ├── common.py       # Shared constants, paths, and LogWriter utilities
│   │   ├── inventory.py    # Part 1: Schema inventory & impossible record checks
│   │   ├── adjustments.py  # Parts 2–5: Price, split, dividend & volume adjustments
│   │   ├── factors.py      # Parts 6–8: Benchmark factors & date alignment
│   │   ├── delistings.py   # Parts 9–12: Delisting codes & liquidation returns
│   │   ├── features.py     # Parts 13–17: Feature feasibility & source boundary drift
│   │   ├── missingness.py  # Missingness streak analysis & empirical retention curves
│   │   └── eligibility.py  # PiT target lock & model lookback eligibility reconciliation
│   └── outputs/            # Empirical forensic artifacts (figures/, metrics/, examples/)
├── docs/                   # Academic reports, research proposals, and system specifications
│   ├── README.md           # Documentation index and compilation guide
│   ├── Phase_1/            # Specialized Project Phase 1 Report (LaTeX source + compiled PDF)
│   ├── Proposal/           # Research Proposal (LaTeX source + compiled PDF)
│   └── specifications/     # Formal specifications (PiT Universe Construction & Input Windows)
├── report/                 # Quality audits, research specifications, and project progress logs
│   ├── README.md           # Summary of report structure and regeneration commands
│   ├── quality/            # Final PiT readiness report, input manifest, and field mappings
│   ├── research/           # Authoritative research specifications and forensic audit reports
│   └── progress/           # Implementation roadmap, design diagrams, and advisory meeting notes
├── notebooks/              # Interactive Jupyter notebooks
│   ├── README.md           # Guide to tracked notebooks and scratch notebook exclusion policy
│   └── audit.ipynb         # 16 publication-quality analytical visualizations for the data audit
├── tests/                  # Automated test suite
│   └── test_model_eligibility.py # Cross-boundary continuity and lookback eligibility unit tests
├── artifacts/              # (Ignored) Model checkpoints and out-of-sample prediction parquets
├── log/                    # (Ignored) Runtime logs and audit traces
├── pyproject.toml          # Project configuration, dependencies, and lint rules
├── requirements.txt        # Pinned dependency requirements for pip and cloud runtime images
└── .env.example            # Template for environment variables and API credentials
```

---

## 4. Data Requirements & Sourcing

The repository does not commit large binary datasets to version control. The required data sources and their destinations in `data/` are:

1. **FTSE Russell 1000 Constituent Lists (2000–2026)**
   - Location: `data/raw/`
   - Content: Annual reconstitution PDFs, JSONs, and XML spreadsheets.
   - Processing: Parsed by `tools/extract_russell_tickers.py` into `data/processed/`.
2. **CRSP US Daily Stock Database (2000–2024)**
   - Location: `data/WRDS/`
   - Content: Annual partitions (`2000.parquet` to `2024.parquet`) containing CRSP daily prices, shares outstanding, adjustment factors, and volume.
   - Reference: See [`data/dictionary.md`](./data/dictionary.md) for field definitions.
3. **Continuation Market Data (2025–2026)**
   - Location: `data/WRDS/2025.parquet`, `2026.parquet` (or Yahoo market feeds).
   - Content: Post-CRSP price series reconciled for corporate actions and boundary continuity.
4. **Historical CRSP Security Master (1925–2024)**
   - Location: `data/US_history.parquet`
   - Content: Full historical CRSP database utilized to backfill pre-index trading history for stocks entering the Russell 1000.
5. **Benchmark Risk Factors**
   - Location: `data/ff.csv`
   - Content: Fama-French 3-factor series and daily risk-free rate ($R_f$).

For complete setup instructions, refer to [`data/README.md`](./data/README.md).

---

## 5. Reproduction & Execution

### Running the Data Quality Audit
Assert compliance across price bounding, reconstitution period mapping, and 40-day lookback eligibility:
```bash
python tools/run_audit.py
```
This regenerates summary tables in `report/quality/tables/` and the comprehensive audit reports in `report/quality/`.

### Running Forensic Research Audits
Execute the modular empirical forensic audit suite:
```bash
# Run all audit modules sequentially
python -m analysis.forensics --task all

# Or run specific forensic modules (e.g., inventory, adjustments, missingness)
python -m analysis.forensics --task inventory
```

### Running Unit Tests
Execute the automated test suite with pytest:
```bash
pytest tests/
```

### Reconstructing the Continuous Panel
Build the unified per-stock continuous panel with pre-index backfilling:
```bash
python tools/build_continuous_panel.py
```
Outputs `data/processed/continuous_panel.parquet`, `data/processed/universe_mask.parquet`, and `data/processed/panel_validation_report.md`.

### Model Training & Experimentation
Run local model training with default configuration:
```bash
python src/train.py
```

Launch distributed training on cloud GPUs (A10G/A100) via Modal:
```bash
modal run src/modal_run.py::run_training
```

---

## 6. Installation & Environment Setup

### Prerequisites
- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) (recommended) or standard Python `venv`

### Installation with `uv`
```bash
# Clone the repository
git clone https://github.com/username/quant-alpha-mlops.git
cd quant-alpha-mlops

# Create virtual environment and install dependencies
uv sync --extra dev
```

### Installation with Standard `pip`
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Configuration
Copy the environment template and provide your API keys:
```bash
cp .env.example .env
```
Key variables include:
- `MODAL_API_TOKEN`: Cloud GPU infrastructure.
- `WANDB_API_KEY`: Experiment tracking and metric logging.
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`: Paper-trading brokerage integration.

---

## 7. Important Specifications & Reports

- **Final Pre-Model PiT Universe Readiness Audit**: [`report/quality/final_pit_readiness_audit.md`](./report/quality/final_pit_readiness_audit.md)
- **Continuous Panel Validation Report**: [`data/processed/panel_validation_report.md`](./data/processed/panel_validation_report.md)
- **PiT Universe Construction & Input Window Spec**: [`docs/specifications/pit_universe_and_input_window.md`](./docs/specifications/pit_universe_and_input_window.md)
- **Feature Library v1 Engineering Specification**: [`report/research/v1_feature_specification.md`](./report/research/v1_feature_specification.md)
- **WRDS / CRSP Data Dictionary**: [`data/dictionary.md`](./data/dictionary.md)
- **Academic Project Report (Phase 1)**: [`docs/Phase_1/main.pdf`](./docs/Phase_1/main.pdf)

---

## 8. Known Limitations & Methodological Constraints

1. **Annual Snapshot Reconstitution Approximation**: Intra-year index additions or deletions occurring between annual June reconstitution cycles are approximated by holding the annual snapshot fixed over $[t_{\text{June}}, t_{\text{June}+1})$.
2. **CRSP Negative Price Convention**: In CRSP WRDS, prices on non-trading days (`VOL == 0`) are recorded as negative values reflecting the bid/ask midpoint quote. The data pipeline takes $|PRC|$ for rolling feature lookbacks while marking the session untradable for trade execution.
3. **Source Boundary Transition**: The primary data provider transitions from CRSP/WRDS to crawled market data at 2024-12-31 / 2025-01-02. Overnight boundary returns for split-adjusted stocks are explicitly handled to prevent artificial price jumps.
4. **Delisting Liquidation**: Terminal delisting returns reflect the final available trading price or delisting payment (`DLPRC` / `DLPDT`). M&A cash events vs. bankruptcy liquidations are distinguished per the delisting audit rules.

---

## License

This project is developed as part of the Specialized Project curriculum at Ho Chi Minh City University of Technology (HCMUT).
