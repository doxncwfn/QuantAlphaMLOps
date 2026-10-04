# Forensic Research & Exploratory Analysis Environment

This directory houses the computational forensic analysis suite, data anomaly audits, and empirical artifacts developed for the historical Russell 1000 dataset (2000–2026).

Written research reports, target specifications, and audit whitepapers are centralized in [`../report/research/`](../report/research/).

---

## 1. Directory Structure

```text
analysis/
├── README.md               # Guide to forensic audit suite and analytical outputs
├── forensics/              # Modular forensic audit engine & research CLI package
│   ├── __init__.py         # Programmatic entry points for audit runners
│   ├── __main__.py         # Unified CLI dispatcher (python -m analysis.forensics)
│   ├── common.py           # Shared paths, directory setup, and LogWriter utilities
│   ├── inventory.py        # Part 1: Schema inventory, deduplication, & impossible record checks
│   ├── adjustments.py      # Parts 2–5: Price adjustments, split detection, dividends, & volume
│   ├── factors.py          # Parts 6–8: Fama-French benchmark factors & trading date alignment
│   ├── delistings.py       # Parts 9–12: Delisting codes, terminal prices, & target liquidation
│   ├── features.py         # Parts 13–17: Feature feasibility & source boundary drift (2024 vs 2025)
│   ├── missingness.py      # Missing run-length distributions & empirical retention curves
│   └── eligibility.py      # Target definition lock & lookback eligibility reconciliation
└── outputs/                # Empirical research artifacts (cleanly partitioned)
    ├── figures/            # Visualizations of empirical retention curves and missing streak lengths (.png)
    ├── metrics/            # Machine-readable audit metrics and reconciliation summaries (.json)
    └── examples/           # Targeted empirical samples of corporate action adjustments (.csv)
```

---

## 2. Forensic Research Suite (`analysis/forensics/`)

The forensic analysis suite is organized into a cohesive, importable Python package with a unified CLI dispatcher.

### Unified Command-Line Interface

Run all forensic modules sequentially:
```bash
python -m analysis.forensics --task all
```

Or run any individual forensic audit module:
```bash
# 1. Dataset inventory, schemas, and anomaly checks
python -m analysis.forensics --task inventory

# 2. Split adjustments, dividend adjustments, and volume scaling
python -m analysis.forensics --task adjustments

# 3. Fama-French factor series & trading calendar alignment
python -m analysis.forensics --task factors

# 4. Delisting codes (DLSTCD) and liquidation returns
python -m analysis.forensics --task delistings

# 5. Feature feasibility and cross-source boundary stability (2024 CRSP vs 2025 Yahoo)
python -m analysis.forensics --task features

# 6. Missingness run lengths and empirical retention curves
python -m analysis.forensics --task missingness

# 7. Final Point-in-Time target lock and eligibility reconciliation
python -m analysis.forensics --task eligibility
```

### Programmatic Python API

All audit functions are also directly importable for notebook exploration or test pipelines:

```python
from analysis.forensics import (
    run_inventory,
    run_adjustments,
    run_factors,
    run_delistings,
    run_features,
    run_missingness,
    run_eligibility,
)

# Execute specific audit
inventory_results = run_inventory()
```

---

## 3. Related Documentation

Written research reports corresponding to these forensic analyses are housed in [`../report/research/`](../report/research/):

| Document | Description |
| :--- | :--- |
| [`report/research/v1_feature_specification.md`](../report/research/v1_feature_specification.md) | Authoritative technical specification for the v1 feature library, lookback warmups, cross-sectional ranking targets, and leakage barriers. |
| [`report/research/final_eligibility_specification_report.md`](../report/research/final_eligibility_specification_report.md) | Formal specification of model lookback eligibility, continuous multi-year stitching, and reconstitution boundary rules. |
| [`report/research/final_target_definition_audit.md`](../report/research/final_target_definition_audit.md) | Forensic evaluation of cross-sectional forward return targets ($t+1 \rightarrow t+5$), market residualization, and delisting returns. |
| [`report/research/missingness_mechanism_audit_report.md`](../report/research/missingness_mechanism_audit_report.md) | In-depth classification of data missingness mechanisms (MCAR vs. MAR vs. structural halts vs. terminal delistings). |
| [`report/research/delisting_target_audit.md`](../report/research/delisting_target_audit.md) | Audit of delisting return calculations, fallback conventions, and liquidation modeling. |
| [`report/research/ohlcv_factor_forensic_eda.md`](../report/research/ohlcv_factor_forensic_eda.md) | Forensic analysis of raw OHLCV price series, volume conventions, split adjustments, and extreme price movement detection. |
| [`report/research/EDA_report.md`](../report/research/EDA_report.md) | Overview EDA report on initial 60-day window feasibility and constituent coverage across time. |

---

## 4. Empirical Artifacts & Outputs

All generated forensic artifacts are strictly isolated in `analysis/outputs/`:

- **Figures (`analysis/outputs/figures/`)**:
  - `retention_vs_m_max.png`: Empirical candidate retention curve as a function of maximum allowed missing sessions $M_{\max}$.
  - `retention_by_year.png`: Candidate retention rate by annual cohort.
  - `missing_streaks_distribution.png`: Histogram and CDF of consecutive missing observation streaks ($r$).
  - `missingness_over_time.png`: Cross-sectional missing rate across the 2000–2026 timeline.
  - `window_missingness_composition.png`: Window composition breakdown (clean, isolated gaps, edge gaps).
  - `streaks_distribution_post2000.png`: Post-2000 missing streak characteristics.
- **Metrics (`analysis/outputs/metrics/`)**:
  - `audit_part1_results.json` through `audit_part17_results.json`: Machine-readable results from the forensic audit phases.
  - `missingness_mechanism_metrics.json`: Structured missingness metrics.
  - `eligibility_reconciliation_metrics.json`: Reconciliation metrics.
- **Targeted Examples (`analysis/outputs/examples/`)**:
  - Empirical case studies demonstrating split corrections (`split_adjustment_samples.csv`), dividend adjustments (`dividend_adjustment_samples.csv`), source transition comparisons (`source_transition_comparison.csv`), and date alignments (`factor_date_alignment_boundary.csv`).
