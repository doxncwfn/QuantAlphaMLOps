# Project Reports & Research Documentation

This directory is the centralized home for all audit deliverables, technical research specifications, data quality reports, and progress tracking documentation across the project.

---

## 1. Directory Structure

```text
report/
├── README.md               # Guide to project reports, specifications, and audit deliverables
├── quality/                # Automated data quality & Point-in-Time readiness audit
│   ├── final_pit_readiness_audit.md              # Final PiT universe readiness declaration
│   ├── russell1000_data_quality_audit_report.md  # Comprehensive 2000–2026 data quality audit
│   ├── audit_input_manifest.json                 # Cryptographic SHA-256 input file manifest
│   ├── source_field_mapping.yaml                 # Field harmonizations (CRSP WRDS vs crawled)
│   ├── figures/                                  # (Generated) 21 publication audit figures
│   └── tables/                                   # (Generated) 21 structured audit summary tables
├── research/               # Technical specifications & forensic empirical research reports
│   ├── v1_feature_specification.md               # Authoritative v1 feature engineering specification
│   ├── final_eligibility_specification_report.md # Model lookback eligibility & stitching specification
│   ├── final_target_definition_audit.md          # Forward target (t+1 -> t+5) specification & reconciliation
│   ├── missingness_mechanism_audit_report.md     # In-depth MCAR/MAR missingness analysis
│   ├── delisting_target_audit.md                 # Delisting returns & terminal cash event modeling
│   ├── ohlcv_factor_forensic_eda.md              # Forensic OHLCV and benchmark factor data analysis
│   └── EDA_report.md                             # Overview EDA on initial 60-day window feasibility
└── progress/               # Project milestones, implementation plans, and meeting notes
    ├── implementation_plan.md                    # Research and engineering implementation roadmap
    ├── problem.excalidraw                        # Architecture and problem space diagrams
    ├── questions.md                              # Advisor discussion notes and research decisions
    └── w0.md                                     # Foundational project phase notes
```

---

## 2. Report Categories & Descriptions

### A. Data Quality & PiT Readiness (`report/quality/`)
Production-grade audit reports validating market integrity across the Russell 1000 universe:
- [`final_pit_readiness_audit.md`](./quality/final_pit_readiness_audit.md): Complete multi-phase readiness scorecard establishing data sufficiency for deep neural ranking models.
- [`russell1000_data_quality_audit_report.md`](./quality/russell1000_data_quality_audit_report.md): Formal data quality audit covering negative price conventions, split adjustments, boundary transitions, and reconstitution mapping.
- Summary tables and figures are populated automatically via:
  ```bash
  python tools/run_audit.py
  ```

### B. Research Specifications & Forensic Audits (`report/research/`)
Milestone research deliverables and mathematical specifications locked for model development:
- [`v1_feature_specification.md`](./research/v1_feature_specification.md): Technical specification of input features, lookback warmups, ranking targets, and leakage barriers.
- [`final_eligibility_specification_report.md`](./research/final_eligibility_specification_report.md): Mathematical definition of the 60-session input window, observation masks, and lookback continuity.
- [`final_target_definition_audit.md`](./research/final_target_definition_audit.md): Forward return target ($t+1 \rightarrow t+5$) reconciliation, market residualization, and terminal returns.
- [`missingness_mechanism_audit_report.md`](./research/missingness_mechanism_audit_report.md): Classification of missingness mechanisms (MCAR vs MAR vs structural halts vs delistings).
- [`delisting_target_audit.md`](./research/delisting_target_audit.md): Delisting return conventions and corporate action handling.
- [`ohlcv_factor_forensic_eda.md`](./research/ohlcv_factor_forensic_eda.md): Forensic analysis of raw OHLCV price series and Fama-French benchmark factor series.
- [`EDA_report.md`](./research/EDA_report.md): Feasibility analysis of the 60-session input lookback across historical annual cohorts.

### C. Progress & Engineering Notes (`report/progress/`)
- [`implementation_plan.md`](./progress/implementation_plan.md): Active development roadmap, engineering status, and risk registry.
