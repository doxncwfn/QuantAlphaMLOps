# Project Documentation

This directory contains academic reports, formal research proposals, and technical specifications for the Quantitative Alpha MLOps platform.

---

## 1. Structure

```text
docs/
├── Phase_1/                  # HCMUT Specialized Project Phase 1 Report
│   ├── main.tex              # Master LaTeX document
│   ├── main.pdf              # Compiled Phase 1 academic report
│   ├── ingestion.tex         # Ingestion layer and survivorship methodology
│   ├── methodology.tex       # Research design and modeling framework
│   ├── references.bib        # Academic bibliography
│   └── images/               # Report figures and institutional logos
├── Proposal/                 # Initial Research Proposal
│   ├── proposal.tex          # LaTeX source for project proposal
│   ├── proposal.pdf          # Compiled project proposal document
│   └── proposal.md           # Markdown version of proposal
└── specifications/           # Formal Engineering & System Specifications
    └── pit_universe_and_input_window.md  # PiT Universe Construction & Input Window Spec
```

---

## 2. Compiling Academic Reports

The reports use standard LaTeX with the institutional `hcmut-report.cls` document class:

```bash
cd docs/Phase_1
latexmk -pdf main.tex
```

Compilation artifacts (`.aux`, `.log`, `.fls`, `.synctex.gz`, etc.) are excluded via `.gitignore`.
