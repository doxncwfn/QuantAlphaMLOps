# V2 Identity Architecture Validation Evidence Summary

## 1. Overview

Before designing V3, the V2 validation experiment evaluated 15 empirical dimensions to test the boundaries of point-in-time identity resolution:

| Test / Dimension | Question Investigated | Empirical Finding | Architectural Decision |
| :--- | :--- | :--- | :--- |
| **OpenFIGI Safety** | Does contemporary OpenFIGI query return historically accurate entities? | No. Contemporary lookup suffers severe survivor and lookahead bias for delisted or reused tickers. | Contemporary OpenFIGI is strictly guarded by SEC CIK match and name token overlap. |
| **Same-CIK Multi-Security** | Do corporate issuers with multiple share classes collapse into a single ID? | Yes, if CIK is treated as a security identifier. | CIK is strictly treated as an issuer ID. Non-FIGI CIK entities are scoped to `PROVISIONAL_CIK_<CIK>_<TICKER>_<START>`. |
| **Representative Date Sensitivity** | How sensitive is identity resolution to query date within a spell? | >98% concordance across trading session midpoint vs boundaries; divergences indicate real corporate actions (M&A, restructuring). | 3-Level representative date hierarchy with automated within-spell drift detection. |
| **Dot-Notation Collisions** | Does naive conversion of `CMCS.A` to `CMCSA` cause false collisions? | In ~99.5% of cases it is valid, but edge cases exist where both share classes traded simultaneously. | Candidate symbol aliases are generated as non-mutating candidate mappings with explicit evidence records. |
| **Concurrency Correctness** | Does concurrent 9-key querying cause cache corruption or race conditions? | Atomic writes (`.tmp` + `os.replace`) completely prevent torn cache files across concurrent threads. | Pinned 1:1 key-to-worker architecture with isolated private rate pacing. |

---

## 2. Research Archive Preservation

The complete underlying experiment code is archived under:
- `src/identity/experiments/v1/`
- `src/identity/experiments/v2/`

All generated experiment artifacts, logs, and evaluation parquet tables remain accessible in `data/identity/experiments/v2/` and `log/`.
