# V3 Identity Resolution Backfill and Promotion-Gate Documentation

## 1. Executive Summary

The V3 Point-in-Time Identity Backfill and Promotion-Gate subsystem completes historical security identity resolution across all **43,757 spells** in `data/universe/spells.csv`.

### Baseline Verification
- **Input Table**: `data/universe/spells.csv`
- **SHA-256 Hash**: `5fc79a37cdc341cf7b10a7017ccd75cf7001aa1b91e5dd6501c829b746191bf1`
- **Total Spells**: 43,757
- **Unique Tickers**: 36,843
- **Multi-Spell Tickers**: 5,555 (representing 12,469 spells)

---

## 2. Whole-Dataset Ticker-Reuse Audit Results

To rigorously evaluate against false identity merges, all 5,555 multi-spell tickers were audited:

| Category | Metric | Result |
| :--- | :--- | :--- |
| **Multi-Spell Tickers Audited** | Count | **5,555** |
| **Multi-Spell Spells Audited** | Count | **12,469** |
| **False Reuse Merges Detected** | Violations | **0** |
| **Status** | Overall Result | **PASS** |

### Benchmark Negative Controls
| Ticker | Spells | Assigned Security IDs | Audit Outcome |
| :--- | :--- | :--- | :--- |
| `ACMR` | 2 | `['BBG00HPSG942', 'PROVISIONAL_CIK_0001385534_ACMR_BECCD01D']` | **PASS (Separated)** |
| `AA` | 1 | `['BBG000BVPV84']` | **PASS (Single Spell)** |

---

## 3. Promotion-Gate Verification

Promotion from candidate datasets (`data/identity/candidates/v3/`) to production tables (`data/identity/`) requires:
1. **31/31 Formal Invariants Evaluated**: `src/identity/validation/invariant_suite.py`.
2. **Quality Suite Execution**: `src/identity/validation/quality_suite.py` (Same-CIK multi-security separation, FIGI 1:1 collision audit, ticker-reuse audit).
3. **Explicit CLI Invocation**:
   ```bash
   python3 src/identity/promotion/gate.py --confirm
   ```

Zero production tables are modified until full backfill acquisition is completed and explicitly certified.
