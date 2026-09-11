# Section 6: Dot-Notation & Symbol-Alias Validation Report

**Investigation Scope**: Punctuation & Symbology Normalization Layer  
**Target Population**: 6,837 Punctuated / Dot / Extended Tickers in `spells.csv`  
**Validation Standard**: Candidate Generation with Multi-Vendor Grounding (No Silent String Mutations)  
**Date**: September 2026  

---

## 1. Executive Summary & Policy

A common failure mode in quantitative equity pipelines is **naive dot-stripping** (`remove(".") -> canonical_ticker`), which silently mutates tickers and collapses distinct financial instruments.

### V2 Architectural Implementation:
1. **Zero Original String Mutation**: The original snapshot string (e.g. `CMCS.A`, `BRK.A`, `METpF`) is immutably preserved in all tables.
2. **Multi-Candidate Generation Layer**: Generates structured hypotheses with explicit provenance:
   - `NASDAQ_CONCATENATION` (`CMCS.A` -> `CMCSA`)
   - `NYSE_SLASH_NOTATION` (`BRK.A` -> `BRK/A`)
   - `BLOOMBERG_SPACE_NOTATION` (`BRK.A` -> `BRK A`)
   - `STANDARD_PREFERRED_STRING` (`METpF` -> `MET PRF`)
3. **External Grounding Required for Acceptance**: An alias candidate is promoted to `VALIDATED_ALIAS` only if it independently resolves to an active security in vendor reference databases.

---

## 2. Transformation Performance Breakdown

| Transformation Rule | Candidates Generated | Vendor Corroborated | Success Rate |
| :--- | ---:| ---:| ---:|
| `ORIGINAL_LITERAL` | 6837 | 3566 | 52.2% |
| `BLOOMBERG_SPACE_NOTATION` | 3771 | 0 | 0.0% |
| `NYSE_SLASH_NOTATION` | 3771 | 0 | 0.0% |
| `STANDARD_PREFERRED_STRING` | 1977 | 0 | 0.0% |
| `NYSE_PREFERRED_SLASH` | 1977 | 0 | 0.0% |
| `NORMALIZED_CASE_PREFERRED` | 1977 | 0 | 0.0% |
| `NASDAQ_CONCATENATION` | 1652 | 707 | 42.8% |
| `STRIP_DOT_INSTRUMENT_EXTENSION` | 1348 | 74 | 5.5% |
| `STRIP_TRAILING_LOWERCASE_W` | 866 | 759 | 87.6% |

---

## 3. False Alias Collision Audit

A false alias collision occurs when two different historical tickers normalize to the **same candidate string**, risking accidental identity collapse:

| Normalized Candidate Alias | Colliding Original Tickers Count | Colliding Original Ticker Strings |
| :--- | :---: | :--- |
| — | 0 | None (Zero Collisions) |

### Critical Architectural Finding:
- Naively stripping punctuation causes dual-class equities, preferred shares, and warrants to collide with core common stock symbols.
- By enforcing **multi-attribute identity resolution** (matching on CIK, FIGI, and Security Type, rather than ticker string alone), V2 guarantees that even when an alias is evaluated, instruments of different types (e.g. Common Stock vs Preferred vs Unit) can **never collapse into the same security**.
