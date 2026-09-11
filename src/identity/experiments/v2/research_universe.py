"""
Section 9: Research Universe Taxonomy, Filtering & Classification Validation
=============================================================================
Defines the strict research universe eligibility policy:
- Common Stock (CS) -> INCLUDE
- American Depositary Receipt (ADR) -> QUARANTINE (requires explicit researcher configuration)
- Exchange-Traded Fund (ETF) -> EXCLUDE
- Unit (UNIT) -> EXCLUDE
- Warrant (WARRANT) -> EXCLUDE
- Preferred Stock (PREFERRED) -> EXCLUDE
- Other / Unknown -> QUARANTINE

Validates classification precision across Massive PIT types and OpenFIGI types.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

from src.identity.resolver.model import (
    SecurityType,
    UniverseStatus,
    classify_universe_status,
    normalize_security_type,
)
from src.identity.massive.worker_pool import ConcurrentKeyWorkerPool

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SPELLS_PATH = REPO_ROOT / "data" / "universe" / "spells.csv"
OUT_DIR = REPO_ROOT / "data" / "identity" / "experiments" / "v2"
OPENFIGI_CACHE_PATH = REPO_ROOT / "data" / "raw" / "openfigi" / "openfigi_cache.parquet"

OUT_PARQUET = OUT_DIR / "security_type_validation.parquet"
OUT_MD = OUT_DIR / "research_universe_policy.md"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("research_universe")


def run_universe_validation():
    logger.info("=" * 80)
    logger.info("STARTING RESEARCH UNIVERSE TAXONOMY & VALIDATION (Section 9)")
    logger.info("=" * 80)

    pool = ConcurrentKeyWorkerPool(min_per_key_interval=12.1, logger=logger)
    df_spells = pl.read_csv(SPELLS_PATH)

    # Load OpenFIGI mapping for security types
    openfigi_types = {}
    if OPENFIGI_CACHE_PATH.exists():
        df_of = pl.read_parquet(OPENFIGI_CACHE_PATH)
        for r in df_of.iter_rows(named=True):
            openfigi_types[r["query_ticker"].upper()] = r.get("top_security_type")

    # Sample representative tickers across all asset classes
    curated_universe = {
        # Common Stocks
        "AAPL": "COMMON_STOCK", "MSFT": "COMMON_STOCK", "CAT": "COMMON_STOCK", "JNJ": "COMMON_STOCK",
        "GOOG": "COMMON_STOCK", "GOOGL": "COMMON_STOCK", "ACMR": "COMMON_STOCK", "MON": "COMMON_STOCK",
        "META": "COMMON_STOCK", "SHOP": "COMMON_STOCK", "NOW": "COMMON_STOCK",
        # ADRs
        "ASML": "ADR", "TSM": "ADR", "BABA": "ADR", "NVO": "ADR", "AZN": "ADR",
        # ETFs
        "AAA": "ETF", "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "VTI": "ETF", "TLT": "ETF",
        # Warrants
        "AAC.WS": "WARRANT", "AAB.WS": "WARRANT", "BTX.WSw": "WARRANT", "ASTLW": "WARRANT",
        # Units
        "AAC.U": "UNIT",
        # Preferred
        "PSA.PR.B": "PREFERRED", "BAC.PR.B": "PREFERRED",
    }

    # Add 100 sampled random spells to test broad coverage
    broad_sample = df_spells.sample(n=120, seed=42)["ticker"].to_list()
    eval_tickers = sorted(list(set(curated_universe.keys()).union(set(broad_sample))))

    df_test = df_spells.filter(pl.col("ticker").is_in(eval_tickers)).sort(["ticker", "spell_seq"])
    logger.info("Evaluating %d spells for universe classification...", df_test.height)

    records = []
    for s in df_test.iter_rows(named=True):
        tk = s["ticker"].strip().upper()
        seq = s["spell_seq"]
        s_date = s["start_date"]
        e_date = s["end_date"]
        dur = s.get("n_sessions") or s.get("duration_sessions", 1)

        rec, _ = pool.query(tk, s_date, spell_id=f"UNIV_{tk}_{seq}")
        massive_type_raw = rec.get("type") if rec else None
        massive_norm = normalize_security_type(massive_type_raw) if massive_type_raw else None

        of_type_raw = openfigi_types.get(tk)
        of_norm = normalize_security_type(of_type_raw) if of_type_raw else None

        # Resolve primary normalized type (Massive PIT takes precedence, OpenFIGI fills gaps)
        final_norm = SecurityType.UNKNOWN
        type_source = "NONE"
        if massive_norm and massive_norm != SecurityType.UNKNOWN:
            final_norm = massive_norm
            type_source = "MASSIVE_PIT"
        elif of_norm and of_norm != SecurityType.UNKNOWN:
            final_norm = of_norm
            type_source = "OPENFIGI"

        univ_status = classify_universe_status(final_norm)

        expected_class = curated_universe.get(tk)
        is_curated = (expected_class is not None)
        curated_match = None
        if is_curated:
            curated_match = (final_norm == expected_class)

        records.append({
            "ticker": tk,
            "spell_seq": seq,
            "start_date": s_date,
            "end_date": e_date,
            "duration_sessions": dur,
            "massive_type_raw": massive_type_raw,
            "massive_type_norm": massive_norm,
            "openfigi_type_raw": of_type_raw,
            "openfigi_type_norm": of_norm,
            "resolved_security_type": final_norm,
            "type_source": type_source,
            "universe_status": univ_status,
            "expected_class": expected_class,
            "is_curated_test": is_curated,
            "classification_correct": curated_match,
        })

    df_results = pl.DataFrame(records)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_results.write_parquet(OUT_PARQUET)
    logger.info("Saved universe validation parquet to %s (%d records)", OUT_PARQUET, df_results.height)

    # Calculate metrics
    curated_subset = df_results.filter(pl.col("is_curated_test") == True)
    curated_correct = curated_subset.filter(pl.col("classification_correct") == True).height
    curated_total = curated_subset.height
    curated_acc = (curated_correct / max(1, curated_total)) * 100

    type_counts = df_results.group_by("resolved_security_type").agg(pl.len().alias("count")).sort("count", descending=True)
    status_counts = df_results.group_by("universe_status").agg(pl.len().alias("count")).sort("count", descending=True)

    md_content = f"""# Section 9: Research Universe Policy & Security-Type Classification Report

## 1. Research Universe Eligibility Policy
The quantitative research mandate requires isolating **operating-company US Common Stocks**. Other financial instruments introduce systematic structural distortions (e.g. tracking error in ETFs, leverage/expiry in warrants, bundled rights in units, senior liquidation in preferreds, foreign currency/depository mechanics in ADRs).

### Policy Taxonomy Matrix
| Security Type | Action | Justification / Operational Handling |
| :--- | :--- | :--- |
| **Common Stock (`COMMON_STOCK`)** | **`INCLUDE`** | Core target universe for cross-sectional equities alpha. |
| **American Depositary Receipt (`ADR`)** | **`QUARANTINE`** | Foreign issuers trading in US markets. Default excluded; requires explicit researcher flag to include. |
| **Exchange-Traded Fund (`ETF`)** | **`EXCLUDE`** | Basket vehicle with creation/redemption mechanics. Excluded from equity factor modeling. |
| **Unit (`UNIT`)** | **`EXCLUDE`** | Bundled common shares + warrants (primarily SPACs). Excluded due to split optionality. |
| **Warrant (`WARRANT`)** | **`EXCLUDE`** | Derivative leverage instrument. Excluded from cash equity pricing. |
| **Preferred Stock (`PREFERRED`)** | **`EXCLUDE`** | Hybrid debt/equity fixed income claim. Excluded from common equity momentum/reversal. |
| **Unknown / Unresolved (`UNKNOWN`)** | **`QUARANTINE`** | Insufficient metadata. Excluded from live backtests until audited. |

---

## 2. Empirical Classification Validation Results
- Curated Multi-Asset Ground Truth Cases: **{curated_total}**
- Correctly Classified: **{curated_correct}**
- **Classification Accuracy**: **{curated_acc:.1f}%**

### Distribution of Evaluated Spells by Asset Class
| Resolved Security Type | Spell Count | Universe Status |
| :--- | :--- | :--- |
"""
    for r in type_counts.iter_rows(named=True):
        st = classify_universe_status(r["resolved_security_type"])
        md_content += f"| `{r['resolved_security_type']}` | **{r['count']}** | `{st}` |\n"

    md_content += f"""
### Universe Filter Distribution
| Universe Status | Spell Count | Action in Research Backtest |
| :--- | :--- | :--- |
"""
    for r in status_counts.iter_rows(named=True):
        action = "Admitted to investment universe" if r["universe_status"] == UniverseStatus.INCLUDE else ("Quarantined / flagged for audit" if r["universe_status"] == UniverseStatus.QUARANTINE else "Strictly excluded from data pipelines")
        md_content += f"| `{r['universe_status']}` | **{r['count']}** | {action} |\n"

    md_content += """
---

## 3. Curated Test Case Evidence Table
| Ticker | Spell | Expected Class | Resolved Type | Source | Universe Decision | Correct? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r in curated_subset.iter_rows(named=True):
        md_content += f"| `{r['ticker']}` | {r['spell_seq']} | `{r['expected_class']}` | `{r['resolved_security_type']}` | {r['type_source']} | `{r['universe_status']}` | {'PASS' if r['classification_correct'] else 'REVIEW'} |\n"

    md_content += """
---

## 4. Production Integration Rules
1. **Multi-Source Fallback**: Type detection strictly prioritizes Massive PIT classification (`CS`, `ETF`, `WAR`, etc.). If Massive is unclassified, OpenFIGI `security_type2` is checked.
2. **Deterministic Status**: Every resolved spell in the production catalog receives an explicit `research_universe_status` field (`INCLUDE`, `EXCLUDE`, `QUARANTINE`).
3. **Audit Trail**: Backtests filter strictly on `research_universe_status == 'INCLUDE'`, preventing warrants, units, and ETFs from contaminating historical cross-sectional regressions.
"""

    OUT_MD.write_text(md_content, encoding="utf-8")
    logger.info("Saved research universe report to %s", OUT_MD)


if __name__ == "__main__":
    run_universe_validation()
