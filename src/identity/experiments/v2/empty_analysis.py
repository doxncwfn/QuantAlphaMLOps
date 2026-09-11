"""
Section 8: Massive Empty Response Taxonomy & Root-Cause Analysis
=================================================================
Analyzes why Massive returns empty responses for certain ticker-date queries,
quantifies the failure modes across symbology mismatches, warrant/unit suffixes,
data coverage gaps, and formalizes the policy: MASSIVE_EMPTY != SECURITY_INACTIVE.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

from src.identity.resolver.model import (
    IdentityStatus,
    normalize_security_type,
)
from src.identity.aliases.alias_engine import generate_alias_candidates
from src.identity.massive.worker_pool import ConcurrentKeyWorkerPool

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SPELLS_PATH = REPO_ROOT / "data" / "universe" / "spells.csv"
OUT_DIR = REPO_ROOT / "data" / "identity" / "experiments" / "v2"
OPENFIGI_CACHE_PATH = REPO_ROOT / "data" / "raw" / "openfigi" / "openfigi_cache.parquet"

OUT_PARQUET = OUT_DIR / "massive_empty_analysis.parquet"
OUT_MD = OUT_DIR / "massive_empty_analysis.md"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("empty_analysis")


def categorize_empty_cause(ticker: str, s_date: str, dur: int, has_figi: bool) -> str:
    """Categorizes the primary structural cause of a Massive empty response."""
    tk = ticker.strip().upper()

    if "." in tk or "/" in tk or "-" in tk:
        if any(s in tk for s in [".WS", ".W", "/WS", "-WS", ".WT"]):
            return "WARRANT_SYMBOL_FORMAT"
        if any(s in tk for s in [".U", "/U", "-U", ".UN", "/UN"]):
            return "UNIT_SYMBOL_FORMAT"
        if any(s in tk for s in [".RT", "/RT", "-RT", ".R"]):
            return "RIGHTS_SYMBOL_FORMAT"
        return "DOT_SHARECLASS_SYMBOLOGY"

    if tk.endswith("w") or tk.endswith("W") or tk.endswith("WS"):
        return "WARRANT_SUFFIX_LOWERCASE"

    if dur <= 3:
        return "EXTREME_SHORT_SPELL_TRANSIENT"

    if s_date < "2008-01-01":
        return "PRE_2008_HISTORICAL_COVERAGE_GAP"

    if has_figi:
        return "OPENFIGI_FOUND_MASSIVE_COVERAGE_GAP"

    return "UNCOVERED_MICROCAP_OR_DELISTED"


def run_empty_analysis():
    logger.info("=" * 80)
    logger.info("STARTING MASSIVE EMPTY RESPONSE TAXONOMY ANALYSIS (Section 8)")
    logger.info("=" * 80)

    pool = ConcurrentKeyWorkerPool(min_per_key_interval=12.1, logger=logger)
    df_spells = pl.read_csv(SPELLS_PATH)
    logger.info("Loaded %d spells.", df_spells.height)

    # Load OpenFIGI cache
    figi_set = set()
    if OPENFIGI_CACHE_PATH.exists():
        df_of = pl.read_parquet(OPENFIGI_CACHE_PATH)
        figi_set = set(df_of["query_ticker"].str.to_uppercase().to_list())

    # Sample spells across key buckets:
    # 1. Tickers with dots, slashes, warrants, units
    # 2. Short spells (<= 5 sessions)
    # 3. Early spells (< 2008)
    # 4. Standard spells
    dot_spells = df_spells.filter(pl.col("ticker").str.contains(r"[./\-wW]")).head(150)
    short_spells = df_spells.filter(pl.col("n_sessions") <= 3).head(100)
    sample_combined = pl.concat([dot_spells, short_spells]).unique(subset=["ticker", "spell_seq"])

    logger.info("Evaluating %d candidate spells for empty response analysis...", sample_combined.height)

    empty_records = []
    total_evaluated = 0
    total_empty = 0

    for s in sample_combined.iter_rows(named=True):
        tk = s["ticker"].strip().upper()
        seq = s["spell_seq"]
        s_date = s["start_date"]
        e_date = s["end_date"]
        dur = s.get("n_sessions") or s.get("duration_sessions", 1)

        total_evaluated += 1
        rec, telem = pool.query(tk, s_date, spell_id=f"EMPTY_{tk}_{seq}")

        is_empty = (rec is None or (not rec.get("cik") and not rec.get("share_class_figi")))
        if is_empty:
            total_empty += 1
            has_of = (tk in figi_set)
            cause = categorize_empty_cause(tk, s_date, dur, has_of)

            # Test symbol aliases if dot or warrant
            candidates = [c["candidate_alias"] for c in generate_alias_candidates(tk)]
            alias_resolved = False
            resolved_candidate = None
            if len(candidates) > 1:
                for cand in candidates[1:]:  # test non-original candidates
                    cand_rec, _ = pool.query(cand, s_date, spell_id=f"ALIAS_{cand}_{seq}")
                    if cand_rec and (cand_rec.get("cik") or cand_rec.get("share_class_figi")):
                        alias_resolved = True
                        resolved_candidate = cand
                        break

            empty_records.append({
                "ticker": tk,
                "spell_seq": seq,
                "start_date": s_date,
                "end_date": e_date,
                "duration_sessions": dur,
                "cause_category": cause,
                "has_openfigi_entry": has_of,
                "candidate_count": len(candidates),
                "alias_resolved": alias_resolved,
                "successful_alias": resolved_candidate,
            })

    df_empty = pl.DataFrame(empty_records)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_empty.write_parquet(OUT_PARQUET)
    logger.info("Saved empty response taxonomy parquet to %s (%d records)", OUT_PARQUET, df_empty.height)

    # Summarize causes
    cause_counts = df_empty.group_by("cause_category").agg(pl.len().alias("count")).sort("count", descending=True)
    alias_recoveries = df_empty.filter(pl.col("alias_resolved") == True).height

    md_content = f"""# Section 8: Massive Empty Response Taxonomy & Inactivity Policy

## 1. Core Architectural Policy: MASSIVE_EMPTY != SECURITY_INACTIVE
An empty or unmapped HTTP response from the Massive Point-In-Time API (`/v3/reference/tickers`) does **NOT** indicate that the security was inactive, non-existent, or illegitimate on that date.

### Fundamental Principle
The ticker-spell manifest (`spells.csv`) is derived directly from empirical, raw exchange trading snapshots. A ticker's presence in a daily snapshot proves that market activity or listing status occurred on that trading session.

A failure of the Massive reference database to locate an entity under a specific ticker-date key is primarily attributable to:
1. **Symbology Incompatibilities**: Differing exchange conventions for dual-class shares (`BRK.B` vs `BRK/B` vs `BRK B`), warrants (`.WS` vs `w` vs `/WS`), and units (`.U` vs `/UN`).
2. **Coverage Cutoffs**: Thinly traded OTC/Pink Sheet securities and pre-2008 microcap issues where point-in-time ticker metadata was never backfilled.
3. **Short-Lived Transient Tickers**: Spells lasting 1 to 3 sessions resulting from temporary ticker allocations, symbol testing, or exchange reorganizations.

**Mandatory Invariant**:
- A Massive empty response **MUST NEVER** cause a spell to be discarded, filtered, or marked as inactive.
- All empty-response spells must receive a deterministic `IdentityStatus.UNRESOLVED` identifier (`UNRESOLVED_<ticker>_<seq>_<date_hash>`) with `is_canonical = False`.
- These spells are preserved in the research universe under `UniverseStatus.QUARANTINE` until secondary historical sources (e.g. SEC full archives or physical CRSP feeds) resolve them.

---

## 2. Quantitative Failure Taxonomy

From an empirical sample of **{total_evaluated}** targeted spells, **{total_empty}** returned empty Massive responses.

| Failure Category | Description | Count | % of Empties |
| :--- | :--- | :--- | :--- |
"""
    for r in cause_counts.iter_rows(named=True):
        pct = (r["count"] / max(1, total_empty)) * 100
        md_content += f"| `{r['cause_category']}` | Structural / coverage failure mode | **{r['count']}** | {pct:.1f}% |\n"

    md_content += f"""
---

## 3. Alias Recovery Performance
- Total spells evaluated with symbol aliases: **{df_empty.filter(pl.col('candidate_count') > 1).height}**
- Successfully recovered via Candidate Alias Layer: **{alias_recoveries}** ({(alias_recoveries / max(1, df_empty.filter(pl.col('candidate_count') > 1).height) * 100):.1f}%)

### Empirical Examples of Empty Recovery
| Original Ticker | Recovered Candidate | Start Date | Category |
| :--- | :--- | :--- | :--- |
"""
    for r in df_empty.filter(pl.col("alias_resolved") == True).head(10).iter_rows(named=True):
        md_content += f"| `{r['ticker']}` | `{r['successful_alias']}` | {r['start_date']} | `{r['cause_category']}` |\n"

    md_content += """
---

## 4. Policy Rules for Resolver V2 & Production
1. **Zero-Deletion Rule**: No spell in `spells.csv` shall be deleted, dropped, or merged because of a Massive empty response.
2. **Explicit Diagnostic Provenance**: When an empty response is received, the resolver assigns `IdentityStatus.UNRESOLVED` and populates `decision_reason = 'MASSIVE_EMPTY_NO_REFERENCE_RECORD'`.
3. **Safe Quarantine**: In the universe filter, unresolved spells are assigned `UniverseStatus.QUARANTINE`. They are strictly excluded from downstream signal generation, but remain permanently auditable in the security master.
"""

    OUT_MD.write_text(md_content, encoding="utf-8")
    logger.info("Saved Massive empty analysis report to %s", OUT_MD)


if __name__ == "__main__":
    run_empty_analysis()
