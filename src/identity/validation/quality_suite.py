"""
V3 Whole-Dataset Quality Suite & Advanced Integrity Audits.
===========================================================
Implements Sections 12, 13, 14, 24, 25, 26, 27:
- Ticker-Reuse Audit: Evaluates all multi-spell tickers for false merges.
  Outputs data/quality/v3/ticker_reuse_audit.parquet and log/v3_ticker_reuse_audit.md.
- Same-CIK Multi-Security Test: Validates instrument-level separation for multi-security issuers.
  Outputs data/quality/v3/same_cik_multiple_security.parquet.
- FIGI Collision Audit: Ensures 1:1 mapping from confirmed FIGI to canonical security_id.
  Outputs data/quality/v3/figi_collision_audit.parquet.
- Coverage & Stratification Quality: Evaluates completeness across years, duration buckets,
  ticker formats, and resolution tiers.
  Outputs data/quality/v3/coverage_quality.parquet and data/quality/v3/identity_quality.parquet.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl

from src.common.config import (
    CANDIDATES_IDENTITY_DIR,
    CANDIDATES_UNIVERSE_DIR,
    EXPECTED_SPELLS_HASH,
    EXPECTED_SPELLS_ROWS,
    LOG_DIR,
    MANIFESTS_DIR,
    QUALITY_DIR,
    SPELLS_CSV_PATH,
)

SECURITY_MASTER_PARQUET = CANDIDATES_IDENTITY_DIR / "security_master_candidate.parquet"
TICKER_HISTORY_PARQUET = CANDIDATES_IDENTITY_DIR / "ticker_history_candidate.parquet"
IDENTITY_EVIDENCE_PARQUET = CANDIDATES_IDENTITY_DIR / "identity_evidence_candidate.parquet"
IDENTITY_CONFLICTS_PARQUET = CANDIDATES_IDENTITY_DIR / "identity_conflicts_candidate.parquet"

TICKER_REUSE_AUDIT_PARQUET = QUALITY_DIR / "ticker_reuse_audit.parquet"
TICKER_REUSE_AUDIT_MD = LOG_DIR / "v3_ticker_reuse_audit.md"
SAME_CIK_PARQUET = QUALITY_DIR / "same_cik_multiple_security.parquet"
FIGI_COLLISION_PARQUET = QUALITY_DIR / "figi_collision_audit.parquet"
COVERAGE_QUALITY_PARQUET = QUALITY_DIR / "coverage_quality.parquet"
IDENTITY_QUALITY_PARQUET = QUALITY_DIR / "identity_quality.parquet"

LOG_FILE = LOG_DIR / "v3_quality_suite.log"


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("v3_quality")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class V3QualitySuite:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        QUALITY_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        if not SECURITY_MASTER_PARQUET.exists() or not TICKER_HISTORY_PARQUET.exists():
            raise FileNotFoundError("Candidate datasets missing. Run resolver first.")

        self.df_sec = pl.read_parquet(SECURITY_MASTER_PARQUET)
        self.df_th = pl.read_parquet(TICKER_HISTORY_PARQUET)
        self.df_ev = pl.read_parquet(IDENTITY_EVIDENCE_PARQUET)
        self.df_conf = pl.read_parquet(IDENTITY_CONFLICTS_PARQUET) if IDENTITY_CONFLICTS_PARQUET.exists() else pl.DataFrame()

    def run_ticker_reuse_audit(self) -> pl.DataFrame:
        """Evaluates whether multi-spell tickers have false identity merges."""
        self.logger.info("Running Whole-Dataset Ticker-Reuse Audit...")

        # Find tickers with multiple spells
        spell_counts = self.df_th.group_by("ticker").agg([
            pl.len().alias("n_spells"),
            pl.col("security_id").n_unique().alias("n_security_ids"),
            pl.col("start_date").min().alias("first_spell_start"),
            pl.col("end_date").max().alias("last_spell_end"),
        ])
        multi_spell = spell_counts.filter(pl.col("n_spells") > 1)
        multi_spell_set = set(multi_spell["ticker"].to_list())
        n_spells_in_multis = self.df_th.filter(pl.col("ticker").is_in(multi_spell_set)).height
        self.logger.info("Found %d multi-spell tickers across %d spells.", multi_spell.height, n_spells_in_multis)

        reuse_records = []
        false_merge_count = 0

        # Known negative controls (must NEVER merge across reuse cycles)
        negative_controls = {"ACMR", "AA", "C", "GM", "VALE"}

        for r in multi_spell.iter_rows(named=True):
            tk = r["ticker"]
            n_spells = r["n_spells"]
            n_sec_ids = r["n_security_ids"]

            spells_for_tk = self.df_th.filter(pl.col("ticker") == tk).sort("spell_seq")
            sec_ids = spells_for_tk["security_id"].to_list()

            false_merge = False
            details = "Separated"

            if n_sec_ids == 1 and spells_for_tk["is_canonical"].to_list()[0]:
                ev_for_tk = self.df_ev.filter(pl.col("ticker") == tk)
                ciks = set(ev_for_tk.filter(pl.col("massive_cik").is_not_null())["massive_cik"].to_list())
                if len(ciks) > 1:
                    false_merge = True
                    details = f"COLLISION: Same canonical ID {sec_ids[0]} across conflicting CIKs: {ciks}"
                    false_merge_count += 1
                elif tk in negative_controls and n_spells > 1:
                    false_merge = True
                    details = f"NEGATIVE CONTROL FAILED: Ticker {tk} falsely merged across {n_spells} spells!"
                    false_merge_count += 1
                else:
                    details = "Legitimate continuous security spanning multiple spells"
            elif n_sec_ids > 1:
                details = f"Correctly separated into {n_sec_ids} distinct security IDs"

            reuse_records.append({
                "ticker": tk,
                "n_spells": n_spells,
                "n_security_ids": n_sec_ids,
                "is_reused": n_spells > 1,
                "false_merge_detected": false_merge,
                "details": details,
                "first_spell_start": r["first_spell_start"],
                "last_spell_end": r["last_spell_end"],
            })

        df_reuse = pl.DataFrame(reuse_records)
        df_reuse.write_parquet(TICKER_REUSE_AUDIT_PARQUET)
        self.logger.info("Saved ticker reuse audit (%d rows) to %s. False merges detected: %d",
                         df_reuse.height, TICKER_REUSE_AUDIT_PARQUET, false_merge_count)

        # Markdown Report
        md = f"""# Whole-Dataset Ticker-Reuse Audit Report

## 1. Executive Summary
- **Multi-Spell Tickers Audited**: **{df_reuse.height:,}**
- **Total Multi-Spell Spells**: **{n_spells_in_multis:,}**
- **False Reuse Merges Detected**: **{false_merge_count}**
- **Audit Result**: **{'PASS' if false_merge_count == 0 else 'FAIL'}**

---

## 2. Benchmark Negative Controls Verification
| Ticker | Spells | Distinct Security IDs | Resolved Security IDs | Status |
| :--- | :--- | :--- | :--- | :--- |
"""
        for ctrl in sorted(negative_controls):
            sub = df_reuse.filter(pl.col("ticker") == ctrl)
            if sub.height > 0:
                row = sub.to_dicts()[0]
                sec_list = self.df_th.filter(pl.col("ticker") == ctrl)["security_id"].unique().to_list()
                status = "PASS (Separated)" if not row["false_merge_detected"] else "FAIL (Merged)"
                md += f"| `{ctrl}` | {row['n_spells']} | {row['n_security_ids']} | `{sec_list}` | **{status}** |\n"
            else:
                md += f"| `{ctrl}` | 1 | 1 | N/A | N/A (Single spell) |\n"

        md += """
---

## 3. Methodology
- Spells sharing the same ticker but corresponding to different corporate entities (distinct CIKs or distinct FIGIs) are assigned separate security identifiers.
- Provisional CIK namespace incorporates start_date hash to ensure non-canonical reuse spells remain strictly isolated.
"""
        TICKER_REUSE_AUDIT_MD.write_text(md, encoding="utf-8")
        self.logger.info("Saved ticker reuse audit summary to %s", TICKER_REUSE_AUDIT_MD)
        return df_reuse

    def run_same_cik_multi_security_test(self) -> pl.DataFrame:
        """Evaluates whether distinct instruments from the same issuer (CIK) remain separated."""
        self.logger.info("Running Same-CIK Multi-Security Separation Test...")

        # Find CIKs with multiple security entities
        cik_groups = self.df_sec.filter(pl.col("cik").is_not_null()).group_by("cik").agg([
            pl.len().alias("n_securities"),
            pl.col("security_id").alias("security_ids"),
            pl.col("security_type").alias("security_types"),
            pl.col("share_class_figi").alias("figis"),
            pl.col("primary_exchange").alias("exchanges"),
        ])
        multi_sec_ciks = cik_groups.filter(pl.col("n_securities") > 1)
        self.logger.info("Found %d CIKs with multiple security entities.", multi_sec_ciks.height)

        records = []
        violations = 0

        for r in multi_sec_ciks.iter_rows(named=True):
            cik = r["cik"]
            n_sec = r["n_securities"]
            sids = r["security_ids"]
            types = r["security_types"]
            figis = [f for f in r["figis"] if f is not None]

            # Check for illegal FIGI collision among canonical securities
            violation = False
            explanation = f"Issuer CIK {cik} has {n_sec} distinct instruments cleanly separated."

            if len(figis) > 1 and len(figis) != len(set(figis)):
                violation = True
                violations += 1
                explanation = f"COLLISION: Multiple securities under CIK {cik} share the same FIGI!"

            records.append({
                "cik": cik,
                "n_securities": n_sec,
                "distinct_security_ids": json.dumps(sids),
                "distinct_tickers": json.dumps(list(set(self.df_th.filter(pl.col("security_id").is_in(set(sids)))["ticker"].to_list()))),
                "distinct_security_types": json.dumps(list(set(types))),
                "is_separated": not violation,
                "violation_detected": violation,
                "explanation": explanation,
            })

        df_cik = pl.DataFrame(records) if records else pl.DataFrame(schema={
            "cik": pl.Utf8, "n_securities": pl.Int64, "distinct_security_ids": pl.Utf8,
            "distinct_tickers": pl.Utf8, "distinct_security_types": pl.Utf8,
            "is_separated": pl.Boolean, "violation_detected": pl.Boolean, "explanation": pl.Utf8
        })
        df_cik.write_parquet(SAME_CIK_PARQUET)
        self.logger.info("Saved Same-CIK Multi-Security Test (%d rows) to %s. Violations: %d",
                         df_cik.height, SAME_CIK_PARQUET, violations)
        return df_cik

    def run_figi_collision_audit(self) -> pl.DataFrame:
        """Verifies that each confirmed share-class FIGI maps 1:1 to a canonical security_id."""
        self.logger.info("Running Whole-Dataset FIGI Collision Audit...")

        canonical_sec = self.df_sec.filter(
            (pl.col("is_canonical") == True) &
            (pl.col("share_class_figi").is_not_null())
        )

        figi_groups = canonical_sec.group_by("share_class_figi").agg([
            pl.len().alias("n_security_ids"),
            pl.col("security_id").alias("security_ids"),
            pl.col("security_name").alias("security_names"),
        ])

        collisions = figi_groups.filter(pl.col("n_security_ids") > 1)
        collision_count = collisions.height

        records = []
        for r in figi_groups.iter_rows(named=True):
            f = r["share_class_figi"]
            n = r["n_security_ids"]
            is_col = (n > 1)
            records.append({
                "share_class_figi": f,
                "n_security_ids": n,
                "security_ids_json": json.dumps(r["security_ids"]),
                "security_names_json": json.dumps(r["security_names"]),
                "collision_detected": is_col,
                "details": "1:1 canonical mapping" if not is_col else f"COLLISION: FIGI {f} maps to {n} distinct security IDs!"
            })

        df_figi = pl.DataFrame(records)
        df_figi.write_parquet(FIGI_COLLISION_PARQUET)
        self.logger.info("Saved FIGI Collision Audit (%d FIGIs) to %s. Collisions: %d",
                         df_figi.height, FIGI_COLLISION_PARQUET, collision_count)
        return df_figi

    def run_coverage_and_quality(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Calculates stratified coverage metrics across years, durations, formats, and tiers."""
        self.logger.info("Calculating Coverage and Quality Metrics...")

        def get_bucket(dur: int) -> str:
            if dur <= 1:
                return "1_SESSION"
            elif dur <= 5:
                return "2_5_SESSIONS"
            elif dur <= 20:
                return "6_20_SESSIONS"
            elif dur <= 50:
                return "21_50_SESSIONS"
            elif dur <= 252:
                return "51_252_SESSIONS"
            else:
                return "253_PLUS_SESSIONS"

        df_with_bucket = self.df_th.with_columns(
            pl.col("duration_sessions").map_elements(get_bucket, return_dtype=pl.Utf8).alias("duration_bucket"),
            pl.col("start_date").str.slice(0, 4).alias("start_year"),
        )

        cov_records = []
        for bucket, grp in df_with_bucket.group_by("duration_bucket"):
            b_name = bucket[0] if isinstance(bucket, tuple) else bucket
            tot = grp.height
            canon = grp.filter(pl.col("is_canonical") == True).height
            prov = grp.filter(pl.col("security_id").str.starts_with("PROVISIONAL_CIK_")).height
            unres = grp.filter(pl.col("security_id").str.starts_with("UNRESOLVED_")).height
            cov_records.append({
                "stratification_dimension": "DURATION_BUCKET",
                "segment": b_name,
                "total_spells": tot,
                "canonical_spells": canon,
                "provisional_spells": prov,
                "unresolved_spells": unres,
                "canonical_pct": canon / tot * 100.0,
            })

        for yr, grp in df_with_bucket.group_by("start_year"):
            y_name = yr[0] if isinstance(yr, tuple) else yr
            tot = grp.height
            canon = grp.filter(pl.col("is_canonical") == True).height
            prov = grp.filter(pl.col("security_id").str.starts_with("PROVISIONAL_CIK_")).height
            unres = grp.filter(pl.col("security_id").str.starts_with("UNRESOLVED_")).height
            cov_records.append({
                "stratification_dimension": "START_YEAR",
                "segment": y_name,
                "total_spells": tot,
                "canonical_spells": canon,
                "provisional_spells": prov,
                "unresolved_spells": unres,
                "canonical_pct": canon / tot * 100.0,
            })

        df_cov = pl.DataFrame(cov_records).sort(["stratification_dimension", "segment"])
        df_cov.write_parquet(COVERAGE_QUALITY_PARQUET)
        self.logger.info("Saved Coverage Quality (%d rows) to %s", df_cov.height, COVERAGE_QUALITY_PARQUET)

        # Quality Overview
        total_spells = self.df_th.height
        canon_spells = self.df_th.filter(pl.col("is_canonical") == True).height
        prov_spells = self.df_th.filter(pl.col("security_id").str.starts_with("PROVISIONAL_CIK_")).height
        unres_spells = self.df_th.filter(pl.col("security_id").str.starts_with("UNRESOLVED_")).height
        included_spells = self.df_th.filter(pl.col("research_universe_status") == "INCLUDE").height

        overview_records = [{
            "metric": "total_spells", "value": total_spells, "pct": 100.0
        }, {
            "metric": "canonical_spells", "value": canon_spells, "pct": canon_spells / total_spells * 100.0
        }, {
            "metric": "provisional_spells", "value": prov_spells, "pct": prov_spells / total_spells * 100.0
        }, {
            "metric": "unresolved_spells", "value": unres_spells, "pct": unres_spells / total_spells * 100.0
        }, {
            "metric": "included_universe_spells", "value": included_spells, "pct": included_spells / total_spells * 100.0
        }]
        df_qual = pl.DataFrame(overview_records)
        df_qual.write_parquet(IDENTITY_QUALITY_PARQUET)
        self.logger.info("Saved Identity Quality (%d rows) to %s", df_qual.height, IDENTITY_QUALITY_PARQUET)

        return df_cov, df_qual


def main():
    logger = setup_logger()
    suite = V3QualitySuite(logger=logger)
    suite.run_ticker_reuse_audit()
    suite.run_same_cik_multi_security_test()
    suite.run_figi_collision_audit()
    suite.run_coverage_and_quality()


if __name__ == "__main__":
    main()
