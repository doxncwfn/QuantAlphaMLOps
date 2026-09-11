"""
V3 Deterministic Candidate Resolver & Universe Pipeline.
========================================================
Implements Sections 10-18, 22-23:
- Recomputes full dataset deterministically over all 43,757 spells.
- Uses master manifest from data/manifests/v3/massive_manifest.parquet.
- Applies 5-Tier Canonical Identity Hierarchy:
    Tier 1: Authoritative Massive FIGI (Canonical)
    Tier 2: Massive CIK + SEC Match + Distinctive Token Corroboration (Canonical)
    Tier 3: Provisional CIK Namespace (is_canonical=False)
    Tier 4: Massive Empty Result (Deterministic UNRESOLVED_*, is_canonical=False)
    Tier 5: Offline Pending Backfill (Deterministic UNRESOLVED_*, is_canonical=False)
- Generates candidate outputs under:
    data/identity/candidates/v3/
        security_master_candidate.parquet
        ticker_history_candidate.parquet
        identity_evidence_candidate.parquet
        identity_conflicts_candidate.parquet
        identity_aliases_candidate.parquet
    data/universe/candidates/v3/
        daily_universe_candidate.parquet
        availability_episodes_candidate.parquet
        expected_security_dates_candidate.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

from src.common.config import (
    CANDIDATES_IDENTITY_DIR,
    CANDIDATES_UNIVERSE_DIR,
    LOG_DIR,
    MANIFESTS_DIR,
    OPENFIGI_CACHE_PARQUET,
    SEC_CACHE_JSON,
    SPELLS_CSV_PATH,
    TRADING_SESSIONS_PATH,
)
from src.identity.resolver.model import (
    IdentityStatus,
    SecurityType,
    UniverseStatus,
    are_names_consistent,
    classify_universe_status,
    generate_symbol_aliases,
    make_deterministic_unresolved_id,
    make_provisional_cik_id,
    normalize_security_type,
)

MASSIVE_MANIFEST_PARQUET = MANIFESTS_DIR / "massive_manifest.parquet"
LOG_FILE = LOG_DIR / "v3_resolver.log"

# Output candidate files
SECURITY_MASTER_PARQUET = CANDIDATES_IDENTITY_DIR / "security_master_candidate.parquet"
TICKER_HISTORY_PARQUET = CANDIDATES_IDENTITY_DIR / "ticker_history_candidate.parquet"
IDENTITY_EVIDENCE_PARQUET = (
    CANDIDATES_IDENTITY_DIR / "identity_evidence_candidate.parquet"
)
IDENTITY_CONFLICTS_PARQUET = (
    CANDIDATES_IDENTITY_DIR / "identity_conflicts_candidate.parquet"
)
IDENTITY_ALIASES_PARQUET = (
    CANDIDATES_IDENTITY_DIR / "identity_aliases_candidate.parquet"
)

DAILY_UNIVERSE_PARQUET = CANDIDATES_UNIVERSE_DIR / "daily_universe_candidate.parquet"
AVAILABILITY_EPISODES_PARQUET = (
    CANDIDATES_UNIVERSE_DIR / "availability_episodes_candidate.parquet"
)
EXPECTED_SECURITY_DATES_PARQUET = (
    CANDIDATES_UNIVERSE_DIR / "expected_security_dates_candidate.parquet"
)
IDENTITY_MANIFEST_PARQUET = MANIFESTS_DIR / "identity_manifest.parquet"


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("v3_resolver")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class V3CandidateResolver:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        CANDIDATES_IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
        CANDIDATES_UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
        MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

        self.sec_map = self._load_sec_mapping()
        self.figi_map = self._load_openfigi_mapping()
        self.calendar_sessions = self._load_sessions()

    def _load_sessions(self) -> list[str]:
        df = pl.read_parquet(TRADING_SESSIONS_PATH)
        return df["session_date"].to_list()

    def _load_sec_mapping(self) -> dict[str, dict[str, Any]]:
        sec_map = {}
        if SEC_CACHE_JSON.exists():
            with open(SEC_CACHE_JSON, encoding="utf-8") as f:
                d = json.load(f)
            fields = d.get("fields", [])
            rows = d.get("data", [])
            for r in rows:
                tk = r[fields.index("ticker")]
                sec_map[tk.upper()] = {
                    "cik": str(r[fields.index("cik")]).zfill(10),
                    "name": r[fields.index("name")],
                    "exchange": r[fields.index("exchange")],
                }
        self.logger.info("Loaded %d SEC EDGAR mappings.", len(sec_map))
        return sec_map

    def _load_openfigi_mapping(self) -> dict[str, dict[str, Any]]:
        figi_map = {}
        if OPENFIGI_CACHE_PARQUET.exists():
            df_of = pl.read_parquet(OPENFIGI_CACHE_PARQUET)
            for r in df_of.iter_rows(named=True):
                figi_map[r["query_ticker"].upper()] = {
                    "share_class_figi": r.get("top_share_class_figi"),
                    "name": r.get("top_name"),
                    "sec_type": r.get("top_security_type"),
                }
        self.logger.info("Loaded %d OpenFIGI cache mappings.", len(figi_map))
        return figi_map

    def run_resolution(self) -> dict[str, Any]:
        self.logger.info("=" * 80)
        self.logger.info("STARTING V3 CANDIDATE RESOLUTION & UNIVERSE GENERATION")
        self.logger.info("=" * 80)
        t_start = time.time()

        if not MASSIVE_MANIFEST_PARQUET.exists():
            raise FileNotFoundError(
                f"Massive manifest not found at {MASSIVE_MANIFEST_PARQUET}. Run backfill engine first."
            )

        df_manifest = pl.read_parquet(MASSIVE_MANIFEST_PARQUET)
        self.logger.info("Loaded Massive manifest with %d rows.", df_manifest.height)

        # Deterministic resolution timestamp for reproducible bit-for-bit reruns
        resolution_ts = "2026-09-09T05:00:00+00:00"

        # Containers for candidate datasets
        ticker_history_records: list[dict[str, Any]] = []
        identity_evidence_records: list[dict[str, Any]] = []
        conflicts_records: list[dict[str, Any]] = []
        aliases_records: list[dict[str, Any]] = []
        seen_aliases: set[str] = set()

        # Grouping for security master
        security_groups: dict[str, list[dict[str, Any]]] = {}

        for row in df_manifest.iter_rows(named=True):
            ticker = row["ticker"].strip()
            clean_tk = ticker.upper()
            seq = row["spell_seq"]
            s_date = row["start_date"]
            e_date = row["end_date"]
            dur = row["duration_sessions"]
            rep_date = row["representative_date"]
            rep_method = row["representative_date_method"]
            spell_id = row["spell_id"]

            lookup_status = row["lookup_status"]
            m_cik = row["massive_cik"]
            m_figi = row["massive_figi"]
            m_comp = row["massive_composite_figi"]
            m_name = row["massive_name"]
            m_type = row["massive_type"]
            m_exch = row["massive_exchange"]
            m_act = row["massive_active"]
            drift_detected = row["drift_detected"]
            drift_details = row["drift_details"]

            # Reference data
            of_data = self.figi_map.get(clean_tk) or self.figi_map.get(ticker) or {}
            of_figi = of_data.get("share_class_figi")
            of_name = of_data.get("name")
            of_type = of_data.get("sec_type")

            sec_data = self.sec_map.get(clean_tk) or self.sec_map.get(ticker) or {}
            sec_cik = sec_data.get("cik")
            sec_name = sec_data.get("name")
            sec_exch = sec_data.get("exchange")

            # Aliases generation (INV_27: original never overwritten)
            aliases = generate_symbol_aliases(ticker)
            for a in aliases:
                key_alias = f"{ticker}:{a['candidate_symbol']}"
                if key_alias not in seen_aliases:
                    seen_aliases.add(key_alias)
                    aliases_records.append(
                        {
                            "original_ticker": ticker,
                            "candidate_symbol": a["candidate_symbol"],
                            "rule": a["rule"],
                            "confidence": a["confidence"],
                            "is_original": a["is_original"],
                        }
                    )

            # Record drift conflicts if detected
            if drift_detected:
                conflicts_records.append(
                    {
                        "ticker": ticker,
                        "spell_seq": seq,
                        "conflict_type": "WITHIN_SPELL_IDENTITY_DRIFT",
                        "severity": "HIGH",
                        "details": drift_details
                        or "Boundary divergence detected between Start, Midpoint, or End evidence",
                        "start_date": s_date,
                        "end_date": e_date,
                        "recorded_timestamp": resolution_ts,
                    }
                )

            # 1. Normalize Security Type
            norm_type = SecurityType.UNKNOWN
            if m_type:
                norm_type = normalize_security_type(m_type)
            elif of_type:
                norm_type = normalize_security_type(of_type)

            # 2. Classify Universe Status (INV_08: UNKNOWN never promoted to INCLUDE)
            univ_status = classify_universe_status(norm_type)

            # 3. 5-Tier Canonical Identity Hierarchy
            security_id = None
            is_canonical = False
            id_status = IdentityStatus.UNRESOLVED
            id_conf = "LOW"
            id_tier = "TIER_5_OFFLINE_PENDING"
            decision_reason = ""
            conflict_flag = False

            if lookup_status == "SUCCESS":
                if m_figi:
                    # Tier 1: Authoritative Massive FIGI
                    security_id = m_figi
                    is_canonical = True
                    id_tier = "TIER_1_MASSIVE_FIGI"
                    if of_figi and of_figi == m_figi:
                        id_status = IdentityStatus.CONFIRMED
                        id_conf = "HIGH"
                        decision_reason = "Massive PIT FIGI corroborated by OpenFIGI."
                    else:
                        id_status = IdentityStatus.PROBABLE
                        id_conf = "HIGH"
                        decision_reason = "Massive PIT FIGI authoritative."
                elif m_cik:
                    # Tier 2: Massive CIK + SEC Match + Token Overlap
                    corroborated = False
                    if sec_cik and sec_cik == m_cik:
                        if are_names_consistent(
                            m_name, of_name
                        ) or are_names_consistent(m_name, sec_name):
                            corroborated = True

                    if corroborated and of_figi:
                        security_id = of_figi
                        is_canonical = True
                        id_tier = "TIER_2_CIK_CORROBORATED_FIGI"
                        id_status = IdentityStatus.CONFIRMED
                        id_conf = "HIGH"
                        decision_reason = (
                            "Massive CIK corroborated by SEC EDGAR with OpenFIGI FIGI."
                        )
                    else:
                        # Tier 3: Provisional CIK Namespace (INV_06, INV_23: is_canonical=False)
                        security_id = make_provisional_cik_id(m_cik, ticker, s_date)
                        is_canonical = False
                        id_tier = "TIER_3_PROVISIONAL_CIK"
                        id_status = IdentityStatus.PROVISIONAL
                        id_conf = "MEDIUM"
                        if sec_cik and sec_cik != m_cik:
                            conflict_flag = True
                            decision_reason = f"Massive CIK {m_cik} diverges from SEC CIK {sec_cik} (Ticker Reuse). Provisional CIK assigned."
                            conflicts_records.append(
                                {
                                    "ticker": ticker,
                                    "spell_seq": seq,
                                    "conflict_type": "TICKER_REUSE_CIK_DIVERGENCE",
                                    "severity": "MEDIUM",
                                    "details": f"Massive CIK={m_cik} vs SEC CIK={sec_cik}",
                                    "start_date": s_date,
                                    "end_date": e_date,
                                    "recorded_timestamp": resolution_ts,
                                }
                            )
                        else:
                            decision_reason = f"Massive CIK {m_cik} without security-level FIGI. Provisional CIK assigned."
                else:
                    # Tier 4: Massive record returned but empty of FIGI and CIK
                    security_id = make_deterministic_unresolved_id(
                        ticker, seq, s_date, e_date
                    )
                    is_canonical = False
                    id_tier = "TIER_4_MASSIVE_EMPTY"
                    id_status = IdentityStatus.UNRESOLVED
                    id_conf = "LOW"
                    decision_reason = "Massive PIT response contained no CIK or FIGI."
                    univ_status = UniverseStatus.QUARANTINE

            elif lookup_status == "MASSIVE_EMPTY":
                security_id = make_deterministic_unresolved_id(
                    ticker, seq, s_date, e_date
                )
                is_canonical = False
                id_tier = "TIER_4_MASSIVE_EMPTY"
                id_status = IdentityStatus.UNRESOLVED
                id_conf = "LOW"
                decision_reason = "Massive PIT returned empty response."
                univ_status = UniverseStatus.QUARANTINE

            else:
                # Tier 5: Offline Pending Backfill
                security_id = make_deterministic_unresolved_id(
                    ticker, seq, s_date, e_date
                )
                is_canonical = False
                id_tier = "TIER_5_OFFLINE_PENDING"
                id_status = IdentityStatus.UNRESOLVED
                id_conf = "LOW"
                decision_reason = "MASSIVE_PIT_OFFLINE_PENDING"
                univ_status = UniverseStatus.QUARANTINE

            # Populate Ticker History candidate
            ticker_history_records.append(
                {
                    "ticker": ticker,
                    "spell_seq": seq,
                    "security_id": security_id,
                    "is_canonical": is_canonical,
                    "start_date": s_date,
                    "end_date": e_date,
                    "duration_sessions": dur,
                    "representative_date": rep_date,
                    "confidence": id_conf,
                    "exchange": m_exch or sec_exch or "UNKNOWN",
                    "security_type": norm_type,
                    "research_universe_status": univ_status,
                }
            )

            # Populate Identity Evidence candidate (INV_19, INV_30)
            identity_evidence_records.append(
                {
                    "spell_id": spell_id,
                    "ticker": ticker,
                    "spell_seq": seq,
                    "representative_date": rep_date,
                    "lookup_status": lookup_status,
                    "cache_status": row["cache_status"],
                    "massive_cik": m_cik,
                    "massive_figi": m_figi,
                    "massive_name": m_name,
                    "massive_type": m_type,
                    "massive_exchange": m_exch,
                    "openfigi_figi": of_figi,
                    "openfigi_name": of_name,
                    "sec_cik": sec_cik,
                    "sec_name": sec_name,
                    "selected_security_id": security_id,
                    "is_canonical": is_canonical,
                    "identity_status": id_status,
                    "identity_confidence": id_conf,
                    "resolution_tier": id_tier,
                    "decision_reason": decision_reason,
                    "resolution_timestamp": resolution_ts,
                }
            )

            # Append to security master groups
            if security_id not in security_groups:
                security_groups[security_id] = []
            security_groups[security_id].append(
                {
                    "ticker": ticker,
                    "start_date": s_date,
                    "end_date": e_date,
                    "is_canonical": is_canonical,
                    "share_class_figi": (m_figi or of_figi) if is_canonical else None,
                    "composite_figi": m_comp if is_canonical else None,
                    "cik": (m_cik or sec_cik)
                    if is_canonical
                    else (m_cik if id_tier == "TIER_3_PROVISIONAL_CIK" else None),
                    "security_name": m_name
                    or (sec_name or of_name if is_canonical else None)
                    or f"SECURITY_{clean_tk}",
                    "security_type": norm_type,
                    "primary_exchange": m_exch or sec_exch or "UNKNOWN",
                    "confidence": id_conf,
                    "status": id_status,
                    "tier": id_tier,
                    "universe_status": univ_status,
                }
            )

        # Build Security Master records
        security_master_records = []
        for sec_id, group in security_groups.items():
            first_obs = min(r["start_date"] for r in group)
            last_obs = max(r["end_date"] for r in group)
            distinct_tickers = sorted({r["ticker"] for r in group})
            is_can = group[0]["is_canonical"]
            sec_type = group[0]["security_type"]
            u_status = group[0]["universe_status"]

            # Most complete metadata row
            best_row = sorted(
                group,
                key=lambda x: (1 if x["share_class_figi"] else 0, 1 if x["cik"] else 0),
                reverse=True,
            )[0]

            security_master_records.append(
                {
                    "security_id": sec_id,
                    "is_canonical": is_can,
                    "share_class_figi": best_row["share_class_figi"],
                    "composite_figi": best_row["composite_figi"],
                    "cik": best_row["cik"],
                    "security_name": best_row["security_name"],
                    "security_type": sec_type,
                    "primary_exchange": best_row["primary_exchange"],
                    "country": "US",
                    "identity_confidence": best_row["confidence"],
                    "identity_status": best_row["status"],
                    "first_observed_date": first_obs,
                    "last_observed_date": last_obs,
                    "n_spells": len(group),
                    "n_tickers": len(distinct_tickers),
                    "identity_source_hierarchy": best_row["tier"],
                    "research_universe_status": u_status,
                }
            )

        # Save Candidate Identity Datasets with strict deterministic sorting
        self.logger.info("Writing candidate identity datasets...")
        df_sec = pl.DataFrame(security_master_records).sort(["security_id"])
        df_sec.write_parquet(SECURITY_MASTER_PARQUET)
        self.logger.info(
            "Wrote %s (%d securities).", SECURITY_MASTER_PARQUET, df_sec.height
        )

        df_th = pl.DataFrame(ticker_history_records).sort(["ticker", "spell_seq"])
        df_th.write_parquet(TICKER_HISTORY_PARQUET)
        self.logger.info("Wrote %s (%d spells).", TICKER_HISTORY_PARQUET, df_th.height)

        df_ev = pl.DataFrame(identity_evidence_records).sort(["ticker", "spell_seq"])
        df_ev.write_parquet(IDENTITY_EVIDENCE_PARQUET)
        self.logger.info(
            "Wrote %s (%d evidence records).", IDENTITY_EVIDENCE_PARQUET, df_ev.height
        )

        df_conf = (
            pl.DataFrame(conflicts_records)
            if conflicts_records
            else pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "spell_seq": pl.Int64,
                    "conflict_type": pl.Utf8,
                    "severity": pl.Utf8,
                    "details": pl.Utf8,
                    "start_date": pl.Utf8,
                    "end_date": pl.Utf8,
                    "recorded_timestamp": pl.Utf8,
                }
            )
        )
        df_conf = df_conf.sort(["ticker", "spell_seq", "conflict_type"])
        df_conf.write_parquet(IDENTITY_CONFLICTS_PARQUET)
        self.logger.info(
            "Wrote %s (%d conflict records).",
            IDENTITY_CONFLICTS_PARQUET,
            df_conf.height,
        )

        df_ali = pl.DataFrame(aliases_records).sort(
            ["original_ticker", "candidate_symbol"]
        )
        df_ali.write_parquet(IDENTITY_ALIASES_PARQUET)
        self.logger.info(
            "Wrote %s (%d alias records).", IDENTITY_ALIASES_PARQUET, df_ali.height
        )

        # Build Candidate Universe Datasets
        self.logger.info("Building candidate universe datasets...")
        self._build_universe_candidates(df_th, df_sec)

        # Write Identity Manifest
        self._write_identity_manifest(df_sec, df_th, df_ev, df_conf, resolution_ts)

        t_elapsed = time.time() - t_start
        self.logger.info("=" * 80)
        self.logger.info("CANDIDATE RESOLUTION COMPLETED IN %.2f SECONDS.", t_elapsed)
        self.logger.info("=" * 80)

        return {
            "securities_count": df_sec.height,
            "spells_count": df_th.height,
            "canonical_securities": df_sec.filter(
                pl.col("is_canonical") == True
            ).height,
            "provisional_securities": df_sec.filter(
                pl.col("security_id").str.starts_with("PROVISIONAL_CIK_")
            ).height,
            "unresolved_securities": df_sec.filter(
                pl.col("security_id").str.starts_with("UNRESOLVED_")
            ).height,
            "conflicts_count": df_conf.height,
            "elapsed_sec": t_elapsed,
        }

    def _build_universe_candidates(self, df_th: pl.DataFrame, df_sec: pl.DataFrame):
        """Constructs availability episodes, expected security dates, and daily universe candidates."""
        # 1. Availability Episodes
        # Group contiguous spells per security_id
        session_to_idx = {d: i for i, d in enumerate(self.calendar_sessions)}
        episodes = []
        episode_idx = 0

        sec_map = {r["security_id"]: r for r in df_sec.iter_rows(named=True)}

        for sec_id, group in df_th.partition_by("security_id", as_dict=True).items():
            sid = sec_id[0] if isinstance(sec_id, tuple) else sec_id
            s_meta = sec_map.get(sid, {})
            spells_sorted = group.sort("start_date").iter_rows(named=True)

            cur_ep = None
            for s in spells_sorted:
                s_date = s["start_date"]
                e_date = s["end_date"]
                s_idx = session_to_idx.get(s_date, 0)
                e_idx = session_to_idx.get(e_date, 0)
                dur = max(1, e_idx - s_idx + 1)

                if cur_ep is None:
                    episode_idx += 1
                    cur_ep = {
                        "episode_id": f"EP_{episode_idx:06d}",
                        "security_id": sid,
                        "is_canonical": s["is_canonical"],
                        "research_universe_status": s["research_universe_status"],
                        "ticker": s["ticker"],
                        "start_date": s_date,
                        "end_date": e_date,
                        "n_sessions": dur,
                        "first_observed_date": s_date,
                        "last_observed_date": e_date,
                        "n_observed_sessions": s["duration_sessions"],
                        "n_inferred_sessions": dur,
                    }
                else:
                    # Check gap between previous end and current start
                    prev_e_idx = session_to_idx.get(cur_ep["end_date"], 0)
                    gap = s_idx - prev_e_idx - 1
                    if gap <= 5:  # Tolerate small gap within episode
                        cur_ep["end_date"] = e_date
                        cur_ep["last_observed_date"] = e_date
                        cur_ep["n_observed_sessions"] += s["duration_sessions"]
                        tot_dur = max(
                            1, e_idx - session_to_idx.get(cur_ep["start_date"], 0) + 1
                        )
                        cur_ep["n_sessions"] = tot_dur
                        cur_ep["n_inferred_sessions"] = tot_dur
                    else:
                        episodes.append(cur_ep)
                        episode_idx += 1
                        cur_ep = {
                            "episode_id": f"EP_{episode_idx:06d}",
                            "security_id": sid,
                            "is_canonical": s["is_canonical"],
                            "research_universe_status": s["research_universe_status"],
                            "ticker": s["ticker"],
                            "start_date": s_date,
                            "end_date": e_date,
                            "n_sessions": dur,
                            "first_observed_date": s_date,
                            "last_observed_date": e_date,
                            "n_observed_sessions": s["duration_sessions"],
                            "n_inferred_sessions": dur,
                        }
            if cur_ep is not None:
                episodes.append(cur_ep)

        df_ep = pl.DataFrame(episodes)
        df_ep.write_parquet(AVAILABILITY_EPISODES_PARQUET)
        self.logger.info(
            "Wrote %s (%d episodes).", AVAILABILITY_EPISODES_PARQUET, df_ep.height
        )

        # 2. Daily Universe Candidate (canonical common-stock securities only)
        # We sample trading dates or build daily active membership
        canonical_spells = df_th.filter(
            (pl.col("is_canonical") == True)
            & (pl.col("research_universe_status") == UniverseStatus.INCLUDE)
        )
        self.logger.info(
            "Canonical included spells for universe: %d.", canonical_spells.height
        )

        daily_rows = []
        for s in canonical_spells.iter_rows(named=True):
            s_idx = session_to_idx.get(s["start_date"])
            e_idx = session_to_idx.get(s["end_date"])
            if s_idx is not None and e_idx is not None:
                # Add bounding dates and midpoint for compact representation
                mid_idx = (s_idx + e_idx) // 2
                key_dates = {
                    self.calendar_sessions[s_idx],
                    self.calendar_sessions[mid_idx],
                    self.calendar_sessions[e_idx],
                }
                for d in sorted(key_dates):
                    daily_rows.append(
                        {
                            "date": d,
                            "security_id": s["security_id"],
                            "ticker": s["ticker"],
                            "is_canonical": True,
                            "research_universe_status": UniverseStatus.INCLUDE,
                        }
                    )

        df_daily = (
            pl.DataFrame(daily_rows)
            .unique(subset=["date", "security_id"])
            .sort(["date", "security_id"])
        )
        df_daily.write_parquet(DAILY_UNIVERSE_PARQUET)
        self.logger.info(
            "Wrote %s (%d daily universe records).",
            DAILY_UNIVERSE_PARQUET,
            df_daily.height,
        )

        # 3. Expected Security Dates Candidate
        exp_rows = []
        for s in df_th.iter_rows(named=True):
            exp_rows.append(
                {
                    "security_id": s["security_id"],
                    "date": s["start_date"],
                    "ticker": s["ticker"],
                    "spell_seq": s["spell_seq"],
                    "is_observed": True,
                    "is_canonical": s["is_canonical"],
                }
            )
            if s["start_date"] != s["end_date"]:
                exp_rows.append(
                    {
                        "security_id": s["security_id"],
                        "date": s["end_date"],
                        "ticker": s["ticker"],
                        "spell_seq": s["spell_seq"],
                        "is_observed": True,
                        "is_canonical": s["is_canonical"],
                    }
                )

        df_exp = (
            pl.DataFrame(exp_rows)
            .unique(subset=["security_id", "date", "ticker"])
            .sort(["security_id", "date", "ticker"])
        )
        df_exp.write_parquet(EXPECTED_SECURITY_DATES_PARQUET)
        self.logger.info(
            "Wrote %s (%d expected security dates records).",
            EXPECTED_SECURITY_DATES_PARQUET,
            df_exp.height,
        )

    def _write_identity_manifest(
        self,
        df_sec: pl.DataFrame,
        df_th: pl.DataFrame,
        df_ev: pl.DataFrame,
        df_conf: pl.DataFrame,
        ts: str,
    ):
        spells_bytes = SPELLS_CSV_PATH.read_bytes()
        spells_hash = hashlib.sha256(spells_bytes).hexdigest()

        manifest_rows = [
            {
                "pipeline_stage": "V3_CANDIDATE_RESOLVER",
                "resolver_version": "3.0.0",
                "spells_sha256": spells_hash,
                "total_spells": df_th.height,
                "unique_securities": df_sec.height,
                "canonical_securities": df_sec.filter(
                    pl.col("is_canonical") == True
                ).height,
                "provisional_securities": df_sec.filter(
                    pl.col("security_id").str.starts_with("PROVISIONAL_CIK_")
                ).height,
                "unresolved_securities": df_sec.filter(
                    pl.col("security_id").str.starts_with("UNRESOLVED_")
                ).height,
                "evidence_rows": df_ev.height,
                "conflicts_count": df_conf.height,
                "resolution_timestamp": ts,
            }
        ]
        df_m = pl.DataFrame(manifest_rows)
        df_m.write_parquet(IDENTITY_MANIFEST_PARQUET)
        self.logger.info("Saved identity manifest to %s", IDENTITY_MANIFEST_PARQUET)


def main():
    logger = setup_logger()
    resolver = V3CandidateResolver(logger=logger)
    resolver.run_resolution()


if __name__ == "__main__":
    main()
