"""
Security Identity Model & Core Engine for Resolver V2.
======================================================
Defines strict identity evidence hierarchies, provisional non-canonical namespaces,
symbol-alias generation, and cross-source corroboration rules.

Key Architectural Guarantees:
1. Canonical Share-Class FIGI from Massive PIT is primary.
2. CIK is an ISSUER identifier, NOT a security identifier.
   CIK-derived fallbacks MUST be explicitly flagged as PROVISIONAL and non-canonical:
   PROVISIONAL_CIK_<CIK>_<normalized_ticker>_<scope_hash>
3. Contemporary OpenFIGI queries are NEVER accepted blindly; they require SEC CIK
   and entity token corroboration.
4. UNKNOWN security type is NEVER silently promoted to INCLUDE.
5. Deterministic unresolved IDs guarantee zero accidental false merging.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# -----------------------------------------------------------------------------
# Security Type & Research Universe Taxonomy
# -----------------------------------------------------------------------------
class SecurityType:
    COMMON_STOCK = "COMMON_STOCK"
    ADR = "ADR"
    PREFERRED = "PREFERRED"
    ETF = "ETF"
    ETN = "ETN"
    UNIT = "UNIT"
    WARRANT = "WARRANT"
    RIGHT = "RIGHT"
    REIT = "REIT"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class UniverseStatus:
    INCLUDE = "INCLUDE"
    EXCLUDE = "EXCLUDE"
    QUARANTINE = "QUARANTINE"


class IdentityStatus:
    CONFIRMED = "CONFIRMED"
    PROBABLE = "PROBABLE"
    PROVISIONAL = "PROVISIONAL"
    UNRESOLVED = "UNRESOLVED"
    EXCLUDED = "EXCLUDED"


def normalize_security_type(raw_type: Optional[str]) -> str:
    """Normalizes vendor-specific security types into standardized taxonomy."""
    if not raw_type or not str(raw_type).strip():
        return SecurityType.UNKNOWN
    rt = str(raw_type).upper().strip()
    if rt in ("CS", "COMMON STOCK", "COMMON", "ORDINARY SHARES", "SHS", "COM"):
        return SecurityType.COMMON_STOCK
    if rt in ("ETF", "EXCHANGE TRADED FUND", "EXCHANGE-TRADED FUND", "ETP"):
        return SecurityType.ETF
    if rt in ("ETN", "EXCHANGE TRADED NOTE"):
        return SecurityType.ETN
    if rt in ("PF", "PREFERRED", "PREFERRED STOCK", "PREF"):
        return SecurityType.PREFERRED
    if rt in ("WT", "WARRANT", "WAR", "WS", "WARRANTS"):
        return SecurityType.WARRANT
    if rt in ("RT", "RIGHT", "RIGHTS", "RTS"):
        return SecurityType.RIGHT
    if rt in ("UNIT", "UNITS", "UT"):
        return SecurityType.UNIT
    if rt in ("ADR", "AMERICAN DEPOSITARY RECEIPT", "ADS"):
        return SecurityType.ADR
    if rt in ("REIT", "REAL ESTATE INVESTMENT TRUST"):
        return SecurityType.REIT
    return SecurityType.OTHER


def classify_universe_status(security_type: str) -> str:
    """
    Strict Research Universe Admission Policy:
    - US-listed Common Stocks only -> INCLUDE
    - Non-common instruments (ETF, ETN, Unit, Warrant, Preferred, Right, Other) -> EXCLUDE
    - ADR -> QUARANTINE (requires explicit research mandate)
    - UNKNOWN -> QUARANTINE (zero silent admission)
    """
    if security_type == SecurityType.COMMON_STOCK:
        return UniverseStatus.INCLUDE
    if security_type in (
        SecurityType.ETF,
        SecurityType.ETN,
        SecurityType.UNIT,
        SecurityType.WARRANT,
        SecurityType.PREFERRED,
        SecurityType.RIGHT,
        SecurityType.OTHER,
    ):
        return UniverseStatus.EXCLUDE
    if security_type == SecurityType.ADR:
        return UniverseStatus.QUARANTINE
    return UniverseStatus.QUARANTINE


# -----------------------------------------------------------------------------
# Identifier Generators
# -----------------------------------------------------------------------------
def make_provisional_cik_id(cik: str, ticker: str, start_date: str) -> str:
    """
    Generates an explicitly non-canonical provisional identifier for CIK-only cases.
    Guarantees that issuer CIK is never confused with a share-class security ID.
    """
    clean_cik = str(cik).zfill(10)
    clean_tk = ticker.strip().upper()
    raw = f"{clean_cik}_{clean_tk}_{start_date}".encode("utf-8")
    scope_hash = hashlib.sha256(raw).hexdigest()[:8].upper()
    return f"PROVISIONAL_CIK_{clean_cik}_{clean_tk}_{scope_hash}"


def make_deterministic_unresolved_id(ticker: str, spell_seq: int, start_date: str, end_date: Optional[str] = None) -> str:
    """Generates deterministic isolated ID for unresolved cases to prevent false merges."""
    clean_tk = ticker.strip()
    raw = f"{clean_tk}_{spell_seq}_{start_date}_{end_date or ''}".encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()[:10].upper()
    return f"UNRESOLVED_{clean_tk}_{spell_seq}_{h}"


# -----------------------------------------------------------------------------
# Entity Token Extraction & Corporate Name Matching
# -----------------------------------------------------------------------------
def extract_entity_tokens(name: Optional[str]) -> Set[str]:
    """Extracts distinctive meaningful entity tokens, filtering out noise and stopwords."""
    if not name:
        return set()
    cleaned = re.sub(r"[^A-Z0-9\s]", " ", str(name).upper())
    tokens = set(cleaned.split())
    stopwords = {
        "INC", "INCORPORATED", "CORP", "CORPORATION", "LTD", "LIMITED",
        "CO", "COMPANY", "COMPANIES", "CLASS", "CL", "A", "B", "C", "D",
        "ORD", "ORDINARY", "SHS", "SHARE", "SHARES", "COMMON", "STOCK",
        "STK", "HLDGS", "HOLDING", "HOLDINGS", "GRP", "GROUP", "THE",
        "OF", "AND", "DE", "NV", "PLC", "LP", "LLC", "ETF", "TRUST",
        "SPON", "ADR", "ADS", "FD", "FUND", "CAPITAL", "GLOBAL", "US", "USA",
        "COM", "FINANCIAL", "MANAGEMENT", "SERVICES", "SYSTEMS", "INVESTMENT"
    }
    return {t for t in tokens if t not in stopwords and len(t) > 1}


def are_names_consistent(name1: Optional[str], name2: Optional[str]) -> bool:
    """Checks whether two entity names share distinctive corporate identity tokens."""
    if not name1 or not name2:
        return False
    t1 = extract_entity_tokens(name1)
    t2 = extract_entity_tokens(name2)
    if not t1 or not t2:
        return False

    # Check numbered SPAC/fund generations (e.g. III vs II or none)
    roman = {"II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"}
    r1 = {t for t in t1 if t in roman}
    r2 = {t for t in t2 if t in roman}
    if r1 != r2:
        return False

    common = t1.intersection(t2)
    generic = {"ACQUISITION", "ACQUISTION", "VENTURES", "PARTNERS", "ENERGY", "HEALTHCARE", "MEDIA"}
    non_generic = common - generic
    if len(non_generic) >= 1:
        return True
    if len(common) >= 2:
        return True
    return False


# -----------------------------------------------------------------------------
# Symbol-Alias Candidate Layer
# -----------------------------------------------------------------------------
def generate_symbol_aliases(ticker: str) -> List[Dict[str, Any]]:
    """
    Generates candidate normalized aliases without mutating the original ticker string.
    Preserves rule, confidence, and validation status.
    """
    raw_tk = ticker.strip()
    orig = raw_tk.upper()
    candidates = []

    # 1. Identity (original literal, preserving exact case)
    candidates.append({
        "candidate_symbol": raw_tk,
        "rule": "ORIGINAL_LITERAL",
        "confidence": 1.0,
        "is_original": True
    })

    # Uppercase normalized alias if original had lowercase characters
    if raw_tk != orig:
        candidates.append({
            "candidate_symbol": orig,
            "rule": "UPPERCASE_NORMALIZED",
            "confidence": 0.95,
            "is_original": False
        })

    # 2. Dot notation conversions (.A, .B -> A, B or /A, /B)
    if "." in orig:
        parts = orig.split(".")
        if len(parts) == 2:
            base, ext = parts[0], parts[1]
            # Nasdaq 5-letter style: CMCS.A -> CMCSA
            if len(ext) == 1 and ext.isalpha():
                candidates.append({
                    "candidate_symbol": f"{base}{ext}",
                    "rule": "NASDAQ_CLASS_CONCAT",
                    "confidence": 0.85,
                    "is_original": False
                })
            # NYSE slash style: BRK.A -> BRK/A
            candidates.append({
                "candidate_symbol": f"{base}/{ext}",
                "rule": "NYSE_SLASH_CLASS",
                "confidence": 0.80,
                "is_original": False
            })
            # Bloomberg space style: BRK.A -> BRK A
            candidates.append({
                "candidate_symbol": f"{base} {ext}",
                "rule": "BLOOMBERG_SPACE_CLASS",
                "confidence": 0.75,
                "is_original": False
            })

    # 3. Lowercase / suffix notation (e.g. pA for preferred, w for warrants)
    if re.search(r"p[A-Z]$", ticker):
        # Preferred share notation
        base = re.sub(r"p[A-Z]$", "", ticker)
        pref = ticker[-1].upper()
        candidates.append({
            "candidate_symbol": f"{base} PR{pref}",
            "rule": "PREFERRED_STANDARD_ALIAS",
            "confidence": 0.80,
            "is_original": False
        })

    return candidates
