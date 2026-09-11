"""
Stratified Sample Builder for Identity Resolver Experiment (v1)
===============================================================
Constructs a deterministic, stratified sample of exactly 350 spells from
data/identity/experiments/representative_date_manifest.parquet.

Categories covered:
- A: Ordinary Historical Equities (120 spells: 40 early, 40 mid, 40 modern)
- B: Known Ticker Reuse & Negative Controls (60 spells: 13 negative controls + 47 multi-spell long gaps)
- C: Dot-Notation / Weak Resolution / Empty Candidates (50 spells)
- D: Delisted / Acquired / Historical Securities (40 spells)
- E: Pre-2010 / Null-Type Candidates (35 spells)
- F: Non-Common Instruments: ETFs, Units, Warrants, Preferred (25 spells)
- G: Ultra-Short / Transient Spells (20 spells: 10 single-session, 10 2-5 sessions)

Total = 350 spells.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "identity" / "experiments" / "representative_date_manifest.parquet"
OUTPUT_DIR = REPO_ROOT / "data" / "identity" / "experiments" / "resolver_v1"
OUTPUT_PARQUET = OUTPUT_DIR / "sample_manifest.parquet"
OUTPUT_CSV = OUTPUT_DIR / "sample_manifest.csv"


def build_stratified_sample(manifest: pl.DataFrame) -> pl.DataFrame:
    selected_spells = []
    selected_keys = set()

    def add_spells(df_subset: pl.DataFrame, cat_code: str, cat_desc: str, limit: int):
        added = 0
        for r in df_subset.iter_rows(named=True):
            key = (r["ticker"], r["spell_seq"])
            if key not in selected_keys:
                selected_keys.add(key)
                r_dict = dict(r)
                r_dict["sampling_category"] = cat_code
                r_dict["category_description"] = cat_desc
                selected_spells.append(r_dict)
                added += 1
                if added >= limit:
                    break
        print(f"[{cat_code}] Added {added}/{limit} spells: {cat_desc}")

    # 1. Category B: Negative Controls (ACMR, AAC, MON, META, AAA) - 13 spells
    neg_tickers = ["ACMR", "AAC", "MON", "META", "AAA"]
    add_spells(
        manifest.filter(pl.col("ticker").is_in(neg_tickers)).sort(["ticker", "spell_seq"]),
        "B_TICKER_REUSE_NEG_CONTROL",
        "Dedicated negative controls that must not be merged",
        13
    )

    # 2. Category B: Ticker Reuse with Multi-Spell Long Gaps - 47 spells
    lg_tickers = [
        "AAAA", "AAAP", "BABA", "BLNK", "CAR", "SAVE", "ROOT", "DNA", "BOX", "COIN",
        "RIVN", "NKLA", "PLTR", "RBLX", "LCID", "HOOD", "PATH", "AVCT", "ARMH", "SMLL",
        "SPRO", "CYT", "OLED", "HKIT", "ADT", "EBC", "AGNT", "IOT", "AMRN", "ATVI"
    ]
    add_spells(
        manifest.filter(pl.col("ticker").is_in(lg_tickers)).sort(["ticker", "spell_seq"]),
        "B_TICKER_REUSE_MULTI_SPELL",
        "Empirical multi-spell tickers with long inter-spell dormant gaps",
        47
    )

    # 3. Category D: Delisted / Acquired / Bankrupt Equities - 40 spells
    delist_tickers = [
        "TWTR", "CELG", "FRC", "SIVB", "BBBY", "LEH", "BSC", "MER", "S", "RHT",
        "ALXN", "INFO", "DRE", "CTXS", "CERN", "XLNX", "MXIM", "VAR", "FLIR", "WFM",
        "YHOO", "BWA", "APC", "PXD", "HES", "ETFC", "TIF", "COV", "KSU", "CBOE", "ESV", "NE"
    ]
    add_spells(
        manifest.filter(pl.col("ticker").is_in(delist_tickers)).sort(["ticker", "spell_seq"]),
        "D_DELISTED_ACQUIRED",
        "Delisted, merged, or bankrupt historical equities",
        40
    )

    # 4. Category F: Non-Common Equity Instruments (ETFs, Units, Warrants, Preferred) - 25 spells
    non_cs_tickers = [
        "SPY", "QQQ", "IWM", "EEM", "VTI", "XLK", "XLE", "XLF", "XLV",
        "AAC.U", "AAC.WS", "AAB.WS", "BAC.PR", "C.PR", "PSTH.U", "PSTH.WS",
        "IPOA.U", "IPOA.WS", "TSM", "ASML", "AZN", "NVO", "BTI", "SHEL", "RIO"
    ]
    add_spells(
        manifest.filter(pl.col("ticker").is_in(non_cs_tickers)).sort(["ticker", "spell_seq"]),
        "F_NON_COMMON_INSTRUMENT",
        "Non-common instruments testing classification and exclusion filters",
        25
    )

    # 5. Category C: Dot-Notation / Weak Resolution / Empty Candidates - 50 spells
    dot_spells = manifest.filter(pl.col("ticker").str.contains(r"\."))
    add_spells(
        dot_spells.sort(["duration_sessions", "ticker"]),
        "C_DOT_NOTATION_WEAK_RESOLUTION",
        "Tickers with share class dot notation or historical snapshot dropouts",
        50
    )

    # 6. Category G: Ultra-Short Spells (1 session and 2–5 sessions) - 20 spells
    s1 = manifest.filter(pl.col("duration_sessions") == 1)
    add_spells(s1.sort("ticker"), "G_ULTRA_SHORT_SPELL", "Ultra-transient 1-session appearance spells", 10)
    s2_5 = manifest.filter((pl.col("duration_sessions") >= 2) & (pl.col("duration_sessions") <= 5))
    add_spells(s2_5.sort("ticker"), "G_SHORT_SPELL_2_5", "Short 2–5 session appearance spells", 10)

    # 7. Category E: Pre-2010 / Null-Type Candidates - 35 spells
    pre_2010 = manifest.filter(
        (pl.col("start_date") < "2008-01-01") &
        (pl.col("end_date") < "2010-01-01") &
        (pl.col("duration_sessions") > 100) &
        (~pl.col("ticker").str.contains(r"\."))
    )
    add_spells(
        pre_2010.sort(["start_date", "ticker"]),
        "E_PRE_2010_NULL_TYPE_CANDIDATE",
        "Historical pre-2010 equities testing null-type classification",
        35
    )

    # 8. Category A: Ordinary Historical Equities (120 spells)
    # A1: Early core (40 spells)
    bluechips_early = [
        "AAPL", "MSFT", "IBM", "JNJ", "GE", "PG", "XOM", "KO", "WMT", "PFE",
        "INTC", "CSCO", "ORCL", "HD", "MCD", "DIS", "NKE", "BA", "CAT", "MMM",
        "VZ", "T", "MRK", "AXP", "CVX", "GS", "HPQ", "UNH", "LOW", "FDX",
        "UPS", "BMY", "LMT", "DE", "GD", "SYK", "CL", "MDLZ", "COST", "MO", "USB"
    ]
    add_spells(
        manifest.filter(pl.col("ticker").is_in(bluechips_early) & (pl.col("representative_date") < "2016-01-01")).sort("ticker"),
        "A_ORDINARY_EARLY_CORE",
        "Established blue-chip operating companies with continuous early listings",
        40
    )

    # A2: Mid era (40 spells)
    mid_tickers = [
        "AMZN", "NFLX", "CRM", "V", "MA", "NOW", "ISRG", "REGN", "VRTX", "ILMN",
        "AVGO", "TXN", "QCOM", "ADBE", "INTU", "AMD", "NVDA", "TSLA", "PYPL", "SQ",
        "SHOP", "TEAM", "DOCU", "ZM", "OKTA", "TWLO", "MDB", "CRWD", "DDOG", "NET",
        "FSLY", "PINS", "SNAP", "PTON", "CHWY", "UBER", "LYFT", "DASH", "ABNB", "SNOW"
    ]
    add_spells(
        manifest.filter(pl.col("ticker").is_in(mid_tickers)).sort("ticker"),
        "A_ORDINARY_MID_ERA",
        "Established operating equities representing 2010–2019 mid-era listings",
        40
    )

    # A3: Modern era (40 spells)
    modern_spells = manifest.filter(
        (pl.col("start_date") >= "2020-01-01") &
        (pl.col("duration_sessions") >= 250) &
        (~pl.col("ticker").str.contains(r"\."))
    )
    add_spells(
        modern_spells.sort(["start_date", "ticker"]),
        "A_ORDINARY_MODERN_ERA",
        "Modern operating equities with active listings originating 2020–2026",
        40
    )

    df_sample = pl.DataFrame(selected_spells)
    print(f"\nTotal stratified sample size: {df_sample.height} spells across {df_sample['ticker'].n_unique()} unique tickers.")
    return df_sample


def main():
    print(f"Loading frozen manifest from {MANIFEST_PATH}...")
    manifest = pl.read_parquet(MANIFEST_PATH)
    print(f"Loaded {manifest.height} total spells.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_sample = build_stratified_sample(manifest)

    print(f"Saving sample manifest to {OUTPUT_PARQUET}...")
    df_sample.write_parquet(OUTPUT_PARQUET)
    print(f"Saving sample manifest to {OUTPUT_CSV}...")
    df_sample.write_csv(OUTPUT_CSV)

    print("\nBreakdown by sampling category:")
    print(df_sample["sampling_category"].value_counts().sort("sampling_category"))
    print("\nSample builder completed successfully!")


if __name__ == "__main__":
    main()
