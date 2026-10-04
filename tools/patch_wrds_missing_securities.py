#!/usr/bin/env python3
"""
Patch missing trading data for Russell 1000 constituents in annual WRDS files.

This script resolves 0% coverage anomalies in data/WRDS/<year>.parquet:
1. Category 1 (Missing from WRDS, present in data/US_history.parquet):
   Extracts missing trading history from US_history.parquet (1925-2024),
   casts columns to the exact target schema of <year>.parquet,
   formats dates into "YYYY-MM-DD" strings, sets TICKER to the constituent list
   symbol (e.g. HUB.B, BF.B, BRK.B, VIA.B), preserves native TSYMBOL, and appends.

2. Category 2 (Present in WRDS under root or renamed ticker):
   Updates TICKER in-place on matching PERMNO rows to match the constituent list
   (e.g., PERMNO 89731 -> LEN.B/LENB, PERMNO 85945 -> HEI.A/HEIA,
   PERMNO 14030 -> CWENA, PERMNO 23505 -> UHALB, PERMNO 15980 -> UAC/C,
   PERMNO 32791 -> WFT, PERMNO 89130 -> LMG.A, PERMNO 84042 -> UAG,
   PERMNO 78916 -> ACT, PERMNO 89070 -> ZMH, PERMNO 89179 -> ANTM,
   PERMNO 22840 -> SLE).

Ensures:
- Exact schema and dtype preservation per annual file.
- Zero duplicate (PERMNO, date) pairs.
- Proper sorting by (PERMNO, date).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("patch_wrds")

# Category 1: Extract from US_history.parquet and insert
# Format: ticker -> (permno, [list_years])
CATEGORY_1_SPECS: list[dict[str, Any]] = [
    {"ticker": "HUB.B", "permno": 32942, "years": list(range(2001, 2016))},
    {"ticker": "BF.B", "permno": 29946, "years": [y for y in range(2001, 2023) if y not in (2019,)]},
    {"ticker": "BFB", "permno": 29946, "years": [2019, 2023]},
    {"ticker": "BF.A", "permno": 29938, "years": [2016, 2017, 2018, 2020, 2021, 2022]},
    {"ticker": "BFA", "permno": 29938, "years": [2019, 2023]},
    {"ticker": "BRK.B", "permno": 83443, "years": [2010, 2011, 2012, 2013, 2014, 2017, 2018, 2020, 2021, 2022]},
    {"ticker": "BRKB", "permno": 83443, "years": [2015, 2016, 2019, 2023]},
    {"ticker": "JW.A", "permno": 82924, "years": [y for y in range(2002, 2021) if y != 2019]},
    {"ticker": "JWA", "permno": 82924, "years": [2019]},
    {"ticker": "FCE.A", "permno": 31974, "years": [y for y in range(2002, 2019) if y != 2016]},
    {"ticker": "FCEA", "permno": 31974, "years": [2016]},
    {"ticker": "LGF.A", "permno": 86408, "years": [2017, 2018, 2020]},
    {"ticker": "LGFA", "permno": 86408, "years": [2019]},
    {"ticker": "LGF.B", "permno": 16495, "years": [2017, 2018, 2020]},
    {"ticker": "LGFB", "permno": 16495, "years": [2019]},
    # Viacom: 2001-2004 (PERMNO 76226), 2005 (both 76226 H1 and 91063 H2), 2006-2011 (PERMNO 91063)
    {"ticker": "VIA.B", "permno": 76226, "years": [2001, 2002, 2003, 2004]},
    {"ticker": "VIA.B", "permno": 76226, "years": [2005], "max_date": 20051231},
    {"ticker": "VIA.B", "permno": 91063, "years": [2005], "min_date": 20060101},
    {"ticker": "VIA.B", "permno": 91063, "years": list(range(2006, 2012))},
    {"ticker": "AGR.A", "permno": 88917, "years": [2001]},
    {"ticker": "AGR.B", "permno": 89400, "years": [2002, 2003, 2004]},
    {"ticker": "COC.B", "permno": 87029, "years": [2001]},
    {"ticker": "FSL.B", "permno": 90435, "years": [2005, 2006]},
    {"ticker": "NMG.A", "permno": 87281, "years": [2001, 2002, 2003, 2004, 2005]},
    {"ticker": "NWS.A", "permno": 90441, "years": [2005, 2006, 2007, 2008]},
    {"ticker": "TAP.A", "permno": 89346, "years": [2002]},
    {"ticker": "TAP.B", "permno": 89495, "years": [2003]},
    {"ticker": "IDT.C", "permno": 83275, "years": [2003]},
    {"ticker": "ADLAC", "permno": 10564, "years": [2000, 2001]},
    {"ticker": "ETFCD", "permno": 83862, "years": [2010]},
    {"ticker": "IMCLE", "permno": 77103, "years": [2003]},
    {"ticker": "NCBCE", "permno": 56275, "years": [2001]},
    {"ticker": "WCOME", "permno": 11042, "years": [2002]},
]

# Category 2: Update TICKER in-place on existing rows in data/WRDS/<year>.parquet
# Format: (year, permno, target_ticker)
CATEGORY_2_UPDATES: list[dict[str, Any]] = [
    # Lennar Class B
    {"year": 2016, "permno": 89731, "ticker": "LEN.B"},
    {"year": 2017, "permno": 89731, "ticker": "LEN.B"},
    {"year": 2018, "permno": 89731, "ticker": "LEN.B"},
    {"year": 2019, "permno": 89731, "ticker": "LENB"},
    {"year": 2020, "permno": 89731, "ticker": "LEN.B"},
    {"year": 2021, "permno": 89731, "ticker": "LEN.B"},
    {"year": 2022, "permno": 89731, "ticker": "LEN.B"},
    {"year": 2023, "permno": 89731, "ticker": "LENB"},
    # Heico Class A
    {"year": 2017, "permno": 85945, "ticker": "HEI.A"},
    {"year": 2018, "permno": 85945, "ticker": "HEI.A"},
    {"year": 2019, "permno": 85945, "ticker": "HEIA"},
    {"year": 2020, "permno": 85945, "ticker": "HEI.A"},
    {"year": 2021, "permno": 85945, "ticker": "HEI.A"},
    {"year": 2022, "permno": 85945, "ticker": "HEI.A"},
    {"year": 2023, "permno": 85945, "ticker": "HEIA"},
    # Clearway Energy Class A
    {"year": 2023, "permno": 14030, "ticker": "CWENA"},
    # U-Haul Series N
    {"year": 2023, "permno": 23505, "ticker": "UHALB"},
    # Under Armour Class C
    {"year": 2016, "permno": 15980, "ticker": "UAC/C"},
    # Weatherford International (null TICKER in CRSP)
    {"year": 2016, "permno": 32791, "ticker": "WFT"},
    {"year": 2017, "permno": 32791, "ticker": "WFT"},
    {"year": 2018, "permno": 32791, "ticker": "WFT"},
    # Liberty Media Group Class A
    {"year": 2001, "permno": 89130, "ticker": "LMG.A"},
    # United Auto Group (renamed to Penske PAG)
    {"year": 2007, "permno": 84042, "ticker": "UAG"},
    # Allergan / Actavis (ACT)
    {"year": 2015, "permno": 78916, "ticker": "ACT"},
    # Zimmer Biomet / Zimmer Holdings (ZMH)
    {"year": 2015, "permno": 89070, "ticker": "ZMH"},
    # Elevance Health / Anthem (ANTM)
    {"year": 2022, "permno": 89179, "ticker": "ANTM"},
    # Hillshire Brands / Sara Lee (SLE)
    {"year": 2012, "permno": 22840, "ticker": "SLE"},
]

DATE_COLS = {"date", "NAMEENDT", "DCLRDT", "DLPDT", "NEXTDT", "PAYDT", "RCRDDT", "SHRENDDT"}


def transform_us_history_rows(
    df_raw: pl.DataFrame, target_schema: dict[str, pl.DataType], constituent_ticker: str
) -> pl.DataFrame:
    """Transform raw rows from US_history.parquet to match target annual file schema."""
    exprs = []
    for col, dtype in target_schema.items():
        if col == "TICKER":
            exprs.append(pl.lit(constituent_ticker, dtype=dtype).alias("TICKER"))
        elif col in DATE_COLS:
            if dtype == pl.String:
                exprs.append(
                    pl.col(col)
                    .cast(pl.String)
                    .str.strptime(pl.Date, "%Y%m%d", strict=False)
                    .dt.to_string("%Y-%m-%d")
                    .alias(col)
                )
            else:
                exprs.append(pl.col(col).cast(dtype, strict=False).alias(col))
        else:
            exprs.append(pl.col(col).cast(dtype, strict=False).alias(col))
    return df_raw.select(exprs)


def main() -> None:
    t_start = time.perf_counter()
    project_root = Path(__file__).resolve().parent.parent
    wrds_dir = project_root / "data" / "WRDS"
    us_history_path = project_root / "data" / "US_history.parquet"

    if not us_history_path.exists():
        raise FileNotFoundError(f"US history file not found: {us_history_path}")

    # 1. Preload US_history for all required PERMNOs
    all_req_permnos = sorted({spec["permno"] for spec in CATEGORY_1_SPECS})
    logger.info("Scanning %s for %d candidate PERMNOs...", us_history_path.name, len(all_req_permnos))
    df_us_master = (
        pl.scan_parquet(us_history_path)
        .filter(pl.col("PERMNO").is_in(all_req_permnos))
        .collect()
    )
    logger.info("Loaded %d candidate rows from US history.", len(df_us_master))

    # Group B updates by year: year -> list of update dicts
    updates_by_year: dict[int, list[dict[str, Any]]] = {
        y: [] for y in range(2000, 2025)
    }
    for u in CATEGORY_2_UPDATES:
        updates_by_year[u["year"]].append(u)

    # 2. Process each annual file from 2000 to 2024
    total_inserted_rows = 0
    total_updated_rows = 0
    modified_years = []

    for year in range(2000, 2025):
        year_file = wrds_dir / f"{year}.parquet"
        if not year_file.exists():
            continue

        target_schema = pl.read_parquet_schema(year_file)
        df_year = pl.read_parquet(year_file)
        init_rows = len(df_year)

        # Get date range of target file (as YYYYMMDD integers for US_history lookup)
        min_date_str = df_year["date"].min()
        max_date_str = df_year["date"].max()
        min_date_int = int(min_date_str.replace("-", ""))
        max_date_int = int(max_date_str.replace("-", ""))

        # A. Apply Category 2 Updates (in-place TICKER update)
        yr_updates = updates_by_year[year]
        yr_updated_count = 0
        if yr_updates:
            for upd in yr_updates:
                p = upd["permno"]
                tk = upd["ticker"]
                mask = pl.col("PERMNO") == p
                matching_count = df_year.filter(mask).shape[0]
                if matching_count > 0:
                    df_year = df_year.with_columns(
                        pl.when(mask).then(pl.lit(tk)).otherwise(pl.col("TICKER")).alias("TICKER")
                    )
                    yr_updated_count += matching_count
                    logger.info(
                        "[%d] Updated TICKER -> '%s' for PERMNO %d (%d rows)",
                        year,
                        tk,
                        p,
                        matching_count,
                    )
            total_updated_rows += yr_updated_count

        # B. Apply Category 1 Insertions
        yr_to_insert = [spec for spec in CATEGORY_1_SPECS if year in spec["years"]]
        new_frames = []

        for spec in yr_to_insert:
            p = spec["permno"]
            tk = spec["ticker"]
            low_date = max(min_date_int, spec.get("min_date", min_date_int))
            high_date = min(max_date_int, spec.get("max_date", max_date_int))

            sub_us = df_us_master.filter(
                (pl.col("PERMNO") == p)
                & (pl.col("date") >= low_date)
                & (pl.col("date") <= high_date)
            )

            if len(sub_us) == 0:
                logger.warning("[%d] No rows found in US history for %s (PERMNO %d) in [%d, %d]", year, tk, p, low_date, high_date)
                continue

            sub_transformed = transform_us_history_rows(sub_us, target_schema, tk)
            new_frames.append(sub_transformed)
            logger.info(
                "[%d] Extracted '%s' (PERMNO %d): %d rows (dates %s to %s)",
                year,
                tk,
                p,
                len(sub_transformed),
                sub_transformed["date"].min(),
                sub_transformed["date"].max(),
            )

        if not new_frames and yr_updated_count == 0:
            continue

        # Combine, sort, and deduplicate
        if new_frames:
            combined_new = pl.concat(new_frames)
            yr_inserted_count = len(combined_new)
            total_inserted_rows += yr_inserted_count
            df_year = pl.concat([df_year, combined_new])
        else:
            yr_inserted_count = 0

        # Enforce unique (PERMNO, date)
        dup_count = df_year.select(["PERMNO", "date"]).is_duplicated().sum()
        if dup_count > 0:
            logger.warning("[%d] Found %d duplicate (PERMNO, date) rows, deduplicating...", year, dup_count)
            df_year = df_year.unique(subset=["PERMNO", "date"], keep="first")

        # Sort by PERMNO, date
        df_year = df_year.sort(["PERMNO", "date"])

        # Write back to <year>.parquet
        df_year.write_parquet(year_file, compression="snappy")
        modified_years.append(year)
        logger.info(
            "✓ [%d] Saved %s: %d -> %d rows (+%d inserted, %d updated)",
            year,
            year_file.name,
            init_rows,
            len(df_year),
            yr_inserted_count,
            yr_updated_count,
        )

    logger.info("=" * 70)
    logger.info(
        "WRDS PATCH COMPLETED: %d modified years, %d rows inserted, %d rows updated in-place (%.2fs)",
        len(modified_years),
        total_inserted_rows,
        total_updated_rows,
        time.perf_counter() - t_start,
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
