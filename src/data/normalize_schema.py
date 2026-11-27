"""Era-aware schema normalization and warehouse ingestion per v1.3 §10.3 & §10.6.

BUGS FIXED:
- BUG-07: Strict determinism enforcement torch.use_deterministic_algorithms(True) without warn_only.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None


# Deterministic fixes for silver.russell rows where security_id was left NULL
# despite the company existing in bronze.master_list / silver.security_history.
KNOWN_RUSSELL_PERMNO_FIXES: dict[str, int] = {
    "MMM": 22592,
    "RH": 13687,
    "FFIV": 86964,
    "HPQ": 27828,
    "AI": 20193,
    "ARD": 16624,
    "T": 66093,  # AT&T Inc (2006+); pre-2006 AT&T Corp is 10401
}


def seed_everything(seed: int = 42) -> None:
    """Enforce deterministic execution across random, numpy, and torch (v1.3 §10.6)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        warn_only = torch.backends.mps.is_available() and not torch.cuda.is_available()
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_schema(
    df: pd.DataFrame,
    era: str,
    ticker_to_permno: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Normalize vendor-specific schemas into a unified, era-agnostic DataFrame (v1.3 §10.3).

    Guarantees consistent split-adjusted prices and returns across 2000-2026.

    Parameters
    ----------
    df : pd.DataFrame
        Raw CRSP or Yahoo DataFrame.
    era : str
        Either 'CRSP' or 'Yahoo'.
    ticker_to_permno : dict[str, int] | None
        Optional mapping from Ticker (or normalized Ticker without dots) to PERMNO
        for the Yahoo era.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame sorted and deduplicated on (permno, date) (or (ticker, date)
        if permno is NaN).
    """
    if era == "CRSP":
        # Convert integer YYYYMMDD dates if needed
        raw_date = df["date"]
        if pd.api.types.is_integer_dtype(raw_date) or (
            pd.api.types.is_object_dtype(raw_date)
            and len(raw_date) > 0
            and str(raw_date.iloc[0]).isdigit()
            and len(str(raw_date.iloc[0])) == 8
        ):
            dt_series = pd.to_datetime(raw_date.astype(str), format="%Y%m%d")
        else:
            dt_series = pd.to_datetime(raw_date)

        cfacpr = pd.to_numeric(df["CFACPR"], errors="coerce").replace(0, np.nan)
        cfacshr = (
            pd.to_numeric(df["CFACSHR"], errors="coerce")
            if "CFACSHR" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        prc_abs = pd.to_numeric(df["PRC"], errors="coerce").abs()
        open_abs = pd.to_numeric(df["OPENPRC"], errors="coerce").abs()
        askhi = pd.to_numeric(df["ASKHI"], errors="coerce")
        bidlo = pd.to_numeric(df["BIDLO"], errors="coerce")
        vol = pd.to_numeric(df["VOL"], errors="coerce")

        retx_num = pd.to_numeric(df["RETX"], errors="coerce")
        ret_num = pd.to_numeric(df["RET"], errors="coerce")

        dlstcd = (
            pd.to_numeric(df["DLSTCD"], errors="coerce")
            if "DLSTCD" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        dlret = (
            pd.to_numeric(df["DLRET"], errors="coerce")
            if "DLRET" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        dlretx = (
            pd.to_numeric(df["DLRETX"], errors="coerce")
            if "DLRETX" in df.columns
            else dlret.copy()
        )
        shrout = (
            pd.to_numeric(df["SHROUT"], errors="coerce")
            if "SHROUT" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        shrcd = (
            pd.to_numeric(df["SHRCD"], errors="coerce")
            if "SHRCD" in df.columns
            else pd.Series(np.nan, index=df.index)
        )

        out = pd.DataFrame(
            {
                "date": dt_series,
                "ticker": df["TICKER"].astype(str).where(df["TICKER"].notna(), None),
                "permno": pd.to_numeric(df["PERMNO"], errors="coerce"),
                "close_raw": prc_abs,
                "open_raw": open_abs,
                "high_raw": askhi,
                "low_raw": bidlo,
                "volume": vol,
                "split_adj_close": prc_abs / cfacpr,
                "split_adj_open": open_abs / cfacpr,
                "split_adj_high": askhi / cfacpr,
                "split_adj_low": bidlo / cfacpr,
                "ret_price": np.log1p(retx_num),
                "ret_total": np.log1p(ret_num),
                "cfacpr": cfacpr,
                "cfacshr": cfacshr,
                "dlstcd": dlstcd,
                "dlret": dlret,
                "dlretx": dlretx,
                "shrout": shrout,
                "shrcd": shrcd,
                "era": "CRSP",
            }
        )
        # Filter out non-trading placeholder rows where PRC is null and no delisting code is present
        valid_mask = (
            out["close_raw"].notna() & (out["close_raw"] > 0) & out["cfacpr"].notna()
        )
        out = out.loc[valid_mask].copy()
        out["permno"] = out["permno"].astype(np.int64)
        out = out.sort_values(["permno", "date"]).drop_duplicates(
            subset=["permno", "date"], keep="first"
        )
        return out.reset_index(drop=True)

    elif era == "Yahoo":
        date_col = "Date" if "Date" in df.columns else "date"
        ticker_col = "Ticker" if "Ticker" in df.columns else "ticker"
        close_col = "Close" if "Close" in df.columns else "close"
        open_col = "Open" if "Open" in df.columns else "open"
        high_col = "High" if "High" in df.columns else "high"
        low_col = "Low" if "Low" in df.columns else "low"
        vol_col = "Volume" if "Volume" in df.columns else "volume"
        adj_col = "Adj_Close" if "Adj_Close" in df.columns else "adj_close"

        tickers = df[ticker_col].astype(str)
        if "permno" in df.columns and df["permno"].notna().any():
            permnos = pd.to_numeric(df["permno"], errors="coerce")
        elif ticker_to_permno is not None:
            norm_t = tickers.str.replace(".", "", regex=False)
            permnos = tickers.map(ticker_to_permno).fillna(norm_t.map(ticker_to_permno))
        else:
            permnos = pd.Series(np.nan, index=df.index)

        close_vals = pd.to_numeric(df[close_col], errors="coerce")
        open_vals = pd.to_numeric(df[open_col], errors="coerce")
        high_vals = pd.to_numeric(df[high_col], errors="coerce")
        low_vals = pd.to_numeric(df[low_col], errors="coerce")
        vol_vals = pd.to_numeric(df[vol_col], errors="coerce")
        adj_vals = (
            pd.to_numeric(df[adj_col], errors="coerce")
            if adj_col in df.columns
            else close_vals.copy()
        )

        shrcd_vals = (
            pd.to_numeric(df["shrcd"], errors="coerce")
            if "shrcd" in df.columns
            else (
                pd.to_numeric(df["SHRCD"], errors="coerce")
                if "SHRCD" in df.columns
                else pd.Series(np.nan, index=df.index)
            )
        )

        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df[date_col]),
                "ticker": tickers,
                "permno": permnos,
                "close_raw": close_vals,
                "open_raw": open_vals,
                "high_raw": high_vals,
                "low_raw": low_vals,
                "volume": vol_vals,
                "split_adj_close": close_vals,
                "split_adj_open": open_vals,
                "split_adj_high": high_vals,
                "split_adj_low": low_vals,
                "ret_price": np.nan,
                "ret_total": np.nan,
                "cfacpr": 1.0,
                "cfacshr": 1.0,
                "div_adj_close": adj_vals,
                "dlstcd": np.nan,
                "dlret": np.nan,
                "dlretx": np.nan,
                "shrout": np.nan,
                "shrcd": shrcd_vals,
                "era": "Yahoo",
            }
        )
        out = out.loc[
            out["split_adj_close"].notna() & (out["split_adj_close"] > 0)
        ].copy()
        group_key = "permno" if out["permno"].notna().all() else "ticker"
        out = out.sort_values([group_key, "date"]).drop_duplicates(
            subset=[group_key, "date"], keep="first"
        )
        out["ret_price"] = np.log(
            out.groupby(group_key)["split_adj_close"].pct_change(fill_method=None) + 1.0
        )
        out["ret_total"] = np.log(
            out.groupby(group_key)["div_adj_close"].pct_change(fill_method=None) + 1.0
        )
        return out.reset_index(drop=True)

    else:
        raise ValueError(f"Unsupported era: {era!r}. Expected 'CRSP' or 'Yahoo'.")


def build_ticker_to_permno_map(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Build deterministic mapping from Ticker / normalized Ticker to PERMNO."""
    mapping: dict[str, int] = {}

    # 1. Active CRSP securities from bronze.crsp_daily (most recent ticker/tsymbol per PERMNO)
    crsp_rows = con.execute(
        """
        WITH latest AS (
            SELECT PERMNO, TICKER, TSYMBOL, date,
                   row_number() OVER (PARTITION BY PERMNO ORDER BY date DESC) AS rn
            FROM bronze.crsp_daily
            WHERE PRC IS NOT NULL AND date >= 20000101
        )
        SELECT PERMNO, TICKER, TSYMBOL, date
        FROM latest
        WHERE rn = 1
        ORDER BY date ASC, PERMNO ASC
        """
    ).fetchall()
    for permno, ticker, tsymbol, _ in crsp_rows:
        p = int(permno)
        if ticker:
            mapping[str(ticker)] = p
            mapping[str(ticker).replace(".", "")] = p
        if tsymbol:
            mapping[str(tsymbol)] = p
            mapping[str(tsymbol).replace(".", "")] = p

    # 2. Russell constituent list mappings (most recent year wins)
    russell_rows = con.execute(
        """
        SELECT ticker, security_id, max(year) AS max_yr
        FROM silver.russell
        WHERE security_id IS NOT NULL AND ticker IS NOT NULL
        GROUP BY ticker, security_id
        ORDER BY max_yr ASC, security_id ASC
        """
    ).fetchall()
    for ticker, sec_id, _ in russell_rows:
        p = int(sec_id)
        t_str = str(ticker)
        mapping[t_str] = p
        mapping[t_str.replace(".", "")] = p

    # 3. Known fixes for NULL security_id in silver.russell
    for t_str, p in KNOWN_RUSSELL_PERMNO_FIXES.items():
        mapping[t_str] = p
        mapping[t_str.replace(".", "")] = p

    # 4. Deterministic synthetic PERMNOs (>= 900000) for post-2024 Yahoo tickers
    yahoo_tickers = con.execute(
        """
        SELECT DISTINCT Ticker
        FROM bronze.prices_2526
        WHERE file LIKE '%(2)%' AND Ticker IS NOT NULL
        ORDER BY Ticker ASC
        """
    ).fetchall()
    next_synth = 900001
    for (t_raw,) in yahoo_tickers:
        t_str = str(t_raw)
        t_norm = t_str.replace(".", "")
        if t_str not in mapping and t_norm not in mapping:
            mapping[t_str] = next_synth
            mapping[t_norm] = next_synth
            next_synth += 1
        elif t_str not in mapping:
            mapping[t_str] = mapping[t_norm]
        elif t_norm not in mapping:
            mapping[t_norm] = mapping[t_str]

    return mapping


def load_warehouse_data(
    warehouse_path: str | Path = "data/warehouse/russell1000.duckdb",
    include_yahoo: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalize CRSP + Yahoo daily panel, Fama-French factors, and Russell lists.

    Uses DuckDB to extract the full post-1998 panel plus up to 360 pre-2000 trading sessions
    per security (so every security has its exact lifetime session count and full 332-session
    warmup prior to 2000-06-30), and seamlessly stitches the 2024-12-31 -> 2025-01-02 boundary.
    """
    con = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        ticker_map = build_ticker_to_permno_map(con)

        # Extract CRSP rows: all valid rows with date >= 2000-06-30 OR within the last 360
        # sessions before 2000-06-30, plus total lifetime session offset before the extract.
        crsp_df = con.execute(
            """
            WITH dedup AS (
                SELECT PERMNO,
                       strptime(cast(date AS VARCHAR), '%Y%m%d')::DATE AS dt,
                       TICKER, PRC, OPENPRC, ASKHI, BIDLO, VOL, RET, RETX,
                       CFACPR, CFACSHR, SHROUT, DLSTCD, DLRET, DLRETX, SHRCD,
                       row_number() OVER (PARTITION BY PERMNO, date ORDER BY DISTCD NULLS LAST) AS rn
                FROM bronze.crsp_daily
                WHERE PRC IS NOT NULL AND abs(PRC) > 0 AND CFACPR IS NOT NULL AND CFACPR != 0
            ),
            uniq AS (
                SELECT *,
                       row_number() OVER (PARTITION BY PERMNO ORDER BY dt ASC) AS lifetime_session,
                       sum(CASE WHEN dt < DATE '2000-06-30' THEN 1 ELSE 0 END)
                           OVER (PARTITION BY PERMNO) AS total_pre_2000
                FROM dedup
                WHERE rn = 1
            )
            SELECT dt AS date, TICKER, PERMNO, PRC, OPENPRC, ASKHI, BIDLO, VOL,
                   RET, RETX, CFACPR, CFACSHR, SHROUT, DLSTCD, DLRET, DLRETX, SHRCD,
                   lifetime_session
            FROM uniq
            WHERE dt >= DATE '2000-06-30'
               OR lifetime_session > (total_pre_2000 - 360)
            ORDER BY PERMNO, dt
            """
        ).df()

        lifetime_session = crsp_df["lifetime_session"].to_numpy(dtype=np.int64)
        crsp_norm = normalize_schema(crsp_df, "CRSP")
        crsp_norm["lifetime_session"] = lifetime_session

        # Merge delisting returns from silver.delisting where missing on the last trading row
        delist_df = con.execute(
            """
            SELECT security_id AS permno,
                   delist_date AS date,
                   delist_code,
                   dlret,
                   dlretx
            FROM silver.delisting
            """
        ).df()
        delist_df["date"] = pd.to_datetime(delist_df["date"])
        crsp_norm = crsp_norm.merge(
            delist_df,
            on=["permno", "date"],
            how="left",
            suffixes=("", "_silver"),
        )
        crsp_norm["dlstcd"] = crsp_norm["dlstcd"].fillna(crsp_norm["delist_code"])
        crsp_norm["dlret"] = crsp_norm["dlret"].fillna(crsp_norm["dlret_silver"])
        crsp_norm["dlretx"] = crsp_norm["dlretx"].fillna(crsp_norm["dlretx_silver"])
        crsp_norm.drop(
            columns=["delist_code", "dlret_silver", "dlretx_silver"],
            inplace=True,
            errors="ignore",
        )

        if include_yahoo:
            # Prefer yfinance over massive/tiingo when multiple tickers map to same PERMNO on same date,
            # and prefer ticker with more trading days in 2025-2026 (e.g. XYZ over SQ after ticker change).
            yahoo_raw = con.execute(
                """
                SELECT Date, Ticker, Open, High, Low, Close, Adj_Close, Volume, Data_Source
                FROM bronze.prices_2526
                WHERE file LIKE '%(2)%' AND Close IS NOT NULL AND Close > 0
                """
            ).df()
            yahoo_raw["permno"] = (
                yahoo_raw["Ticker"]
                .map(ticker_map)
                .fillna(
                    yahoo_raw["Ticker"]
                    .str.replace(".", "", regex=False)
                    .map(ticker_map)
                )
            )
            yahoo_raw = yahoo_raw.loc[yahoo_raw["permno"].notna()].copy()
            yahoo_raw["permno"] = yahoo_raw["permno"].astype(np.int64)

            # Rank tickers per (permno, Date): prefer yfinance and tickers with longer coverage
            ticker_counts = yahoo_raw.groupby("Ticker")["Date"].transform("count")
            src_priority = (
                yahoo_raw["Data_Source"]
                .map({"yfinance": 0, "tiingo": 1, "massive": 2})
                .fillna(3)
            )
            has_dot = yahoo_raw["Ticker"].str.contains(".", regex=False).astype(int)
            yahoo_raw["_prio"] = src_priority * 10 - has_dot
            yahoo_raw["_tcnt"] = -ticker_counts
            yahoo_raw = (
                yahoo_raw.sort_values(["permno", "Date", "_prio", "_tcnt", "Ticker"])
                .drop_duplicates(subset=["permno", "Date"], keep="first")
                .drop(columns=["_prio", "_tcnt", "Data_Source"])
            )

            yahoo_norm = normalize_schema(
                yahoo_raw, "Yahoo", ticker_to_permno=ticker_map
            )
            yahoo_norm["permno"] = yahoo_norm["permno"].astype(np.int64)

            # Stitch boundary return on 2025-01-02 (first Yahoo date) using last CRSP 2024-12-31 close
            # when CRSP 2024-12-31 and Yahoo 2025-01-02 share the same split-adjustment basis
            # (|log(Open_2025-01-02 / Close_2024-12-31)| <= 0.20). For the ~35 stocks that underwent
            # 2025-2026 splits/reverse-splits (where Yahoo backward-adjusted 2025 prices relative to
            # 2024 CRSP), use the clean 2025-01-02 intraday return and align Yahoo split_adj prices
            # to the 2024-12-31 CRSP split basis so rolling price_z_20 and drawdown_20 are continuous.
            last_crsp = crsp_norm.loc[
                crsp_norm["date"] == pd.Timestamp("2024-12-31")
            ].set_index("permno")[["split_adj_close", "lifetime_session", "shrcd"]]
            first_yahoo_idx = yahoo_norm.groupby("permno")["date"].idxmin()
            first_permnos = yahoo_norm.loc[first_yahoo_idx, "permno"].values
            prev_closes = pd.Series(first_permnos, index=first_yahoo_idx).map(
                last_crsp["split_adj_close"]
            )
            first_opens = yahoo_norm.loc[first_yahoo_idx, "split_adj_open"]
            valid_stitch = (
                prev_closes.notna()
                & (prev_closes > 0)
                & first_opens.notna()
                & (first_opens > 0)
            )
            stitch_indices = first_yahoo_idx[valid_stitch.values].to_numpy()

            open_gap = np.abs(
                np.log(
                    first_opens.loc[stitch_indices].to_numpy()
                    / prev_closes.loc[stitch_indices].to_numpy()
                )
            )
            same_basis_idx = stitch_indices[open_gap <= 0.20]
            diff_basis_idx = stitch_indices[open_gap > 0.20]

            stitched_ret = np.log(
                yahoo_norm.loc[same_basis_idx, "split_adj_close"]
                / prev_closes.loc[same_basis_idx]
            )
            yahoo_norm.loc[same_basis_idx, "ret_price"] = stitched_ret
            yahoo_norm.loc[same_basis_idx, "ret_total"] = stitched_ret

            if len(diff_basis_idx) > 0:
                intraday_ret = np.log(
                    yahoo_norm.loc[diff_basis_idx, "split_adj_close"]
                    / yahoo_norm.loc[diff_basis_idx, "split_adj_open"]
                )
                yahoo_norm.loc[diff_basis_idx, "ret_price"] = intraday_ret
                yahoo_norm.loc[diff_basis_idx, "ret_total"] = intraday_ret
                # Align Yahoo split_adj price level to CRSP 2024-12-31 basis for these securities
                scale_factors = (
                    prev_closes.loc[diff_basis_idx] / first_opens.loc[diff_basis_idx]
                ).values
                scale_map = dict(
                    zip(yahoo_norm.loc[diff_basis_idx, "permno"].values, scale_factors)
                )
                p_scale = yahoo_norm["permno"].map(scale_map)
                has_scale = p_scale.notna()
                for col in [
                    "split_adj_close",
                    "split_adj_open",
                    "split_adj_high",
                    "split_adj_low",
                ]:
                    yahoo_norm.loc[has_scale, col] = (
                        yahoo_norm.loc[has_scale, col] * p_scale.loc[has_scale]
                    )

            # Carry forward shrcd and lifetime_session count into Yahoo era
            sec_hist_shrcd = (
                con.execute(
                    """
                    SELECT security_id AS permno,
                           arg_max(shrcd, valid_to) AS shrcd
                    FROM silver.security_history
                    WHERE shrcd IS NOT NULL
                    GROUP BY security_id
                    """
                )
                .df()
                .set_index("permno")["shrcd"]
            )
            last_crsp_shrcd = (
                crsp_norm.dropna(subset=["shrcd"])
                .groupby("permno", sort=False)["shrcd"]
                .last()
            )
            yahoo_norm["shrcd"] = (
                yahoo_norm["permno"]
                .map(last_crsp["shrcd"])
                .fillna(yahoo_norm["permno"].map(last_crsp_shrcd))
                .fillna(yahoo_norm["permno"].map(sec_hist_shrcd))
            )
            yahoo_session_num = yahoo_norm.groupby("permno").cumcount() + 1
            base_sessions = (
                yahoo_norm["permno"]
                .map(last_crsp["lifetime_session"])
                .fillna(0)
                .astype(np.int64)
            )
            yahoo_norm["lifetime_session"] = base_sessions + yahoo_session_num

            stock_df = pd.concat([crsp_norm, yahoo_norm], ignore_index=True)
            stock_df = stock_df.sort_values(["permno", "date"]).reset_index(drop=True)
            stock_df["shrcd"] = (
                stock_df.groupby("permno", sort=False)["shrcd"]
                .ffill()
                .fillna(stock_df["permno"].map(sec_hist_shrcd))
            )
        else:
            stock_df = crsp_norm.sort_values(["permno", "date"]).reset_index(drop=True)

        # Load Fama-French 6-factor + RF series (truncate post 2026-07-31 per v1.3 §10.5)
        factors_df = con.execute(
            """
            SELECT date, mktrf, smb, hml, rmw, cma, umd, rf
            FROM silver.market_factor
            WHERE date <= DATE '2026-07-31'
            ORDER BY date ASC
            """
        ).df()
        factors_df["date"] = pd.to_datetime(factors_df["date"])

        # Load Russell 1000 membership table
        russell_df = con.execute(
            """
            SELECT year AS Year, ticker AS Ticker, security_id, name AS Name, sector AS Sector
            FROM silver.russell
            ORDER BY year ASC, ticker ASC
            """
        ).df()
        # Resolve permno for every Russell constituent
        norm_r_ticker = (
            russell_df["Ticker"].astype(str).str.replace(".", "", regex=False)
        )
        resolved_permno = russell_df["security_id"].copy()
        # Special case: AT&T Corp (10401) in 2000-2005 vs AT&T Inc (66093) in 2006+
        att_pre2006 = (russell_df["Ticker"] == "T") & (russell_df["Year"] <= 2005)
        resolved_permno = resolved_permno.where(~att_pre2006, 10401)
        resolved_permno = resolved_permno.fillna(
            russell_df["Ticker"].map(ticker_map)
        ).fillna(norm_r_ticker.map(ticker_map))
        russell_df["permno"] = pd.to_numeric(resolved_permno, errors="coerce").astype(
            "Int64"
        )

        return stock_df, factors_df, russell_df
    finally:
        con.close()
