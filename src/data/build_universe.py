"""Point-in-time Russell 1000 universe construction per v1.3 §10.1 & §2.2."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reconstitution_effective_date(
    year: int, convention: str = "june_30"
) -> pd.Timestamp:
    """Compute the annual Russell reconstitution effective date for a given year.

    Parameters
    ----------
    year : int
        Reconstitution year (e.g., 2000..2026).
    convention : str
        Either 'june_30' (v1.3 §10.1 end-of-June approximation, where June 30 Y
        through June 29 Y+1 uses Year=Y constituents) or 'fourth_friday_monday'
        (Monday following the 4th Friday of June).
    """
    if convention == "june_30":
        return pd.Timestamp(year=year, month=6, day=30)
    elif convention == "fourth_friday_monday":
        june_first = pd.Timestamp(year=year, month=6, day=1)
        # weekday(): Monday=0 ... Friday=4
        days_to_first_friday = (4 - june_first.weekday()) % 7
        fourth_friday_day = 1 + days_to_first_friday + 21
        fourth_friday = pd.Timestamp(year=year, month=6, day=fourth_friday_day)
        return fourth_friday + pd.Timedelta(days=3)
    else:
        raise ValueError(f"Unsupported convention: {convention!r}")


def get_recon_year(
    date: pd.Timestamp, convention: str = "june_30", min_year: int = 2000
) -> int:
    """Determine the active reconstitution year for a given calendar date."""
    ts = pd.Timestamp(date)
    eff_date = reconstitution_effective_date(ts.year, convention=convention)
    recon_yr = ts.year if ts >= eff_date else ts.year - 1
    return max(recon_yr, min_year)


def build_universe(
    reconstitution_csv: pd.DataFrame,
    date: pd.Timestamp,
    id_col: str = "Ticker",
    convention: str = "june_30",
) -> set:
    """Return active Russell 1000 constituents on `date` per v1.3 §10.1.

    Reconstitution takes place at the end of June annually.
    Dates in [June 30 Y, June 30 Y+1) (or [ July Y, June Y+1 ) ) use Year=Y constituents.
    """
    year_col = "Year" if "Year" in reconstitution_csv.columns else "year"
    min_yr = (
        int(reconstitution_csv[year_col].min()) if len(reconstitution_csv) > 0 else 2000
    )
    recon_year = get_recon_year(
        pd.Timestamp(date), convention=convention, min_year=min_yr
    )
    members = reconstitution_csv[reconstitution_csv[year_col] == recon_year]
    target_col = (
        id_col
        if id_col in members.columns
        else ("Ticker" if "Ticker" in members.columns else "ticker")
    )
    return set(members[target_col].dropna().unique())


def build_strict_universe(
    russell_membership_df: pd.DataFrame,
    security_history_df: pd.DataFrame,
    date: pd.Timestamp | str,
    convention: str = "june_30",
) -> set[int]:
    """Return Point-in-Time Russell 1000 PERMNOs on `date` strictly filtered to `shrcd IN (10, 11)`.

    Joins active Russell 1000 constituents on `date` to `silver.security_history` on
    `(security_id, date)` (where `date` falls within `[valid_from, valid_to]` /
    `[start_date, end_date]`, or propagates the security's last known `shrcd` on or before
    `date` for post-2024 Yahoo rows after CRSP ends on 2024-12-31), and excludes any
    security with `shrcd` not in `(10, 11)` or with missing `shrcd`.
    """
    if len(russell_membership_df) == 0 or len(security_history_df) == 0:
        return set()

    ts = pd.Timestamp(date)
    year_col = "Year" if "Year" in russell_membership_df.columns else "year"
    min_yr = int(russell_membership_df[year_col].min())
    recon_year = get_recon_year(ts, convention=convention, min_year=min_yr)
    members = russell_membership_df[
        russell_membership_df[year_col] == recon_year
    ].copy()
    if len(members) == 0:
        return set()

    # Resolve security_id / permno on members
    if "permno" in members.columns:
        sec_ids = pd.to_numeric(members["permno"], errors="coerce")
        if "security_id" in members.columns:
            sec_ids = sec_ids.fillna(
                pd.to_numeric(members["security_id"], errors="coerce")
            )
    elif "security_id" in members.columns:
        sec_ids = pd.to_numeric(members["security_id"], errors="coerce")
    else:
        sec_ids = pd.Series(np.nan, index=members.index, dtype=float)

    # Identify columns in security_history_df
    hist_id_col = (
        "security_id"
        if "security_id" in security_history_df.columns
        else ("permno" if "permno" in security_history_df.columns else "PERMNO")
    )
    shrcd_col = "shrcd" if "shrcd" in security_history_df.columns else "SHRCD"
    start_col = (
        "valid_from"
        if "valid_from" in security_history_df.columns
        else ("start_date" if "start_date" in security_history_df.columns else None)
    )
    end_col = (
        "valid_to"
        if "valid_to" in security_history_df.columns
        else ("end_date" if "end_date" in security_history_df.columns else None)
    )

    # Fallback ticker -> security_id resolution if raw silver.russell has NULL security_id rows
    ticker_col = (
        "Ticker"
        if "Ticker" in members.columns
        else ("ticker" if "ticker" in members.columns else None)
    )
    if sec_ids.isna().any() and ticker_col is not None:
        from src.data.normalize_schema import KNOWN_RUSSELL_PERMNO_FIXES

        t_map: dict[str, int] = {}
        if "ticker" in security_history_df.columns:
            hist_valid_t = security_history_df.dropna(
                subset=[hist_id_col, "ticker"]
            ).copy()
            if end_col is not None:
                hist_valid_t = hist_valid_t.sort_values(end_col)
            for h_id, h_t in zip(hist_valid_t[hist_id_col], hist_valid_t["ticker"]):
                p = int(h_id)
                t_s = str(h_t)
                t_map[t_s] = p
                t_map[t_s.replace(".", "")] = p

        r_id_col = (
            "permno"
            if "permno" in russell_membership_df.columns
            else (
                "security_id"
                if "security_id" in russell_membership_df.columns
                else None
            )
        )
        if r_id_col is not None:
            r_valid = russell_membership_df.dropna(
                subset=[r_id_col, ticker_col]
            ).sort_values(year_col)
            for r_id, r_t in zip(r_valid[r_id_col], r_valid[ticker_col]):
                p = int(r_id)
                t_s = str(r_t)
                t_map[t_s] = p
                t_map[t_s.replace(".", "")] = p

        for k_t, k_p in KNOWN_RUSSELL_PERMNO_FIXES.items():
            t_map[k_t] = k_p
            t_map[k_t.replace(".", "")] = k_p

        m_tickers = members[ticker_col].astype(str)
        m_norm = m_tickers.str.replace(".", "", regex=False)
        sec_ids = sec_ids.fillna(m_tickers.map(t_map)).fillna(m_norm.map(t_map))
        if recon_year <= 2005:
            sec_ids = sec_ids.where(m_tickers != "T", 10401)

    cand_permnos = set(sec_ids.dropna().astype(np.int64).unique())
    if not cand_permnos:
        return set()

    hist = security_history_df.loc[
        security_history_df[hist_id_col].isin(cand_permnos)
        & security_history_df[shrcd_col].notna()
    ].copy()
    if len(hist) == 0:
        return set()

    hist[hist_id_col] = hist[hist_id_col].astype(np.int64)
    hist[shrcd_col] = pd.to_numeric(hist[shrcd_col], errors="coerce")

    if start_col is not None and end_col is not None:
        s_dates = pd.to_datetime(hist[start_col])
        e_dates = pd.to_datetime(hist[end_col])
        in_interval = (s_dates <= ts) & (ts <= e_dates)
        active_on_date = (
            hist.loc[in_interval]
            .assign(_s=s_dates.loc[in_interval], _e=e_dates.loc[in_interval])
            .sort_values([hist_id_col, "_e", "_s"])
            .groupby(hist_id_col, sort=False)[shrcd_col]
            .last()
        )
        # Propagate last known shrcd on or before `ts` for post-2024 Yahoo rows (after CRSP ends on 2024-12-31)
        prior_mask = s_dates <= ts
        last_known = (
            hist.loc[prior_mask]
            .assign(_s=s_dates.loc[prior_mask], _e=e_dates.loc[prior_mask])
            .sort_values([hist_id_col, "_e", "_s"])
            .groupby(hist_id_col, sort=False)[shrcd_col]
            .last()
        )
        resolved_shrcd = active_on_date.combine_first(last_known)
    elif "date" in hist.columns:
        d_dates = pd.to_datetime(hist["date"])
        prior = (
            hist.loc[d_dates <= ts]
            .assign(_d=d_dates.loc[d_dates <= ts])
            .sort_values([hist_id_col, "_d"])
        )
        resolved_shrcd = prior.groupby(hist_id_col, sort=False)[shrcd_col].last()
    else:
        resolved_shrcd = hist.groupby(hist_id_col, sort=False)[shrcd_col].last()

    strict_ids = resolved_shrcd.index[resolved_shrcd.isin([10, 11])]
    return {int(p) for p in strict_ids}


def assign_in_universe(
    stock_df: pd.DataFrame,
    reconstitution_csv: pd.DataFrame,
    filter_shrcd: bool = False,
    convention: str = "june_30",
) -> pd.Series:
    """Vectorized Point-in-Time universe assignment for a panel of (permno, date) rows.

    Enforces:
    1. Only constituents in the active reconstitution cycle for `date` are marked True.
    2. Pre-reconstitution dates (< 2000-06-30) are marked False (warmup history only).
    3. No constituent appears in the universe before its first valid price observation.
    4. Optional filter `shrcd IN (10, 11)` when `filter_shrcd=True`.
    """
    year_col = "Year" if "Year" in reconstitution_csv.columns else "year"
    dates = pd.to_datetime(stock_df["date"])

    if convention == "june_30":
        is_post_recon = (dates.dt.month > 6) | (
            (dates.dt.month == 6) & (dates.dt.day >= 30)
        )
        recon_years = np.where(is_post_recon, dates.dt.year, dates.dt.year - 1)
    else:
        unique_years = dates.dt.year.unique()
        eff_map = {
            int(y): reconstitution_effective_date(int(y), convention=convention)
            for y in unique_years
        }
        eff_dates = dates.dt.year.map(eff_map)
        recon_years = np.where(dates >= eff_dates, dates.dt.year, dates.dt.year - 1)

    # Match primarily by (Year, permno), falling back to (Year, Ticker) where needed
    valid_permno_members = reconstitution_csv.dropna(subset=[year_col, "permno"]).copy()
    valid_permno_members[year_col] = valid_permno_members[year_col].astype(np.int64)
    valid_permno_members["permno"] = valid_permno_members["permno"].astype(np.int64)
    permno_pairs = pd.MultiIndex.from_frame(
        valid_permno_members[[year_col, "permno"]].drop_duplicates()
    )

    row_permno_Index = pd.MultiIndex.from_arrays(
        [recon_years.astype(np.int64), stock_df["permno"].astype(np.int64)]
    )
    in_univ = pd.Series(
        row_permno_Index.isin(permno_pairs), index=stock_df.index, dtype=bool
    )

    # Also check Ticker match for any rows where permno wasn't in permno_pairs
    ticker_col = "Ticker" if "Ticker" in reconstitution_csv.columns else "ticker"
    if ticker_col in reconstitution_csv.columns and "ticker" in stock_df.columns:
        valid_ticker_members = reconstitution_csv.dropna(
            subset=[year_col, ticker_col]
        ).copy()
        valid_ticker_members[year_col] = valid_ticker_members[year_col].astype(np.int64)
        valid_ticker_members["_t"] = (
            valid_ticker_members[ticker_col]
            .astype(str)
            .str.replace(".", "", regex=False)
        )
        ticker_pairs = pd.MultiIndex.from_frame(
            valid_ticker_members[[year_col, "_t"]].drop_duplicates()
        )
        row_t_Index = pd.MultiIndex.from_arrays(
            [
                recon_years.astype(np.int64),
                stock_df["ticker"]
                .fillna("")
                .astype(str)
                .str.replace(".", "", regex=False),
            ]
        )
        in_univ = in_univ | pd.Series(
            row_t_Index.isin(ticker_pairs), index=stock_df.index, dtype=bool
        )

    # Dates strictly before 2000-06-30 are warmup-only and not part of the evaluation universe
    in_univ = in_univ & (dates >= pd.Timestamp("2000-06-30"))

    # Ensure valid price observation exists
    if "split_adj_close" in stock_df.columns:
        first_valid_date = (
            stock_df.loc[
                stock_df["split_adj_close"].notna() & (stock_df["split_adj_close"] > 0)
            ]
            .groupby("permno")["date"]
            .min()
        )
        in_univ = in_univ & (dates >= stock_df["permno"].map(first_valid_date))

    if filter_shrcd and "shrcd" in stock_df.columns:
        in_univ = in_univ & stock_df["shrcd"].isin([10, 11])

    return in_univ
