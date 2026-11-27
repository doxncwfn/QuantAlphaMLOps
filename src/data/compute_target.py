"""Target construction (target_5d and y_star) and delisting policy per v1.3 §0.1–0.4 & §10.4."""

from __future__ import annotations

import numpy as np
import pandas as pd

MA_DELIST_CODES: frozenset[int] = frozenset({200, 231, 233, 241, 244})
ADVERSE_DELIST_CODES: frozenset[int] = frozenset({500, 520, 560, 570, 574, 580, 584})


def apply_delisting(row: pd.Series | dict) -> float | None:
    """Return terminal return for a delisting event per v1.3 §10.4.

    Rules:
    - If `dlstcd` is missing or equals 100 (still active): return None.
    - If `dlret` (or `dlretx`) is available (non-NaN): return float(dlret).
    - If `dlstcd` in (200, 231, 233, 241, 244) (M&A): return 0.0.
    - If `dlstcd` in (500, 520, 560, 570, 574, 580, 584) (adverse): return -0.30.
    - Otherwise (unknown/unresolved delisting code): return np.nan.
    """
    dlstcd_val = row.get("dlstcd")
    if pd.isna(dlstcd_val):
        return None
    code = int(dlstcd_val)
    if code == 100:
        return None  # Still active (Risk R7: 2024 active code 100)

    dlret = row.get("dlret")
    if pd.isna(dlret):
        dlret = row.get("dlretx")
    if pd.notna(dlret):
        return float(dlret)

    if code in MA_DELIST_CODES:
        return 0.0  # M&A: deal price proxy
    elif code in ADVERSE_DELIST_CODES:
        return -0.30  # Adverse failure penalty
    return np.nan


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compute `target_5d` and cross-sectional rank `y_star` per v1.3 §0.1–0.4.

    Parameters
    ----------
    df : pd.DataFrame
        Panel sorted by `(permno, date)` with columns `permno`, `date`, `ret_price`,
        `dlstcd`, `dlret`, `in_universe`, and `era`.

    Returns
    -------
    pd.DataFrame
        DataFrame with `target_5d` and `y_star` columns populated.
    """
    out = df.copy()
    p_key = out["permno"]
    ret_p = out["ret_price"]

    # Standard 5-session forward cumulative log price return: sum_{s=1..5} ret_price_{t+s}
    s1 = ret_p.groupby(p_key, sort=False).shift(-1)
    s2 = ret_p.groupby(p_key, sort=False).shift(-2)
    s3 = ret_p.groupby(p_key, sort=False).shift(-3)
    s4 = ret_p.groupby(p_key, sort=False).shift(-4)
    s5 = ret_p.groupby(p_key, sort=False).shift(-5)

    target_5d = (s1 + s2 + s3 + s4 + s5).to_numpy(dtype=np.float64, copy=True)

    # Handle delistings occurring within [t, t+5] in the CRSP era:
    # When a security's terminal session L has a delisting code != 100, for any decision
    # session t in [max(start, L-4), L] where fewer than 5 future trading sessions exist,
    # target_5d[t] = sum(ret_price[t+1 : L+1]) + apply_delisting(row_L).
    permnos = out["permno"].to_numpy()
    split_points = np.flatnonzero(permnos[1:] != permnos[:-1]) + 1
    starts = np.concatenate([[0], split_points])
    ends = np.concatenate([split_points, [len(out)]])

    dlstcd_arr = out["dlstcd"].to_numpy(dtype=np.float64)
    dlret_arr = out["dlret"].to_numpy(dtype=np.float64)
    dlretx_arr = (
        out["dlretx"].to_numpy(dtype=np.float64)
        if "dlretx" in out.columns
        else dlret_arr
    )
    ret_p_arr = ret_p.to_numpy(dtype=np.float64)
    era_arr = out["era"].to_numpy()

    for st_idx, end_idx in zip(starts, ends):
        last_idx = end_idx - 1
        if era_arr[last_idx] != "CRSP":
            continue
        code_val = dlstcd_arr[last_idx]
        if np.isnan(code_val) or int(code_val) == 100:
            continue

        row_dict = {
            "dlstcd": code_val,
            "dlret": dlret_arr[last_idx],
            "dlretx": dlretx_arr[last_idx],
        }
        term_ret = apply_delisting(row_dict)
        if term_ret is None:
            continue

        # Apply terminal return + intermediate prior returns for t in [max(st_idx, last_idx - 4), last_idx]
        win_start = max(st_idx, last_idx - 4)
        for t_idx in range(win_start, last_idx + 1):
            if np.isnan(term_ret):
                target_5d[t_idx] = np.nan
            else:
                inter_rets = ret_p_arr[t_idx + 1 : last_idx + 1]
                if len(inter_rets) > 0 and np.isnan(inter_rets).any():
                    target_5d[t_idx] = np.nan
                else:
                    prior_sum = (
                        float(np.sum(inter_rets)) if len(inter_rets) > 0 else 0.0
                    )
                    target_5d[t_idx] = prior_sum + float(term_ret)

    out["target_5d"] = target_5d

    # Cross-sectional rank y_star across active constituents U_t^valid (v1.3 §0.3 & §0.5.1):
    # y^*_{i,t} = (Rank(Y_{i,t}) - 1) / (|U_t^valid| - 1) - 0.5
    in_univ = (
        out["in_universe"].to_numpy(dtype=bool)
        if "in_universe" in out.columns
        else np.ones(len(out), dtype=bool)
    )
    valid_target_mask = in_univ & out["target_5d"].notna().to_numpy()

    y_star = pd.Series(np.nan, index=out.index, dtype=np.float64)
    if valid_target_mask.any():
        valid_sub = out.loc[valid_target_mask, ["date", "target_5d"]]
        ranks = valid_sub.groupby("date")["target_5d"].rank(
            method="average", ascending=True
        )
        counts = valid_sub.groupby("date")["target_5d"].transform("count")
        has_pairs = counts >= 2
        y_star_sub = pd.Series(np.nan, index=valid_sub.index, dtype=np.float64)
        y_star_sub.loc[has_pairs] = (ranks.loc[has_pairs] - 1.0) / (
            counts.loc[has_pairs] - 1.0
        ) - 0.5
        y_star.loc[valid_sub.index] = y_star_sub

    out["y_star"] = y_star
    return out
