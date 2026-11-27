"""Computation of Layer 1, 2, 4, and 5 features per frozen v1.3 specification."""

from __future__ import annotations

import numpy as np
import pandas as pd

STOCK_FEATURE_COLS: list[str] = [
    # Layer 1 — Primitives (6)
    "r_cc",
    "r_on",
    "r_id",
    "range",
    "close_loc",
    "rel_vol",
    # Layer 2 — Temporal Statistics (10)
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "vol_20d",
    "vol_60d",
    "vol_ratio",
    "price_z_20",
    "drawdown_20",
    "eff_ratio_20",
    "vol_shock_5",
    # Layer 4 — Six-Factor Residualization (3)
    "resid_ret",
    "resid_mom_20",
    "resid_vol_20",
]

CONTEXT_FEATURE_COLS: list[str] = [
    # Layer 5 — Market-Context Branch (11)
    "ctx_mktrf",
    "ctx_smb",
    "ctx_hml",
    "ctx_rmw",
    "ctx_cma",
    "ctx_umd",
    "ctx_mkt_vol_20",
    "ctx_mkt_ret_20",
    "ctx_breadth",
    "ctx_dispersion",
    "ctx_factor_disp",
]

ALL_FEATURE_COLS: list[str] = STOCK_FEATURE_COLS + CONTEXT_FEATURE_COLS

FACTOR_COLS: list[str] = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
STYLE_FACTOR_COLS: list[str] = ["smb", "hml", "rmw", "cma", "umd"]


def compute_layer1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Layer 1 primitive daily features (6 features) per v1.3 §1.2.

    Assumes `df` is sorted by `(permno, date)`.
    """
    out = df.copy()
    g = out.groupby("permno", sort=False)

    # 1. r_cc: daily close-to-close log price return (authoritative ret_price)
    is_first_obs = g.cumcount() == 0
    r_cc = out["ret_price"].where(~is_first_obs, np.nan)
    out["r_cc"] = r_cc

    # 2 & 3. r_id = log(C_s / O_s) and r_on = log(O_s / C_{s-1})
    # When OPENPRC is valid (>0) and PRC is valid (>0), r_id = log(split_adj_close / split_adj_open).
    # For r_on, using r_cc - r_id where r_cc is valid guarantees exact split/distribution factor
    # alignment with CRSP RETX and enforces the algebraic identity r_on + r_id = r_cc to <1e-15.
    valid_open = (
        out["open_raw"].notna()
        & (out["open_raw"] > 0)
        & out["split_adj_open"].notna()
        & (out["split_adj_open"] > 0)
    )
    valid_close = (
        out["close_raw"].notna()
        & (out["close_raw"] > 0)
        & out["split_adj_close"].notna()
        & (out["split_adj_close"] > 0)
    )
    valid_oc = valid_open & valid_close

    r_id = pd.Series(np.nan, index=out.index, dtype=np.float64)
    r_id.loc[valid_oc] = np.log(
        out.loc[valid_oc, "split_adj_close"] / out.loc[valid_oc, "split_adj_open"]
    )
    out["r_id"] = r_id

    prev_close = g["split_adj_close"].shift(1)
    prev_cfacpr = g["cfacpr"].shift(1)
    r_on = pd.Series(np.nan, index=out.index, dtype=np.float64)
    has_rcc_and_rid = r_cc.notna() & r_id.notna()
    r_on.loc[has_rcc_and_rid] = r_cc.loc[has_rcc_and_rid] - r_id.loc[has_rcc_and_rid]

    # Fallback if r_cc is NaN on a non-first row with unchanged cfacpr and valid open/prev_close
    fallback_mask = (
        (~has_rcc_and_rid)
        & (~is_first_obs)
        & valid_open
        & prev_close.notna()
        & (prev_close > 0)
        & (out["cfacpr"] == prev_cfacpr)
    )
    if fallback_mask.any():
        r_on.loc[fallback_mask] = np.log(
            out.loc[fallback_mask, "split_adj_open"] / prev_close.loc[fallback_mask]
        )
    out["r_on"] = r_on

    # 4. range = (H_s - L_s) / C_s; NaN if any OHLC null
    valid_hl = (
        out["high_raw"].notna()
        & (out["high_raw"] > 0)
        & out["low_raw"].notna()
        & (out["low_raw"] > 0)
        & out["split_adj_high"].notna()
        & out["split_adj_low"].notna()
        & (out["split_adj_high"] >= out["split_adj_low"])
    )
    valid_ohlc = valid_oc & valid_hl

    hl_diff = out["split_adj_high"] - out["split_adj_low"]
    range_feat = pd.Series(np.nan, index=out.index, dtype=np.float64)
    range_feat.loc[valid_ohlc] = (
        hl_diff.loc[valid_ohlc] / out.loc[valid_ohlc, "split_adj_close"]
    ).clip(lower=0.0)
    out["range"] = range_feat

    # 5. close_loc = (C_s - L_s) / (H_s - L_s); NaN if H_s == L_s or any OHLC null
    valid_cloc = valid_ohlc & (hl_diff > 0)
    close_loc = pd.Series(np.nan, index=out.index, dtype=np.float64)
    close_loc.loc[valid_cloc] = (
        (out.loc[valid_cloc, "split_adj_close"] - out.loc[valid_cloc, "split_adj_low"])
        / hl_diff.loc[valid_cloc]
    ).clip(0.0, 1.0)
    out["close_loc"] = close_loc

    # 6. rel_vol = log(V_s / median_20(V_s)), clipped [-5, 5]; NaN if V_s <= 0 or <20 valid obs
    pos_vol = out["volume"].where(out["volume"] > 0, np.nan)
    med_20_vol = (
        pos_vol.groupby(out["permno"], sort=False)
        .rolling(window=20, min_periods=20)
        .median()
        .reset_index(level=0, drop=True)
    )
    valid_rvol = pos_vol.notna() & med_20_vol.notna() & (med_20_vol > 0)
    rel_vol = pd.Series(np.nan, index=out.index, dtype=np.float64)
    rel_vol.loc[valid_rvol] = np.log(
        pos_vol.loc[valid_rvol] / med_20_vol.loc[valid_rvol]
    ).clip(-5.0, 5.0)
    out["rel_vol"] = rel_vol

    return out


def compute_layer2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Layer 2 temporal statistics (10 features) per v1.3 §1.3.

    Assumes `df` is sorted by `(permno, date)` and contains Layer 1 features.
    """
    out = df.copy()
    p_key = out["permno"]
    r_cc = out["r_cc"]

    # 7. ret_5d: sum(r_cc[s-4:s+1]), NaN if any r_cc missing (min_periods=5)
    out["ret_5d"] = (
        r_cc.groupby(p_key, sort=False)
        .rolling(window=5, min_periods=5)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # 8. ret_20d: sum(r_cc[s-19:s+1]), NaN if >4 missing (min_periods=16)
    out["ret_20d"] = (
        r_cc.groupby(p_key, sort=False)
        .rolling(window=20, min_periods=16)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # 9. ret_60d: sum(r_cc[s-59:s+1]), NaN if >12 missing (min_periods=48)
    out["ret_60d"] = (
        r_cc.groupby(p_key, sort=False)
        .rolling(window=60, min_periods=48)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # 10. vol_20d: std(r_cc[s-19:s+1], ddof=1), NaN if <10 valid returns
    vol_20d = (
        r_cc.groupby(p_key, sort=False)
        .rolling(window=20, min_periods=10)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )
    out["vol_20d"] = vol_20d

    # 11. vol_60d: std(r_cc[s-59:s+1], ddof=1), NaN if <30 valid returns
    vol_60d = (
        r_cc.groupby(p_key, sort=False)
        .rolling(window=60, min_periods=30)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )
    out["vol_60d"] = vol_60d

    # 12. vol_ratio: vol_20d / vol_60d
    out["vol_ratio"] = np.where(
        vol_20d.notna() & vol_60d.notna() & (vol_60d > 0),
        vol_20d / vol_60d,
        np.nan,
    )

    # Split-contamination rule (v1.3 §1.3 & §10.3):
    # In the CRSP era, if a split event (CFACPR_tau != CFACPR_{tau-1}) occurs at any
    # session tau within the 20-session window [s-19, s] (equivalently CFACPR is not
    # constant across [s-20, s]), price_z_20 and drawdown_20 are assigned NaN.
    # For a split on session s_0, this sets s_0, s_0+1, ..., s_0+19 (20 sessions) to NaN,
    # resuming non-NaN values on session 21 (s_0+20).
    # In the Yahoo era (cfacpr == 1.0), this check is a no-op.
    cfacpr_s = out["cfacpr"]
    prev_cfacpr = cfacpr_s.groupby(p_key, sort=False).shift(1)
    is_split_session = (
        (out["era"] == "CRSP")
        & prev_cfacpr.notna()
        & cfacpr_s.notna()
        & (np.abs(cfacpr_s - prev_cfacpr) > 1e-9)
    ).astype(np.float64)
    split_contaminated = (
        is_split_session.groupby(p_key, sort=False)
        .rolling(window=20, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        > 0
    )

    # 13. price_z_20: log(C_s / SMA_20,s) / (vol_20d * sqrt(20))
    # where SMA_20,s = exp(mean(log(C[s-19:s+1]))) -> log(C_s / SMA_20,s) = log(C_s) - mean(log(C))
    valid_c = out["split_adj_close"].where(out["split_adj_close"] > 0, np.nan)
    log_c = np.log(valid_c)
    mean_log_c_20 = (
        log_c.groupby(p_key, sort=False)
        .rolling(window=20, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )
    denom_pz = vol_20d * np.sqrt(20.0)
    valid_pz = (
        (~split_contaminated)
        & log_c.notna()
        & mean_log_c_20.notna()
        & denom_pz.notna()
        & (denom_pz > 0)
    )
    price_z_20 = pd.Series(np.nan, index=out.index, dtype=np.float64)
    price_z_20.loc[valid_pz] = (
        log_c.loc[valid_pz] - mean_log_c_20.loc[valid_pz]
    ) / denom_pz.loc[valid_pz]
    out["price_z_20"] = price_z_20

    # 14. drawdown_20: log(C_s / max(C[s-19:s+1]))
    max_c_20 = (
        valid_c.groupby(p_key, sort=False)
        .rolling(window=20, min_periods=10)
        .max()
        .reset_index(level=0, drop=True)
    )
    valid_dd = (
        (~split_contaminated) & valid_c.notna() & max_c_20.notna() & (max_c_20 > 0)
    )
    drawdown_20 = pd.Series(np.nan, index=out.index, dtype=np.float64)
    drawdown_20.loc[valid_dd] = np.log(
        valid_c.loc[valid_dd] / max_c_20.loc[valid_dd]
    ).clip(upper=0.0)
    out["drawdown_20"] = drawdown_20

    # 15. eff_ratio_20: |ret_20d| / sum(|r_cc[s-19:s+1]|), NaN if denominator == 0 or ret_20d is NaN
    abs_rcc_sum_20 = (
        r_cc.abs()
        .groupby(p_key, sort=False)
        .rolling(window=20, min_periods=16)
        .sum()
        .reset_index(level=0, drop=True)
    )
    ret_20d = out["ret_20d"]
    valid_eff = ret_20d.notna() & abs_rcc_sum_20.notna() & (abs_rcc_sum_20 > 0)
    eff_ratio_20 = pd.Series(np.nan, index=out.index, dtype=np.float64)
    eff_ratio_20.loc[valid_eff] = (
        ret_20d.loc[valid_eff].abs() / abs_rcc_sum_20.loc[valid_eff]
    ).clip(0.0, 1.0)
    out["eff_ratio_20"] = eff_ratio_20

    # 16. vol_shock_5: mean(rel_vol[s-4:s+1]), NaN if >2 missing (min_periods=3)
    out["vol_shock_5"] = (
        out["rel_vol"]
        .groupby(p_key, sort=False)
        .rolling(window=5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return out


def compute_residuals(
    stock_excess: pd.Series,
    factors: pd.DataFrame,
    min_obs: int = 126,
    n_obs: int = 252,
) -> dict[str, pd.Series]:
    """Compute strictly causal 6-factor residuals per v1.3 §6.1.

    Beta is estimated over the trailing 252 sessions ending strictly at s-1.
    Residual at s is computed using the lagged beta vector beta_{s-1}.
    Historical residuals are permanently stored and never re-estimated.
    Guards against collinear factor regimes via condition number thresholding (cond(X) > 1e10).
    """
    dates = stock_excess.index.sort_values()
    stock_excess_sorted = stock_excess.reindex(dates)
    factors_aligned = factors.reindex(dates)[FACTOR_COLS]

    epsilon = pd.Series(np.nan, index=dates, dtype=np.float64)
    beta_mkt = pd.Series(np.nan, index=dates, dtype=np.float64)
    alpha = pd.Series(np.nan, index=dates, dtype=np.float64)

    y_vals = stock_excess_sorted.to_numpy(dtype=np.float64)
    f_vals = factors_aligned.to_numpy(dtype=np.float64)
    valid_row = (~np.isnan(y_vals)) & (~np.isnan(f_vals).any(axis=1))

    n_total = len(dates)
    for idx in range(n_obs, n_total):
        train_start = idx - n_obs
        train_end = (
            idx  # slice [idx - n_obs : idx] corresponds to positions [s-252, ..., s-1]
        )
        win_valid = valid_row[train_start:train_end]
        n_common = int(win_valid.sum())
        if n_common < min_obs:
            continue

        y_win = y_vals[train_start:train_end][win_valid]
        f_win = f_vals[train_start:train_end][win_valid]
        X = np.column_stack([np.ones(n_common, dtype=np.float64), f_win])

        if np.linalg.cond(X) > 1e10:
            continue

        try:
            params = np.linalg.lstsq(X, y_win, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        alpha.iloc[idx] = params[0]
        beta_mkt.iloc[idx] = params[1]
        beta_vec = params[1:]

        if valid_row[idx]:
            F_s = f_vals[idx]
            predicted = params[0] + float(np.dot(beta_vec, F_s))
            epsilon.iloc[idx] = y_vals[idx] - predicted

    resid_mom_20 = epsilon.rolling(20, min_periods=16).sum()
    resid_vol_20 = epsilon.rolling(20, min_periods=10).std(ddof=1)

    return {
        "epsilon": epsilon,
        "resid_mom_20": resid_mom_20,
        "resid_vol_20": resid_vol_20,
        "beta_mkt": beta_mkt,
        "alpha": alpha,
    }


def _compute_residuals_fast_array(
    y_vals: np.ndarray,
    f_vals: np.ndarray,
    min_obs: int = 126,
    n_obs: int = 252,
) -> np.ndarray:
    """Fast vectorized exact OLS solver for a single security's excess return array.

    Matches `compute_residuals()` to machine epsilon (<1e-13) by computing rolling
    XtX and Xty via cumulative sums and verifying condition numbers.
    """
    n_total = len(y_vals)
    eps = np.full(n_total, np.nan, dtype=np.float64)
    if n_total <= n_obs:
        return eps

    valid = (~np.isnan(y_vals)) & (~np.isnan(f_vals).any(axis=1))
    if valid.sum() < min_obs:
        return eps

    k = 7  # intercept + 6 factors
    X_clean = np.zeros((n_total, k), dtype=np.float64)
    X_clean[valid, 0] = 1.0
    X_clean[valid, 1:] = f_vals[valid]
    y_clean = np.where(valid, y_vals, 0.0)

    # Cumulative sums of valid counts, XtX (T x 7 x 7), and Xty (T x 7)
    cs_n = np.zeros(n_total + 1, dtype=np.int64)
    cs_n[1:] = np.cumsum(valid.astype(np.int64))

    XX = X_clean[:, :, None] * X_clean[:, None, :]
    cs_XX = np.zeros((n_total + 1, k, k), dtype=np.float64)
    np.cumsum(XX, axis=0, out=cs_XX[1:])

    Xy = X_clean * y_clean[:, None]
    cs_Xy = np.zeros((n_total + 1, k), dtype=np.float64)
    np.cumsum(Xy, axis=0, out=cs_Xy[1:])

    # Windows [idx - n_obs, idx - 1] for idx in [n_obs, ..., n_total - 1]
    n_win = cs_n[n_obs:n_total] - cs_n[: n_total - n_obs]
    XtX_win = cs_XX[n_obs:n_total] - cs_XX[: n_total - n_obs]
    Xty_win = cs_Xy[n_obs:n_total] - cs_Xy[: n_total - n_obs]

    target_valid = valid[n_obs:n_total]
    has_min_obs = (n_win >= min_obs) & target_valid
    if not has_min_obs.any():
        return eps

    cand_indices = np.flatnonzero(has_min_obs)
    XtX_cand = XtX_win[cand_indices]
    Xty_cand = Xty_win[cand_indices]

    # Eigenvalues of symmetric XtX_cand to screen condition number:
    # cond(X) = sqrt(lambda_max(XtX) / lambda_min(XtX))
    eigs = np.linalg.eigvalsh(XtX_cand)
    l_min = eigs[:, 0]
    l_max = eigs[:, -1]

    # Fast well-conditioned regime: cond(X) <= 1e6 -> l_min > 0 and l_max / l_min <= 1e12
    well_cond = (l_min > 1e-18) & ((l_max / np.maximum(l_min, 1e-30)) <= 1e12)

    if well_cond.any():
        wc_pos = cand_indices[well_cond]
        params_wc = np.linalg.solve(
            XtX_cand[well_cond], Xty_cand[well_cond][..., None]
        ).squeeze(-1)
        idx_wc = wc_pos + n_obs
        pred_wc = params_wc[:, 0] + np.einsum(
            "ij,ij->i", params_wc[:, 1:], f_vals[idx_wc]
        )
        eps[idx_wc] = y_vals[idx_wc] - pred_wc

    # Borderline / ill-conditioned candidates: check exact np.linalg.cond(X) > 1e10 and lstsq
    borderline = ~well_cond
    if borderline.any():
        for pos in cand_indices[borderline]:
            idx = int(pos + n_obs)
            train_start = idx - n_obs
            win_valid = valid[train_start:idx]
            y_w = y_vals[train_start:idx][win_valid]
            f_w = f_vals[train_start:idx][win_valid]
            X_w = np.column_stack([np.ones(len(y_w), dtype=np.float64), f_w])
            if np.linalg.cond(X_w) > 1e10:
                continue
            try:
                params = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
                eps[idx] = y_vals[idx] - (
                    params[0] + float(np.dot(params[1:], f_vals[idx]))
                )
            except np.linalg.LinAlgError:
                continue

    return eps


def compute_layer4_features(df: pd.DataFrame, factors_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Layer 4 six-factor residualization features (3 features) per v1.3 §1.4 & §6.1.

    Assumes `df` is sorted by `(permno, date)`.
    """
    out = df.copy()
    fac = (
        factors_df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="first")
        .set_index("date")
    )

    # Align factor values and RF to stock dates
    stock_dates = out["date"]
    f_aligned = fac.reindex(stock_dates)
    rf_vals = f_aligned["rf"].to_numpy(dtype=np.float64)
    f_matrix = f_aligned[FACTOR_COLS].to_numpy(dtype=np.float64)

    # Excess return R^e_s = ret_total_s - rf_s (v1.3 §1.1)
    is_first_obs = out.groupby("permno", sort=False).cumcount() == 0
    ret_total = out["ret_total"].where(~is_first_obs, np.nan).to_numpy(dtype=np.float64)
    stock_excess = ret_total - rf_vals

    # Group boundaries by permno (since df is sorted by permno, date)
    permnos = out["permno"].to_numpy()
    split_points = np.flatnonzero(permnos[1:] != permnos[:-1]) + 1
    starts = np.concatenate([[0], split_points])
    ends = np.concatenate([split_points, [len(out)]])

    eps_all = np.full(len(out), np.nan, dtype=np.float64)
    for st_idx, end_idx in zip(starts, ends):
        if end_idx - st_idx <= 252:
            continue
        eps_all[st_idx:end_idx] = _compute_residuals_fast_array(
            stock_excess[st_idx:end_idx],
            f_matrix[st_idx:end_idx],
            min_obs=126,
            n_obs=252,
        )

    out["resid_ret"] = eps_all
    p_key = out["permno"]
    out["resid_mom_20"] = (
        out["resid_ret"]
        .groupby(p_key, sort=False)
        .rolling(window=20, min_periods=16)
        .sum()
        .reset_index(level=0, drop=True)
    )
    out["resid_vol_20"] = (
        out["resid_ret"]
        .groupby(p_key, sort=False)
        .rolling(window=20, min_periods=10)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )

    return out


def compute_layer5_features(df: pd.DataFrame, factors_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Layer 5 market-context features (11 features) per v1.3 §1.5.

    Context features are computed once per trading date s and broadcast across all stocks on date s.
    `ctx_breadth` and `ctx_dispersion` are computed from RAW `r_cc` across active universe U_s
    (`in_universe == True`) before any cross-sectional transformation.
    """
    out = df.copy()
    fac = (
        factors_df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="first")
        .set_index("date")
        .copy()
    )

    # Factor-based context series on the factor trading calendar
    fac["ctx_mktrf"] = fac["mktrf"]
    fac["ctx_smb"] = fac["smb"]
    fac["ctx_hml"] = fac["hml"]
    fac["ctx_rmw"] = fac["rmw"]
    fac["ctx_cma"] = fac["cma"]
    fac["ctx_umd"] = fac["umd"]
    fac["ctx_mkt_vol_20"] = fac["mktrf"].rolling(window=20, min_periods=10).std(ddof=1)
    fac["ctx_mkt_ret_20"] = fac["mktrf"].rolling(window=20, min_periods=10).sum()

    # M11: ctx_factor_disp = std(smb_s, hml_s, rmw_s, cma_s, umd_s) with ddof=1; NaN if any missing
    style_mat = fac[STYLE_FACTOR_COLS]
    any_style_missing = style_mat.isna().any(axis=1)
    fac["ctx_factor_disp"] = style_mat.std(axis=1, ddof=1).where(
        ~any_style_missing, np.nan
    )

    # M9 & M10: ctx_breadth and ctx_dispersion across active constituents U_s (in_universe == True)
    if "in_universe" in out.columns:
        univ_rows = out.loc[out["in_universe"] & out["r_cc"].notna(), ["date", "r_cc"]]
    else:
        univ_rows = out.loc[out["r_cc"].notna(), ["date", "r_cc"]]

    if len(univ_rows) > 0:
        pos_ind = (univ_rows["r_cc"] > 0).astype(np.float64)
        breadth_by_date = pos_ind.groupby(univ_rows["date"]).mean()
        disp_group = univ_rows.groupby("date")["r_cc"]
        disp_count = disp_group.count()
        disp_std = disp_group.std(ddof=1).where(disp_count >= 30, np.nan)
    else:
        breadth_by_date = pd.Series(dtype=np.float64)
        disp_std = pd.Series(dtype=np.float64)

    # Map all 11 context features onto `out` by `date`
    factor_ctx_cols = [
        "ctx_mktrf",
        "ctx_smb",
        "ctx_hml",
        "ctx_rmw",
        "ctx_cma",
        "ctx_umd",
        "ctx_mkt_vol_20",
        "ctx_mkt_ret_20",
        "ctx_factor_disp",
    ]
    for col in factor_ctx_cols:
        out[col] = out["date"].map(fac[col]).to_numpy(dtype=np.float64, copy=True)

    ctx_b = out["date"].map(breadth_by_date).to_numpy(dtype=np.float64, copy=True)
    # Per v1.3 §1.5 M9: if U_s is empty on a post-2000-06-30 date, default to 0.5
    if "in_universe" in out.columns:
        empty_univ_dates = (out["date"] >= pd.Timestamp("2000-06-30")).to_numpy()
        ctx_b = np.where(empty_univ_dates & np.isnan(ctx_b), 0.5, ctx_b)
    out["ctx_breadth"] = ctx_b
    out["ctx_dispersion"] = (
        out["date"].map(disp_std).to_numpy(dtype=np.float64, copy=True)
    )

    return out


def compute_all_features(df: pd.DataFrame, factors_df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature computation layers (Layers 1, 2, 4, 5) in causal order."""
    out = df.sort_values(["permno", "date"]).reset_index(drop=True)
    out = compute_layer1_features(out)
    out = compute_layer2_features(out)
    out = compute_layer4_features(out, factors_df)
    out = compute_layer5_features(out, factors_df)
    return out
