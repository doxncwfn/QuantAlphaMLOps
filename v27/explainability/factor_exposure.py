"""Factor attribution (3F Newey-West OLS, FF6 HAC OLS) and 12-sector net/active exposures."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v27.config import CANDIDATES_24
from v27.data.bundle import V7DataBundle


def newey_west_ols(y: np.ndarray, X: np.ndarray, max_lag: int = 5) -> dict[str, Any]:
    """Run OLS with Newey-West HAC covariance matrix."""
    n, _ = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    Xe = X * resid[:, None]
    S = (Xe.T @ Xe) / n
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        G = (Xe[lag:].T @ Xe[:-lag]) / n
        S += w * (G + G.T)
    cov_b = n * (XtX_inv @ S @ XtX_inv)
    se = np.sqrt(np.maximum(np.diag(cov_b), 1e-18))
    tstat = beta / se
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / max(ss_tot, 1e-18)
    return {"beta": beta, "se": se, "tstat": tstat, "r2": r2}


def compute_3f_attribution(
    bundle: V7DataBundle,
    net_df: pd.DataFrame,
    tab_dir: Path | None = None,
    ann_days: int = 252,
) -> pd.DataFrame:
    """Attribute 24 candidates onto custom MKT/VOL/MOM factors with Newey-West HAC alpha t-stats."""
    t_d = len(bundle.dates)
    X_3f = np.column_stack(
        [
            np.ones(t_d),
            bundle.factors_3f["MKT"].to_numpy(),
            bundle.factors_3f["VOL"].to_numpy(),
            bundle.factors_3f["MOM"].to_numpy(),
        ]
    )
    rows = []
    for cid in CANDIDATES_24:
        r = net_df[cid]
        res = newey_west_ols(r.to_numpy(dtype=np.float64), X_3f, max_lag=5)
        beta, se = res["beta"], res["se"]
        rows.append(
            {
                "Candidate": cid,
                "Corr_MKT": float(r.corr(bundle.factors_3f["MKT"])),
                "Corr_VOL": float(r.corr(bundle.factors_3f["VOL"])),
                "Corr_MOM": float(r.corr(bundle.factors_3f["MOM"])),
                "Beta_MKT": float(beta[1]),
                "Beta_VOL": float(beta[2]),
                "Beta_MOM": float(beta[3]),
                "R2(%)": float(res["r2"] * 100.0),
                "FacAttrRet(%)": float(
                    (X_3f[:, 1:] @ beta[1:]).mean() * ann_days * 100.0
                ),
                "PureAlpha(%/yr)": float(beta[0] * ann_days * 100.0),
                "Alpha_HAC_t": float(beta[0] / se[0]),
            }
        )
    df_fac = pd.DataFrame(rows).set_index("Candidate")
    if tab_dir is not None:
        df_fac.to_csv(tab_dir / "sec4_3factor_attribution_24.csv")
    return df_fac


def compute_ff6_and_sector_exposure(
    bundle: V7DataBundle,
    all_entries: dict[str, dict[str, Any]],
    focus_names: Iterable[str],
    warmup_days: int = 252,
    ann_days: int = 252,
    tab_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute FF6 Newey-West OLS regression table and 12-sector average net exposure table."""
    oos_dates = bundle.dates[warmup_days:]
    ff6_oos = bundle.ff6_df.reindex(oos_dates).fillna(0.0)
    f_cols = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
    X_ff = np.column_stack(
        [np.ones(len(oos_dates)), ff6_oos[f_cols].to_numpy(dtype=np.float64)]
    )

    ff_rows = []
    sec_rows = []
    uniq_secs = sorted(set(bundle.sectors))
    sec_masks = {s: (bundle.sectors == s) for s in uniq_secs}

    for fn in focus_names:
        if fn not in all_entries:
            continue
        ent = all_entries[fn]
        r_oos = ent["bt"]["net"].iloc[warmup_days:].to_numpy(dtype=np.float64)
        ols = newey_west_ols(r_oos, X_ff, max_lag=5)
        ff_rows.append(
            {
                "Strategy": fn,
                "FF6_Alpha(%/yr)": round(float(ols["beta"][0] * ann_days * 100.0), 2),
                "Alpha_t": round(float(ols["tstat"][0]), 2),
                "MKT": round(float(ols["beta"][1]), 3),
                "MKT_t": round(float(ols["tstat"][1]), 2),
                "SMB": round(float(ols["beta"][2]), 3),
                "HML": round(float(ols["beta"][3]), 3),
                "RMW": round(float(ols["beta"][4]), 3),
                "CMA": round(float(ols["beta"][5]), 3),
                "MOM": round(float(ols["beta"][6]), 3),
                "R2": round(float(ols["r2"]), 3),
            }
        )
        W_mat = ent.get("weights")
        if W_mat is not None:
            W_oos = W_mat[warmup_days:]
            row_s: dict[str, Any] = {"Strategy": fn}
            for sname, smask in sec_masks.items():
                row_s[sname] = round(
                    float(W_oos[:, smask].sum(axis=1).mean() * 100.0), 2
                )
            sec_rows.append(row_s)

    ff_df = pd.DataFrame(ff_rows)
    sec_df = pd.DataFrame(sec_rows)
    if tab_dir is not None:
        ff_df.to_csv(tab_dir / "tab08_factor_attribution_ff6.csv", index=False)
        sec_df.to_csv(tab_dir / "tab08_sector_exposure.csv", index=False)
    return ff_df, sec_df
