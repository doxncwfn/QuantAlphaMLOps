"""Load aligned OOS predictions, cached risk matrices, FF6 factors, and SIC sectors using Polars."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

from v27.config import EXPERIMENTS_V7, DataConfig, PathConfig, TrainConfig
from v27.portfolio.neutralization import renorm_legs


@dataclass
class V7DataBundle:
    """Aligned daily panel shared across diagnostics, strategies, and backtesting."""

    dates: pd.DatetimeIndex
    security_ids: np.ndarray
    scores: dict[str, pd.DataFrame]
    ret1d_df: pd.DataFrame
    ret5d_fwd_df: pd.DataFrame
    fold_ser: pd.Series
    vol20_df: pd.DataFrame
    vol60_df: pd.DataFrame
    sigma_prev_df: pd.DataFrame
    beta60_df: pd.DataFrame
    mom252_df: pd.DataFrame
    adv20_df: pd.DataFrame
    sectors: np.ndarray
    factors_3f: pd.DataFrame
    ff6_df: pd.DataFrame
    ew_mkt_1d: pd.Series


def sic_to_sector12(sic_val: object) -> str:
    """Map 4-digit SIC code onto 12 Fama-French-inspired sector buckets."""
    if sic_val is None:
        return "Other_Services"
    try:
        s = int(str(sic_val).strip()[:4])
    except (TypeError, ValueError):
        return "Other_Services"
    if (
        (100 <= s <= 999)
        or (2000 <= s <= 2399)
        or (2700 <= s <= 2749)
        or (2770 <= s <= 2799)
        or (3100 <= s <= 3199)
        or (3940 <= s <= 3989)
    ):
        return "Consumer_NonDurables"
    if (
        (2500 <= s <= 2519)
        or (2590 <= s <= 2599)
        or (3630 <= s <= 3659)
        or s in (3710, 3711, 3714, 3716, 3750, 3751, 3792)
        or (3900 <= s <= 3939)
        or (3990 <= s <= 3999)
    ):
        return "Consumer_Durables"
    if (
        (2520 <= s <= 2589)
        or (2600 <= s <= 2699)
        or (2750 <= s <= 2769)
        or (3000 <= s <= 3099)
        or (3200 <= s <= 3569)
        or (3580 <= s <= 3629)
        or (3700 <= s <= 3709)
        or s in (3712, 3713, 3715)
        or (3717 <= s <= 3749)
        or (3752 <= s <= 3791)
        or (3793 <= s <= 3799)
        or (3830 <= s <= 3839)
        or (3860 <= s <= 3899)
    ):
        return "Manufacturing_Ind"
    if (1200 <= s <= 1399) or (2900 <= s <= 2999):
        return "Energy"
    if (2800 <= s <= 2829) or (2840 <= s <= 2899):
        return "Chemicals_Materials"
    if (
        (3570 <= s <= 3579)
        or (3660 <= s <= 3692)
        or (3694 <= s <= 3699)
        or (3810 <= s <= 3829)
        or (7370 <= s <= 7379)
    ):
        return "Tech_BusinessEquip"
    if 4800 <= s <= 4899:
        return "Telecom_Comm"
    if 4900 <= s <= 4949:
        return "Utilities"
    if (5000 <= s <= 5999) or (7200 <= s <= 7299) or (7600 <= s <= 7699):
        return "Retail_Wholesale"
    if (2830 <= s <= 2839) or s == 3693 or (3840 <= s <= 3859) or (8000 <= s <= 8099):
        return "Healthcare"
    if 6000 <= s <= 6999:
        return "Financials_RealEstate"
    return "Other_Services"


def _scatter_panel(
    dt_ns: np.ndarray,
    sids: np.ndarray,
    vals: np.ndarray,
    dates_ns: np.ndarray,
    sec_ids: np.ndarray,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Vectorized coordinate scatter replacing slow pandas multi-million-row pivot."""
    r = np.searchsorted(dates_ns, dt_ns)
    c = np.searchsorted(sec_ids, sids)
    ok = (r < len(dates_ns)) & (c < len(sec_ids))
    ok[ok] &= (dates_ns[r[ok]] == dt_ns[ok]) & (sec_ids[c[ok]] == sids[ok])
    mat = np.full((len(dates_ns), len(sec_ids)), np.nan, dtype=np.float64)
    mat[r[ok], c[ok]] = vals[ok]
    return pd.DataFrame(mat, index=dates, columns=sec_ids, copy=False)


def load_v7_data_bundle(
    qh_dir: Path | None = None,
    warehouse_path: Path | None = None,
    min_stocks: int | None = None,
    vol_floor: float | None = None,
) -> V7DataBundle:
    """Load OOS predictions via Polars, strictly lagged (t-1) risk matrices, 3F/FF6 factors, and sectors."""
    paths_cfg = PathConfig()
    qh_dir = qh_dir or paths_cfg.qh_v7_dir
    warehouse_path = warehouse_path or paths_cfg.warehouse_path
    min_stocks = TrainConfig().min_stocks if min_stocks is None else min_stocks
    vol_floor = DataConfig().vol_floor if vol_floor is None else vol_floor

    t0 = time.time()
    preds_map = {
        e: (
            pl.read_parquet(qh_dir / f"predictions_{e}.parquet")
            .drop_nulls(subset=["score", "label"])
            .filter(pl.col("score").is_not_nan() & pl.col("label").is_not_nan())
            .with_columns(pl.col("date").cast(pl.Datetime("ns")))
        )
        for e in EXPERIMENTS_V7
    }

    base_pl = preds_map["raw_t1"]
    counts = (
        base_pl.group_by("date").len().filter(pl.col("len") >= min_stocks).sort("date")
    )
    valid_dates_ns = counts["date"].to_numpy()
    dates = pd.DatetimeIndex(valid_dates_ns)
    sec_ids = np.sort(base_pl["security_id"].unique().to_numpy()).astype(np.int64)

    fold_df_pl = base_pl.group_by("date").agg(pl.col("fold").first()).sort("date")
    fold_ser = pd.Series(
        fold_df_pl["fold"].to_list(),
        index=pd.DatetimeIndex(fold_df_pl["date"].to_numpy()),
    ).reindex(dates)

    ret1d_df = _scatter_panel(
        base_pl["date"].to_numpy(),
        base_pl["security_id"].to_numpy(),
        base_pl["label"].to_numpy().astype(np.float64),
        valid_dates_ns,
        sec_ids,
        dates,
    )

    scores: dict[str, pd.DataFrame] = {}
    for exp_name, df_pl in preds_map.items():
        scores[exp_name] = _scatter_panel(
            df_pl["date"].to_numpy(),
            df_pl["security_id"].to_numpy(),
            df_pl["score"].to_numpy().astype(np.float64),
            valid_dates_ns,
            sec_ids,
            dates,
        )

    pt = torch.load(
        qh_dir / PathConfig.cache_filename, map_location="cpu", weights_only=False
    )
    cal = pd.to_datetime(pt["calendar"]).to_numpy()
    active_ids = pt["active_ids"]
    feats = pt["FEATURES"]
    X = pt["X_dense"]
    Y = pt["Y_dense"]

    cal_idx = np.searchsorted(cal, dates.to_numpy())
    id_idx = np.searchsorted(active_ids, sec_ids)

    def _lag_feat(fname: str) -> pd.DataFrame:
        fi = feats.index(fname)
        arr = X[np.ix_(cal_idx - 1, id_idx, [fi])][:, :, 0].astype(np.float64)
        return pd.DataFrame(arr, index=dates, columns=sec_ids)

    vol20_df = _lag_feat("vol_20")
    vol60_df = _lag_feat("vol_60")
    sigma_prev_df = np.maximum(np.maximum(vol20_df, vol60_df), vol_floor)
    beta60_df = _lag_feat("beta_60")
    mom252_df = _lag_feat("mom_252_20")
    log_dv20_df = _lag_feat("log_dollar_volume_20")
    adv20_df = np.expm1(log_dv20_df.clip(lower=0.0, upper=28.0)).clip(lower=1e6)

    Y_sub = Y[:, id_idx].astype(np.float64)
    r5_mat = np.full((len(dates), len(sec_ids)), np.nan, dtype=np.float64)
    for k_i, ci in enumerate(cal_idx):
        if ci + 4 < Y_sub.shape[0]:
            seg = Y_sub[ci : ci + 5]
            ok = np.isfinite(seg).all(axis=0)
            r5_mat[k_i, ok] = np.prod(1.0 + seg[:, ok], axis=0) - 1.0
    ret5d_fwd_df = pd.DataFrame(r5_mat, index=dates, columns=sec_ids)

    ew_mkt_1d = ret1d_df.mean(axis=1).fillna(0.0)
    r_clean = ret1d_df.fillna(0.0)
    v_rk = sigma_prev_df.where(ret1d_df.notna()).rank(axis=1, pct=True)
    vol_fac = (
        renorm_legs(
            ((v_rk > 0.80).astype(float) - (v_rk <= 0.20).astype(float)).fillna(0.0)
        )
        * r_clean
    ).sum(axis=1)
    m_rk = mom252_df.where(ret1d_df.notna()).rank(axis=1, pct=True)
    mom_fac = (
        renorm_legs(
            ((m_rk > 0.80).astype(float) - (m_rk <= 0.20).astype(float)).fillna(0.0)
        )
        * r_clean
    ).sum(axis=1)
    factors_3f = pd.DataFrame(
        {"MKT": ew_mkt_1d, "VOL": vol_fac, "MOM": mom_fac}, index=dates
    )

    sectors = np.array(["Other_Services"] * len(sec_ids), dtype=object)
    ff6_df = pd.DataFrame(
        0.0, index=dates, columns=["mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]
    )
    if warehouse_path.exists():
        import duckdb

        con = duckdb.connect(str(warehouse_path), read_only=True)
        try:
            d_min, d_max = dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")
            df_ff_pl = con.sql(
                f"SELECT date AS dt, mktrf, smb, hml, rmw, cma, umd, rf FROM silver.market_factor "
                f"WHERE date BETWEEN DATE '{d_min}' AND DATE '{d_max}' ORDER BY date"
            ).pl()
            ff_idx = pd.to_datetime(df_ff_pl["dt"].to_numpy())
            ff6_df = (
                df_ff_pl.drop("dt")
                .to_pandas()
                .set_axis(ff_idx)
                .reindex(dates)
                .ffill()
                .fillna(0.0)
            )

            df_sec_pl = con.sql(
                "SELECT security_id, siccd FROM ("
                "  SELECT security_id, siccd, ROW_NUMBER() OVER (PARTITION BY security_id ORDER BY valid_to DESC) AS rn "
                "  FROM silver.security_history"
                ") WHERE rn = 1"
            ).pl()
            sec_map = {
                int(sid): sic_to_sector12(sic) for sid, sic in df_sec_pl.iter_rows()
            }
            sectors = np.array(
                [sec_map.get(int(s), "Other_Services") for s in sec_ids], dtype=object
            )
        finally:
            con.close()

    print(
        f"[Load] V7 Bundle (Polars): {len(dates)} days ({dates[0].strftime('%Y-%m-%d')}..{dates[-1].strftime('%Y-%m-%d')}) "
        f"x {len(sec_ids)} stocks in {time.time() - t0:.1f}s"
    )
    return V7DataBundle(
        dates=dates,
        security_ids=sec_ids,
        scores=scores,
        ret1d_df=ret1d_df,
        ret5d_fwd_df=ret5d_fwd_df,
        fold_ser=fold_ser,
        vol20_df=vol20_df,
        vol60_df=vol60_df,
        sigma_prev_df=sigma_prev_df,
        beta60_df=beta60_df,
        mom252_df=mom252_df,
        adv20_df=adv20_df,
        sectors=sectors,
        factors_3f=factors_3f,
        ff6_df=ff6_df,
        ew_mkt_1d=ew_mkt_1d,
    )
