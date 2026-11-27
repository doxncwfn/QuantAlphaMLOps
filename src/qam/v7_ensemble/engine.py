"""Core v7 data loader, 24-candidate strategy constructor, audit engine, and consolidated execution."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXPERIMENTS_V7: tuple[str, ...] = ("raw_t1", "risk_t1", "risk_t5")
STRATEGIES_V7: tuple[str, ...] = (
    "rank_RAW",
    "top10_RAW",
    "top10_InvVol",
    "top10_Buf10_20",
    "top10_InvVol_Buf",
    "top10_BetaNeu_InvVol_Buf",
    "top10_HOLD5",
    "top10_InvVol_HOLD5",
)
CANDIDATES_24: tuple[str, ...] = tuple(
    f"{e}__{s}" for e in EXPERIMENTS_V7 for s in STRATEGIES_V7
)
CANDIDATES_1D_16: tuple[str, ...] = tuple(
    f"{e}__{s}" for e in ("raw_t1", "risk_t1") for s in STRATEGIES_V7
)

ANN_DAYS = 252.0
BASELINE_COST_BPS = 2.0
COST_GRID_BPS: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 15.0)
SHORT_BORROW_ANN = 0.0050
IMPACT_ETA = 0.10
BOOK_SIZE = 20_000_000.0
MIN_STOCKS = 20


@dataclass
class V7DataBundle:
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


def renorm_legs(w: pd.DataFrame) -> pd.DataFrame:
    """Normalize long leg sum = +1.0 and short leg sum = -1.0 (gross = 2.0)."""
    W = np.nan_to_num(w.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    pos = np.maximum(W, 0.0)
    neg = np.maximum(-W, 0.0)
    ps = pos.sum(axis=1, keepdims=True)
    ns = neg.sum(axis=1, keepdims=True)
    pos = np.where(ps > 1e-12, pos / np.maximum(ps, 1e-12), 0.0)
    neg = np.where(ns > 1e-12, neg / np.maximum(ns, 1e-12), 0.0)
    out = np.where((ps > 1e-12) & (ns > 1e-12), pos - neg, 0.0)
    return pd.DataFrame(out, index=w.index, columns=w.columns)


def neutralize_beta(sc_df: pd.DataFrame, beta_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional OLS residualization of prediction scores against lagged beta_60."""
    S = sc_df.to_numpy(dtype=np.float64)
    B = beta_df.to_numpy(dtype=np.float64)
    out = np.full_like(S, np.nan)
    valid = np.isfinite(S) & np.isfinite(B)
    for t in range(S.shape[0]):
        v = valid[t]
        if v.sum() < MIN_STOCKS:
            out[t] = S[t]
            continue
        sv, bv = S[t, v], B[t, v]
        bc = bv - bv.mean()
        denom = float(np.dot(bc, bc))
        if denom > 1e-12:
            slope = float(np.dot(sv - sv.mean(), bc) / denom)
            out[t, v] = sv - slope * bc
        else:
            out[t, v] = sv
    return pd.DataFrame(out, index=sc_df.index, columns=sc_df.columns)


def strat_rank_raw(sc_df: pd.DataFrame) -> pd.DataFrame:
    rk = sc_df.rank(axis=1)
    return renorm_legs(rk.sub(rk.mean(axis=1), axis=0).where(sc_df.notna()).fillna(0.0))


def strat_top10_raw(sc_df: pd.DataFrame, q: float = 0.10) -> pd.DataFrame:
    p = sc_df.rank(axis=1, pct=True)
    raw = (
        ((p > 1.0 - q).astype(float) - (p <= q).astype(float))
        .where(sc_df.notna())
        .fillna(0.0)
    )
    return renorm_legs(raw)


def strat_top10_invvol(
    sc_df: pd.DataFrame, inv_vol_df: pd.DataFrame, q: float = 0.10
) -> pd.DataFrame:
    p = sc_df.rank(axis=1, pct=True)
    raw = (
        ((p > 1.0 - q).astype(float) - (p <= q).astype(float))
        .where(sc_df.notna())
        .fillna(0.0)
    )
    return renorm_legs(raw * inv_vol_df.where(sc_df.notna()).fillna(0.0))


def strat_top10_buf(
    sc_df: pd.DataFrame,
    inv_vol_df: pd.DataFrame | None = None,
    q_in: float = 0.10,
    q_out: float = 0.20,
) -> pd.DataFrame:
    """10/20 hysteresis buffer portfolio construction matching dacn-stockranking v7."""
    pct = sc_df.rank(axis=1, pct=True).to_numpy(dtype=np.float64)
    ACT = sc_df.notna().to_numpy(dtype=bool)
    IV = (
        np.ones_like(pct)
        if inv_vol_df is None
        else np.nan_to_num(inv_vol_df.to_numpy(dtype=np.float64), nan=0.0)
    )
    T, N = pct.shape
    W = np.zeros((T, N), dtype=np.float64)
    l_st = np.zeros(N, dtype=bool)
    s_st = np.zeros(N, dtype=bool)
    for t in range(T):
        p, a, iv = pct[t], ACT[t], IV[t]
        p_ok = np.where(a & np.isfinite(p), p, 0.5)
        l_st = a & ((p_ok > 1.0 - q_in) | (l_st & (p_ok > 1.0 - q_out)))
        s_st = a & ((p_ok <= q_in) | (s_st & (p_ok <= q_out)))
        wl = np.where(l_st, iv, 0.0)
        ws = np.where(s_st, iv, 0.0)
        if wl.sum() > 1e-12:
            wl /= wl.sum()
        if ws.sum() > 1e-12:
            ws /= ws.sum()
        W[t] = wl - ws
    return pd.DataFrame(W, index=sc_df.index, columns=sc_df.columns)


def apply_overlapping_hold(
    w_target: pd.DataFrame, r_1d_df: pd.DataFrame, hold_days: int = 5
) -> pd.DataFrame:
    """Genuine `hold_days` overlapping tranche portfolio marked daily using actual 1-day returns."""
    W = np.nan_to_num(w_target.to_numpy(dtype=np.float64), nan=0.0)
    R = np.nan_to_num(r_1d_df.to_numpy(dtype=np.float64), nan=0.0)
    T, N = W.shape
    tranches = np.zeros((hold_days, N), dtype=np.float64)
    W_out = np.zeros((T, N), dtype=np.float64)
    for t in range(T):
        if t > 0:
            rt_p = R[t - 1]
            g = tranches @ rt_p
            tranches = (
                tranches * (1.0 + rt_p[None, :]) / np.maximum(1.0 + g[:, None], 0.25)
            )
        tranches[t % hold_days] = W[t]
        W_out[t] = tranches[: min(t + 1, hold_days)].mean(axis=0)
    return renorm_legs(
        pd.DataFrame(W_out, index=w_target.index, columns=w_target.columns)
    )


def build_8_strategies_for_score(
    sc_df: pd.DataFrame,
    r_1d_df: pd.DataFrame,
    sigma_prev_df: pd.DataFrame,
    beta60_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Construct all 8 Stock Ranking v7 portfolio strategies from a prediction score panel."""
    inv_vol = (1.0 / sigma_prev_df).where(sc_df.notna()).fillna(0.0)
    w_t10 = strat_top10_raw(sc_df)
    w_t10_iv = strat_top10_invvol(sc_df, inv_vol)
    sc_bn = neutralize_beta(sc_df, beta60_df)
    return {
        "rank_RAW": strat_rank_raw(sc_df),
        "top10_RAW": w_t10,
        "top10_InvVol": w_t10_iv,
        "top10_Buf10_20": strat_top10_buf(sc_df, None),
        "top10_InvVol_Buf": strat_top10_buf(sc_df, inv_vol),
        "top10_BetaNeu_InvVol_Buf": strat_top10_buf(sc_bn, inv_vol),
        "top10_HOLD5": apply_overlapping_hold(w_t10, r_1d_df, hold_days=5),
        "top10_InvVol_HOLD5": apply_overlapping_hold(w_t10_iv, r_1d_df, hold_days=5),
    }


def _sic_to_sector12(sic_val: object) -> str:
    if sic_val is None:
        return "Other"
    try:
        s = int(str(sic_val).strip()[:4])
    except Exception:  # noqa: BLE001
        return "Other"
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
        or (3710 <= s <= 3711)
        or (3714 <= s <= 3714)
        or (3716 <= s <= 3716)
        or (3750 <= s <= 3751)
        or (3792 <= s <= 3792)
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
        or (3712 <= s <= 3713)
        or (3715 <= s <= 3715)
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
    if (
        (2830 <= s <= 2839)
        or (3693 <= s <= 3693)
        or (3840 <= s <= 3859)
        or (8000 <= s <= 8099)
    ):
        return "Healthcare"
    if 6000 <= s <= 6999:
        return "Financials_RealEstate"
    return "Other_Services"


def load_v7_data_bundle(
    qh_dir: Path = Path("outputs/QH_v7"),
    warehouse_path: Path = Path("data/warehouse/russell1000.duckdb"),
) -> V7DataBundle:
    """Load OOS predictions, cached risk/feature matrices, sectors, and FF factors."""
    t0 = time.time()
    preds_map = {
        e: pd.read_parquet(qh_dir / f"predictions_{e}.parquet") for e in EXPERIMENTS_V7
    }

    # Filter out rows with missing score/label consistently with dacn-stockranking prep_mats
    clean_base = preds_map["raw_t1"].dropna(subset=["score", "label"]).copy()
    clean_base["date"] = pd.to_datetime(clean_base["date"])

    ret1d_df = clean_base.pivot(index="date", columns="security_id", values="label")
    keep = ret1d_df.notna().sum(axis=1) >= MIN_STOCKS
    ret1d_df = ret1d_df.loc[keep].sort_index()
    dates = pd.DatetimeIndex(ret1d_df.index)
    sec_ids = np.array(ret1d_df.columns, dtype=np.int64)
    ret1d_df = ret1d_df.astype(np.float64)

    fold_ser = clean_base.groupby("date")["fold"].first().reindex(dates)

    scores: dict[str, pd.DataFrame] = {}
    for e, df_e in preds_map.items():
        d_c = df_e.dropna(subset=["score", "label"]).copy()
        d_c["date"] = pd.to_datetime(d_c["date"])
        sc_p = d_c.pivot(index="date", columns="security_id", values="score").reindex(
            index=dates, columns=sec_ids
        )
        scores[e] = sc_p.astype(np.float64)

    # Extract strictly lagged (cal_idx - 1) risk features from dataset_top1000_f34_cache.pt
    pt = torch.load(
        qh_dir / "dataset_top1000_f34_cache.pt", map_location="cpu", weights_only=False
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
    sigma_prev_df = np.maximum(np.maximum(vol20_df, vol60_df), 0.005)
    beta60_df = _lag_feat("beta_60")
    mom252_df = _lag_feat("mom_252_20")
    log_dv20_df = _lag_feat("log_dollar_volume_20")
    adv20_df = np.expm1(log_dv20_df.clip(lower=0.0, upper=28.0)).clip(lower=1e6)

    # Exact 5-day compounded forward return raw_5d from Y_dense (matching risk_5d evaluation target!)
    Y_sub = Y[:, id_idx].astype(np.float64)
    r5_mat = np.full((len(dates), len(sec_ids)), np.nan, dtype=np.float64)
    for k_i, ci in enumerate(cal_idx):
        if ci + 4 < Y_sub.shape[0]:
            seg = Y_sub[ci : ci + 5]
            ok = np.isfinite(seg).all(axis=0)
            r5_mat[k_i, ok] = np.prod(1.0 + seg[:, ok], axis=0) - 1.0
    ret5d_fwd_df = pd.DataFrame(r5_mat, index=dates, columns=sec_ids)

    # Active-universe equal-weight market return & cross-sectional 3-factor proxy (MKT, VOL, MOM)
    ew_mkt_1d = ret1d_df.mean(axis=1).fillna(0.0)
    r_clean = ret1d_df.fillna(0.0)
    # Cross-sectional VOL factor: High-vol top 20% minus Low-vol bottom 20% (lagged sigma_prev)
    v_rk = sigma_prev_df.where(ret1d_df.notna()).rank(axis=1, pct=True)
    vol_fac = (
        renorm_legs(
            ((v_rk > 0.80).astype(float) - (v_rk <= 0.20).astype(float)).fillna(0.0)
        )
        * r_clean
    ).sum(axis=1)
    # Cross-sectional MOM factor: High mom_252_20 top 20% minus Low bottom 20%
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

    # Sectors & Fama-French 5 + Momentum from DuckDB warehouse
    sectors = np.array(["Other_Services"] * len(sec_ids), dtype=object)
    ff6_df = pd.DataFrame(
        0.0, index=dates, columns=["mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]
    )
    if warehouse_path.exists():
        import duckdb

        con = duckdb.connect(str(warehouse_path), read_only=True)
        try:
            d_min, d_max = dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")
            df_ff = con.execute(
                f"SELECT date AS dt, mktrf, smb, hml, rmw, cma, umd, rf FROM silver.market_factor "
                f"WHERE date BETWEEN DATE '{d_min}' AND DATE '{d_max}' ORDER BY date"
            ).df()
            df_ff["dt"] = pd.to_datetime(df_ff["dt"])
            ff6_df = df_ff.set_index("dt").reindex(dates).ffill().fillna(0.0)
            df_sec = con.execute(
                "SELECT security_id, siccd FROM ("
                "  SELECT security_id, siccd, ROW_NUMBER() OVER (PARTITION BY security_id ORDER BY valid_to DESC) AS rn "
                "  FROM silver.security_history"
                ") WHERE rn = 1"
            ).df()
            sec_map = {
                int(r.security_id): _sic_to_sector12(r.siccd)
                for r in df_sec.itertuples(index=False)
            }
            sectors = np.array(
                [sec_map.get(int(s), "Other_Services") for s in sec_ids], dtype=object
            )
        finally:
            con.close()

    print(
        f"[Load] V7 Bundle: {len(dates)} days ({dates[0].strftime('%Y-%m-%d')}..{dates[-1].strftime('%Y-%m-%d')}) x {len(sec_ids)} stocks in {time.time() - t0:.1f}s"
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


def consolidate_positions(
    alloc_schedule: pd.DataFrame,
    cand_weights: dict[str, np.ndarray],
) -> np.ndarray:
    """Construct consolidated stock positions W_{p,i,t} = sum_k a_{k,t} W_{k,i,t}."""
    cols = list(alloc_schedule.columns)
    first_w = next(iter(cand_weights.values()))
    T, N = first_w.shape
    assert alloc_schedule.shape[0] == T, (
        f"Schedule length {alloc_schedule.shape[0]} != {T}"
    )
    A = np.nan_to_num(alloc_schedule.to_numpy(dtype=np.float64), nan=0.0)
    W_cons = np.zeros((T, N), dtype=np.float64)
    for j, cname in enumerate(cols):
        aj = A[:, j]
        if float(np.abs(aj).max()) > 1e-12:
            W_cons += aj[:, None] * cand_weights[cname]
    return W_cons


def apply_portfolio_constraints(
    W_in: np.ndarray,
    beta60_np: np.ndarray | None = None,
    sectors: np.ndarray | None = None,
    adv20_np: np.ndarray | None = None,
    r1d_np: np.ndarray | None = None,
    max_single_w: float | None = 0.030,
    max_gross: float | None = None,
    max_abs_beta: float | None = None,
    max_sector_net: float | None = None,
    max_adv_frac: float | None = None,
    book_size: float = BOOK_SIZE,
) -> np.ndarray:
    """Apply post-consolidation position, gross, beta, sector, and ADV liquidity constraints."""
    W = np.nan_to_num(W_in.astype(np.float64), nan=0.0)
    T, _N = W.shape

    if max_single_w is not None:
        W = np.clip(W, -max_single_w, max_single_w)

    if max_sector_net is not None and sectors is not None:
        for sec in np.unique(sectors):
            sm = sectors == sec
            sec_net = W[:, sm].sum(axis=1)
            over_p = sec_net > max_sector_net
            if np.any(over_p):
                ex = sec_net[over_p] - max_sector_net
                sub = W[over_p][:, sm]
                pos = np.maximum(sub, 0.0)
                W[np.ix_(over_p, sm)] = (
                    sub
                    - (pos / np.maximum(pos.sum(axis=1, keepdims=True), 1e-12))
                    * ex[:, None]
                )
            over_n = sec_net < -max_sector_net
            if np.any(over_n):
                ex = (-sec_net[over_n]) - max_sector_net
                sub = W[over_n][:, sm]
                neg = np.maximum(-sub, 0.0)
                W[np.ix_(over_n, sm)] = (
                    sub
                    + (neg / np.maximum(neg.sum(axis=1, keepdims=True), 1e-12))
                    * ex[:, None]
                )

    if max_abs_beta is not None and beta60_np is not None:
        B = np.nan_to_num(beta60_np, nan=1.0)
        for t in range(T):
            w_t = W[t]
            b_t = B[t]
            act = np.abs(w_t) > 1e-10
            if act.sum() < 10:
                continue
            port_b = float(np.dot(w_t, b_t))
            if abs(port_b) > max_abs_beta:
                # Project beta excess orthogonally onto active dollar-neutral subspace
                target_b = math.copysign(max_abs_beta, port_b)
                b_a = b_t[act]
                bc = b_a - b_a.mean()
                denom = float(np.dot(bc, bc))
                if denom > 1e-10:
                    adj = ((port_b - target_b) / denom) * bc
                    w_t_new = w_t.copy()
                    w_t_new[act] = w_t[act] - adj
                    # Preserve original gross scale smoothly
                    g_old = float(np.abs(w_t).sum())
                    g_new = float(np.abs(w_t_new).sum())
                    if g_new > 1e-12:
                        w_t_new *= g_old / g_new
                    W[t] = w_t_new

    if max_single_w is not None:
        W = np.clip(W, -max_single_w, max_single_w)

    if max_gross is not None:
        g = np.abs(W).sum(axis=1)
        s = np.where(g > max_gross, max_gross / np.maximum(g, 1e-12), 1.0)
        W = W * s[:, None]

    if max_adv_frac is not None and adv20_np is not None and r1d_np is not None:
        ADV = np.maximum(np.nan_to_num(adv20_np, nan=50e6), 1e6)
        R = np.nan_to_num(r1d_np, nan=0.0)
        max_dw = (max_adv_frac * ADV) / book_size
        for t in range(1, T):
            gp = float(np.dot(W[t - 1], R[t - 1]))
            dr = W[t - 1] * (1.0 + R[t - 1]) / max(1.0 + gp, 0.25)
            dw = np.clip(W[t] - dr, -max_dw[t], max_dw[t])
            W[t] = dr + dw

    return W


def backtest_weights_daily(
    W: np.ndarray,
    r_1d_np: np.ndarray,
    dates: pd.DatetimeIndex,
    cost_bps: float = BASELINE_COST_BPS,
    adv20_np: np.ndarray | None = None,
    sigma_np: np.ndarray | None = None,
    rf_np: np.ndarray | None = None,
    book_size: float = BOOK_SIZE,
    inception_zero_trade: bool = False,
) -> dict[str, pd.Series]:
    """Exact daily mark-to-market accounting matching dacn-stockranking & institutional extensions.

    Conventions:
      - `W[t]` is traded entering day t and earns `r_1d[t]`.
      - `drifted[t] = W[t-1] * (1 + r_1d[t-1]) / max(1 + gross[t-1], 1e-6)`
      - `traded[t] = sum_i |W[t, i] - drifted[t, i]|` (two-way on gross=2 book; half-turnover = 0.5*traded).
      - If `inception_zero_trade=False` (matching `dacn-stockranking.ipynb`), t=0 trade = sum|W[0]|;
        otherwise t=0 trade = 0.0.
    """
    W_c = np.nan_to_num(W.astype(np.float64), nan=0.0)
    R = np.nan_to_num(r_1d_np.astype(np.float64), nan=0.0)
    T, _ = W_c.shape

    gross = np.einsum("ti,ti->t", W_c, R)
    drifted = np.zeros_like(W_c)
    drifted[1:] = (W_c[:-1] * (1.0 + R[:-1])) / np.maximum(1.0 + gross[:-1, None], 1e-6)
    delta_w = np.abs(W_c - drifted)
    traded = delta_w.sum(axis=1)
    if inception_zero_trade:
        traded[0] = 0.0

    long_exp = np.maximum(W_c, 0.0).sum(axis=1)
    short_exp = np.maximum(-W_c, 0.0).sum(axis=1)
    gross_exp = long_exp + short_exp
    net_exp = long_exp - short_exp

    tc_linear = traded * (cost_bps / 1e4)
    net_linear = gross - tc_linear

    borrow = short_exp * (SHORT_BORROW_ANN / ANN_DAYS)
    rf_arr = (
        np.zeros(T, dtype=np.float64)
        if rf_np is None
        else np.nan_to_num(rf_np, nan=0.0)
    )
    fin = np.maximum(long_exp - 1.0, 0.0) * (
        np.maximum(rf_arr, 0.0) + 0.0050 / ANN_DAYS
    )
    if adv20_np is not None and sigma_np is not None:
        ADV = np.maximum(np.nan_to_num(adv20_np, nan=50e6), 1e6)
        SIG = np.clip(np.nan_to_num(sigma_np, nan=0.02), 0.003, 0.25)
        part = np.minimum((delta_w * book_size) / ADV, 1.0)
        impact = (delta_w * IMPACT_ETA * SIG * np.sqrt(part)).sum(axis=1)
    else:
        impact = np.zeros(T, dtype=np.float64)

    net_allin = gross - tc_linear - borrow - fin - impact

    abs_w = np.abs(W_c)
    norm_abs = abs_w / np.maximum(gross_exp[:, None], 1e-12)
    hhi = (norm_abs**2).sum(axis=1)
    eff_n = np.where(hhi > 1e-12, 1.0 / hhi, 0.0)

    return {
        "gross": pd.Series(gross, index=dates),
        "net": pd.Series(net_linear, index=dates),
        "net_allin": pd.Series(net_allin, index=dates),
        "traded": pd.Series(traded, index=dates),
        "tc": pd.Series(tc_linear, index=dates),
        "borrow": pd.Series(borrow, index=dates),
        "fin": pd.Series(fin, index=dates),
        "impact": pd.Series(impact, index=dates),
        "long_exp": pd.Series(long_exp, index=dates),
        "short_exp": pd.Series(short_exp, index=dates),
        "gross_exp": pd.Series(gross_exp, index=dates),
        "net_exp": pd.Series(net_exp, index=dates),
        "eff_n": pd.Series(eff_n, index=dates),
    }


def compute_legacy_brain_stats(
    bt: dict[str, pd.Series], cost_bps: float = BASELINE_COST_BPS
) -> dict[str, float]:
    """Legacy BRAIN arithmetic stats matching grand_summary_v7.csv for reference parity."""
    g = bt["gross"]
    tr = bt["traded"]
    net = g - tr * (cost_bps / 1e4)
    sh = float(g.mean() / (g.std() + 1e-12) * math.sqrt(ANN_DAYS))
    ret = float(g.mean() * ANN_DAYS)
    turn = float(0.5 * tr.mean())
    fit = sh * math.sqrt(abs(ret) / max(turn, 0.125))
    cum = g.cumsum()
    dd = float((np.maximum(cum.cummax(), 0.0) - cum).max())
    mar = float(g.sum() / (tr.sum() + 1e-12) * 1e4)
    nsh = float(net.mean() / (net.std() + 1e-12) * math.sqrt(ANN_DAYS))
    return {
        "Legacy_GrossSh": sh,
        "Legacy_Turn(%)": turn * 100.0,
        "Legacy_Fitness": fit,
        "Legacy_Ret(%)": ret * 100.0,
        "Legacy_ArithDD(%)": dd * 100.0,
        "Legacy_Margin(bps)": mar,
        f"Legacy_NetSh@{cost_bps:g}bps": nsh,
    }
