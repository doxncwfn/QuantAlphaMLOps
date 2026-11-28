"""Section 3 Audit & Section 4 Preliminary Complementarity, Overlap, Risk & Regime Diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

from src.qam.v7_ensemble.engine import (
    ANN_DAYS,
    BASELINE_COST_BPS,
    CANDIDATES_24,
    EXPERIMENTS_V7,
    STRATEGIES_V7,
    V7DataBundle,
    backtest_weights_daily,
    build_8_strategies_for_score,
    compute_legacy_brain_stats,
    renorm_legs,
)


def _compounded_dd_episodes(r: pd.Series) -> dict[str, float]:
    arr = r.dropna().to_numpy(dtype=np.float64)
    nav = np.cumprod(1.0 + np.clip(arr, -0.95, 5.0))
    hwm = np.maximum.accumulate(np.maximum(nav, 1.0))
    dd = nav / hwm - 1.0
    mdd = float(-dd.min())
    ulcer = float(np.sqrt(np.mean((dd * 100.0) ** 2)))
    uw = dd < -1e-9
    durs, recs = [], []
    i, n = 0, len(dd)
    while i < n:
        if uw[i]:
            st = i
            while i < n and uw[i]:
                i += 1
            en = i
            tr = st + int(np.argmin(dd[st:en]))
            durs.append(en - st)
            recs.append(min(n, en) - tr)
        else:
            i += 1
    return {
        "Comp_MaxDD(%)": mdd * 100.0,
        "MaxDD_Dur(d)": float(max(durs)) if durs else 0.0,
        "MaxRecov(d)": float(max(recs)) if recs else 0.0,
        "UlcerIdx": ulcer,
    }


def _non_overlapping_rebal_weights(
    w_daily_df: pd.DataFrame, r_1d_df: pd.DataFrame, stride: int = 5
) -> pd.DataFrame:
    """Rebalance only every `stride` days and drift weights daily with realized 1d returns in between."""
    W_tar = np.nan_to_num(w_daily_df.to_numpy(dtype=np.float64), nan=0.0)
    R = np.nan_to_num(r_1d_df.to_numpy(dtype=np.float64), nan=0.0)
    T, N = W_tar.shape
    W_out = np.zeros((T, N), dtype=np.float64)
    for t in range(T):
        if t % stride == 0:
            W_out[t] = W_tar[t]
        else:
            gp = float(np.dot(W_out[t - 1], R[t - 1]))
            W_out[t] = W_out[t - 1] * (1.0 + R[t - 1]) / max(1.0 + gp, 0.25)
    return pd.DataFrame(W_out, index=w_daily_df.index, columns=w_daily_df.columns)


def run_section3_audit(
    bundle: V7DataBundle,
    qh_dir: Path,
    tab_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, pd.Series]], pd.DataFrame]:
    """Construct all 24 candidate weight matrices, audit parity vs grand_summary_v7, and audit 1d vs 5d."""
    r1_np = bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64)
    adv_np = bundle.adv20_df.to_numpy(dtype=np.float64)
    sig_np = bundle.sigma_prev_df.to_numpy(dtype=np.float64)
    rf_np = bundle.ff6_df["rf"].to_numpy(dtype=np.float64)

    gs_v7 = pd.read_csv(qh_dir / "grand_summary_v7.csv").set_index(
        ["experiment", "portfolio"]
    )

    cand_weights: dict[str, np.ndarray] = {}
    cand_bts: dict[str, dict[str, pd.Series]] = {}
    audit_rows = []

    for exp in EXPERIMENTS_V7:
        w_dict = build_8_strategies_for_score(
            bundle.scores[exp], bundle.ret1d_df, bundle.sigma_prev_df, bundle.beta60_df
        )
        for sname in STRATEGIES_V7:
            cid = f"{exp}__{sname}"
            W_np = w_dict[sname].to_numpy(dtype=np.float64)
            cand_weights[cid] = W_np
            bt = backtest_weights_daily(
                W_np,
                r1_np,
                bundle.dates,
                cost_bps=BASELINE_COST_BPS,
                adv20_np=adv_np,
                sigma_np=sig_np,
                rf_np=rf_np,
                inception_zero_trade=False,
            )
            cand_bts[cid] = bt
            leg = compute_legacy_brain_stats(bt, cost_bps=BASELINE_COST_BPS)

            rn = bt["net"]
            nav = (1.0 + rn.clip(lower=-0.95)).cumprod()
            n_yrs = len(rn) / ANN_DAYS
            cagr = float((nav.iloc[-1] ** (1.0 / n_yrs) - 1.0) * 100.0)
            dd_i = _compounded_dd_episodes(rn)
            gs_row = gs_v7.loc[(exp, sname)]

            audit_rows.append(
                {
                    "Candidate": cid,
                    "Experiment": exp,
                    "Strategy": sname,
                    "Legacy_GrossSh": leg["Legacy_GrossSh"],
                    "GS_v7_GrossSh": float(gs_row["Sharpe"]),
                    "Legacy_NetSh2": leg["Legacy_NetSh@2bps"],
                    "GS_v7_NetSh2": float(gs_row["NetSharpe@2bps"]),
                    "Legacy_Turn(%)": leg["Legacy_Turn(%)"],
                    "GS_v7_Turn(%)": float(gs_row["Turnover(%)"]),
                    "Legacy_ArithDD(%)": leg["Legacy_ArithDD(%)"],
                    "Comp_NetSh2": float(
                        rn.mean() / (rn.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS)
                    ),
                    "Comp_CAGR2(%)": cagr,
                    "Comp_MaxDD(%)": dd_i["Comp_MaxDD(%)"],
                    "Comp_Calmar2": cagr / max(dd_i["Comp_MaxDD(%)"], 0.01),
                }
            )

    df_audit24 = pd.DataFrame(audit_rows).set_index("Candidate")
    df_audit24.to_csv(tab_dir / "sec3_audit_24_candidates_parity.csv")

    # Section 3 Signal Horizon Audit: 1d vs 5d predictive IC & Top/Bottom Hit Rate + risk_t5 horizon accounting
    sig_rows = []
    for exp in EXPERIMENTS_V7:
        sc = bundle.scores[exp]
        act = sc.notna()
        ic_1d = (
            sc.rank(axis=1)
            .corrwith(bundle.ret1d_df.where(act).rank(axis=1), axis=1)
            .dropna()
        )
        ic_5d = (
            sc.rank(axis=1)
            .corrwith(bundle.ret5d_fwd_df.where(act).rank(axis=1), axis=1)
            .dropna()
        )
        p = sc.rank(axis=1, pct=True)
        w_t10 = renorm_legs(
            ((p > 0.90).astype(float) - (p <= 0.10).astype(float))
            .where(act)
            .fillna(0.0)
        )
        spread_1d = (w_t10 * bundle.ret1d_df.fillna(0.0)).sum(axis=1)
        spread_5d = (w_t10 * bundle.ret5d_fwd_df.fillna(0.0)).sum(axis=1).iloc[:-4]
        sig_rows.append(
            {
                "Experiment": exp,
                "IC_vs_1d": float(ic_1d.mean()),
                "ICIR_1d": float(
                    ic_1d.mean() / (ic_1d.std() + 1e-12) * math.sqrt(ANN_DAYS)
                ),
                "TopBot10_Hit_1d(%)": float((spread_1d > 0).mean() * 100.0),
                "IC_vs_5d": float(ic_5d.mean()),
                "ICIR_5d_OverlapAdj": float(
                    ic_5d.mean() / (ic_5d.std() + 1e-12) * math.sqrt(ANN_DAYS / 5.0)
                ),
                "TopBot10_Hit_5d(%)": float((spread_5d > 0).mean() * 100.0),
            }
        )
    df_sig_audit = pd.DataFrame(sig_rows)
    df_sig_audit.to_csv(tab_dir / "sec3_signal_1d_vs_5d_audit.csv", index=False)

    # Compare risk_t5 accounting modes: Daily Rebal (1d MTM) vs Non-Overlapping 5d Rebal (1d MTM) vs HOLD5 (1d MTM) vs Invalid 5d-label MTM
    t5_rows = []
    sc5 = bundle.scores["risk_t5"]
    inv_vol = (1.0 / bundle.sigma_prev_df).where(sc5.notna()).fillna(0.0)
    from src.qam.v7_ensemble.engine import neutralize_beta, strat_top10_buf

    w_t5_bn_daily = strat_top10_buf(neutralize_beta(sc5, bundle.beta60_df), inv_vol)
    w_t5_bn_5d_nonov = _non_overlapping_rebal_weights(
        w_t5_bn_daily, bundle.ret1d_df, stride=5
    )
    from src.qam.v7_ensemble.engine import apply_overlapping_hold

    w_t5_bn_hold5 = apply_overlapping_hold(w_t5_bn_daily, bundle.ret1d_df, hold_days=5)
    for tag, W_m, ret_mat, lbl in [
        (
            "DailyRebal_Realized1d",
            w_t5_bn_daily.to_numpy(),
            r1_np,
            "Valid daily 1d MTM",
        ),
        (
            "NonOverlap5d_Realized1d",
            w_t5_bn_5d_nonov.to_numpy(),
            r1_np,
            "Valid 5d-stride MTM",
        ),
        (
            "OverlapHOLD5_Realized1d",
            w_t5_bn_hold5.to_numpy(),
            r1_np,
            "Valid 5-tranche MTM",
        ),
        (
            "INVALID_5dFwdLabel_MTM",
            w_t5_bn_daily.to_numpy(),
            np.nan_to_num(bundle.ret5d_fwd_df.to_numpy(), nan=0.0),
            "DISTORTED (5d fwd as 1d)",
        ),
    ]:
        bt_m = backtest_weights_daily(
            W_m, ret_mat, bundle.dates, cost_bps=BASELINE_COST_BPS
        )
        rn = bt_m["net"]
        nav = (1.0 + rn.clip(lower=-0.95)).cumprod()
        t5_rows.append(
            {
                "Mode": tag,
                "Note": lbl,
                "GrossSh": float(
                    bt_m["gross"].mean()
                    / (bt_m["gross"].std(ddof=1) + 1e-12)
                    * math.sqrt(ANN_DAYS)
                ),
                "NetSh@2bps": float(
                    rn.mean() / (rn.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS)
                ),
                "CAGR2(%)": float(
                    (nav.iloc[-1] ** (1.0 / (len(rn) / ANN_DAYS)) - 1.0) * 100.0
                ),
                "Turn_1Way(%)": float(0.5 * bt_m["traded"].mean() * 100.0),
                "CompMaxDD(%)": _compounded_dd_episodes(rn)["Comp_MaxDD(%)"],
            }
        )
    df_t5_audit = pd.DataFrame(t5_rows)
    df_t5_audit.to_csv(
        tab_dir / "sec3_risk_t5_horizon_accounting_audit.csv", index=False
    )

    return cand_weights, cand_bts, df_audit24


def run_section4_preliminary(
    bundle: V7DataBundle,
    cand_weights: dict[str, np.ndarray],
    cand_bts: dict[str, dict[str, pd.Series]],
    tab_dir: Path,
    fig_dir: Path,
) -> dict:
    """Section 4: Correlation, Position Overlap, Risk Contribution, 3-Factor & Regime Diagnostics."""
    cids = list(CANDIDATES_24)
    net_df = pd.DataFrame({c: cand_bts[c]["net"] for c in cids}, index=bundle.dates)
    mkt = bundle.factors_3f["MKT"]

    # 4.1 Return Correlations
    corr_p = net_df.corr(method="pearson")
    corr_s = net_df.corr(method="spearman")

    X = net_df.to_numpy(dtype=np.float64)
    N_c = len(cids)
    down_mat = np.eye(N_c)
    for i in range(N_c):
        for j in range(i + 1, N_c):
            m = (X[:, i] < 0.0) | (X[:, j] < 0.0)
            c = float(np.corrcoef(X[m, i], X[m, j])[0, 1]) if m.sum() > 20 else np.nan
            down_mat[i, j] = down_mat[j, i] = c
    corr_down = pd.DataFrame(down_mat, index=cids, columns=cids)

    ew_eq = net_df.mean(axis=1)
    ew_nav = (1.0 + ew_eq).cumprod()
    ew_dd = ew_nav / ew_nav.cummax().clip(lower=1.0) - 1.0
    large_dd_mask = ew_dd <= ew_dd.quantile(0.15)
    v20_mkt = mkt.rolling(20, min_periods=5).std().shift(1).bfill()
    hi_vol_mask = v20_mkt >= v20_mkt.quantile(0.80)

    corr_dd = net_df.loc[large_dd_mask].corr(method="pearson")
    corr_hivol = net_df.loc[hi_vol_mask].corr(method="pearson")

    corr_p.to_csv(tab_dir / "sec4_corr_pearson_24.csv")
    corr_s.to_csv(tab_dir / "sec4_corr_spearman_24.csv")
    corr_down.to_csv(tab_dir / "sec4_corr_downside_24.csv")
    corr_dd.to_csv(tab_dir / "sec4_corr_large_dd_24.csv")
    corr_hivol.to_csv(tab_dir / "sec4_corr_highvol_24.csv")

    # Underlying Score Correlations across raw_t1, risk_t1, risk_t5
    sc_corrs = pd.DataFrame(
        index=EXPERIMENTS_V7, columns=EXPERIMENTS_V7, dtype=np.float64
    )
    for e1 in EXPERIMENTS_V7:
        for e2 in EXPERIMENTS_V7:
            c_ser = (
                bundle.scores[e1]
                .corrwith(bundle.scores[e2], axis=1, method="spearman")
                .dropna()
            )
            sc_corrs.loc[e1, e2] = float(c_ser.mean())
    sc_corrs.to_csv(tab_dir / "sec4_model_score_rank_correlations.csv")

    # 4.1b & 4.2 Portfolio Weight Cosine Similarity & Position Overlap (Jaccard Long / Short / Overall)
    T_d = len(bundle.dates)
    cos_sim = np.eye(N_c)
    ov_long = np.eye(N_c)
    ov_short = np.eye(N_c)
    ov_all = np.eye(N_c)
    # Use every 5th day for fast exact O(T/5) holdings overlap across 24x24 pairs
    sl5 = slice(0, T_d, 5)
    W5_list = [cand_weights[c][sl5] for c in cids]
    W5_norms = [np.linalg.norm(W, axis=1) for W in W5_list]
    L5_list = [W > 1e-8 for W in W5_list]
    S5_list = [W < -1e-8 for W in W5_list]
    A5_list = [np.abs(W) > 1e-8 for W in W5_list]

    for i in range(N_c):
        for j in range(i + 1, N_c):
            dots = np.einsum("ti,ti->t", W5_list[i], W5_list[j])
            den = np.maximum(W5_norms[i] * W5_norms[j], 1e-12)
            cs = float(np.mean(dots / den))
            jl = float(
                np.mean(
                    (L5_list[i] & L5_list[j]).sum(axis=1)
                    / np.maximum((L5_list[i] | L5_list[j]).sum(axis=1), 1)
                )
            )
            js = float(
                np.mean(
                    (S5_list[i] & S5_list[j]).sum(axis=1)
                    / np.maximum((S5_list[i] | S5_list[j]).sum(axis=1), 1)
                )
            )
            ja = float(
                np.mean(
                    (A5_list[i] & A5_list[j]).sum(axis=1)
                    / np.maximum((A5_list[i] | A5_list[j]).sum(axis=1), 1)
                )
            )
            cos_sim[i, j] = cos_sim[j, i] = cs
            ov_long[i, j] = ov_long[j, i] = jl
            ov_short[i, j] = ov_short[j, i] = js
            ov_all[i, j] = ov_all[j, i] = ja

    df_wcos = pd.DataFrame(cos_sim, index=cids, columns=cids)
    df_ov_all = pd.DataFrame(ov_all, index=cids, columns=cids)
    df_ov_l = pd.DataFrame(ov_long, index=cids, columns=cids)
    df_ov_s = pd.DataFrame(ov_short, index=cids, columns=cids)
    df_wcos.to_csv(tab_dir / "sec4_weight_cosine_similarity_24.csv")
    df_ov_all.to_csv(tab_dir / "sec4_position_overlap_all_24.csv")
    df_ov_l.to_csv(tab_dir / "sec4_position_overlap_long_24.csv")
    df_ov_s.to_csv(tab_dir / "sec4_position_overlap_short_24.csv")

    # Pairwise summary table (all 276 pairs)
    pair_rows = []
    for i in range(N_c):
        for j in range(i + 1, N_c):
            ci, cj = cids[i], cids[j]
            pair_rows.append(
                {
                    "Cand_A": ci,
                    "Cand_B": cj,
                    "Ret_Pearson": float(corr_p.iloc[i, j]),
                    "Ret_Spearman": float(corr_s.iloc[i, j]),
                    "Downside_Corr": float(corr_down.iloc[i, j]),
                    "HighVol_Corr": float(corr_hivol.iloc[i, j]),
                    "LargeDD_Corr": float(corr_dd.iloc[i, j]),
                    "Weight_CosSim": float(cos_sim[i, j]),
                    "Hold_Jaccard": float(ov_all[i, j]),
                    "Long_Jaccard": float(ov_long[i, j]),
                    "Short_Jaccard": float(ov_short[i, j]),
                }
            )
    df_pairs = pd.DataFrame(pair_rows).sort_values("Ret_Pearson")
    df_pairs.to_csv(tab_dir / "sec4_pairwise_complementarity_276.csv", index=False)

    # Rolling 60d, 126d, 252d correlations for priority & low-corr pairs
    key_pairs = [
        ("raw_t1__top10_InvVol_Buf", "risk_t1__top10_BetaNeu_InvVol_Buf"),
        ("raw_t1__top10_Buf10_20", "risk_t1__top10_BetaNeu_InvVol_Buf"),
        ("raw_t1__top10_InvVol_Buf", "risk_t1__top10_Buf10_20"),
        ("raw_t1__top10_InvVol_HOLD5", "risk_t1__top10_BetaNeu_InvVol_Buf"),
        ("risk_t5__top10_BetaNeu_InvVol_Buf", "raw_t1__top10_InvVol_Buf"),
        ("risk_t5__rank_RAW", "risk_t1__top10_BetaNeu_InvVol_Buf"),
    ]
    roll_figs = {}
    for ca, cb in key_pairs:
        roll_figs[f"{ca}\nvs {cb}"] = (
            net_df[ca].rolling(60).corr(net_df[cb]),
            net_df[ca].rolling(126).corr(net_df[cb]),
            net_df[ca].rolling(252).corr(net_df[cb]),
        )

    # 4.3 Risk Contribution + Portfolio Exposures + 3-Factor Attribution (MKT, VOL, MOM)
    cov_ann = net_df.cov().to_numpy(dtype=np.float64) * ANN_DAYS
    w_eq24 = np.ones(N_c) / N_c
    p_vol24 = math.sqrt(float(w_eq24 @ cov_ann @ w_eq24))
    mcr24 = (cov_ann @ w_eq24) / max(p_vol24, 1e-12)
    ccr24 = w_eq24 * mcr24
    rc_pct24 = ccr24 / max(p_vol24, 1e-12) * 100.0

    # Lower tail dependence q=5%
    qs = np.quantile(X, 0.05, axis=0)
    t_ind = (X <= qs[None, :]).astype(np.float64)
    tail_dep = (t_ind.T @ t_ind) / max(T_d * 0.05, 1e-12)
    np.fill_diagonal(tail_dep, 1.0)
    pd.DataFrame(tail_dep, index=cids, columns=cids).to_csv(
        tab_dir / "sec4_tail_dependence_24.csv"
    )

    B60_np = np.nan_to_num(bundle.beta60_df.to_numpy(dtype=np.float64), nan=1.0)
    S60_np = np.nan_to_num(bundle.sigma_prev_df.to_numpy(dtype=np.float64), nan=0.02)
    M252_np = np.nan_to_num(bundle.mom252_df.to_numpy(dtype=np.float64), nan=0.0)
    uniq_sec = np.unique(bundle.sectors)
    sec_masks = [bundle.sectors == s for s in uniq_sec]
    X_3f = np.column_stack(
        [
            np.ones(T_d),
            bundle.factors_3f["MKT"].to_numpy(),
            bundle.factors_3f["VOL"].to_numpy(),
            bundle.factors_3f["MOM"].to_numpy(),
        ]
    )
    XtX_inv = np.linalg.pinv(X_3f.T @ X_3f)

    rc_rows = []
    fac_rows = []
    for i, cid in enumerate(cids):
        r = net_df[cid]
        bt = cand_bts[cid]
        W_c = cand_weights[cid]
        dd_i = _compounded_dd_episodes(r)
        ann_ret = float(r.mean() * ANN_DAYS * 100.0)
        ann_vol = float(r.std(ddof=1) * math.sqrt(ANN_DAYS) * 100.0)
        down = r[r < 0]
        down_dev = (
            math.sqrt(float((down**2).mean()) * ANN_DAYS) * 100.0 if len(down) else 0.0
        )
        q05 = float(r.quantile(0.05))
        cvar95 = float(r[r <= q05].mean() * 100.0)

        # Realized portfolio holdings exposures (beta_60, sigma_prev, mom_252, max sector net)
        port_b60 = float(np.mean(np.einsum("ti,ti->t", W_c, B60_np)))
        port_vol_exp = float(np.mean(np.einsum("ti,ti->t", W_c, S60_np)) * 100.0)
        port_mom_exp = float(np.mean(np.einsum("ti,ti->t", W_c, M252_np)) * 100.0)
        max_sec_net = float(
            max(np.abs(W_c[:, sm].sum(axis=1)).mean() for sm in sec_masks) * 100.0
        )

        rc_rows.append(
            {
                "Candidate": cid,
                "AnnRet2(%)": ann_ret,
                "AnnVol2(%)": ann_vol,
                "NetSh2": float(
                    r.mean() / (r.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS)
                ),
                "DownDev(%)": down_dev,
                "MCR_EW24(%)": mcr24[i] * 100.0,
                "RiskShare_EW24(%)": rc_pct24[i],
                "Comp_MaxDD(%)": dd_i["Comp_MaxDD(%)"],
                "MaxDD_Dur(d)": dd_i["MaxDD_Dur(d)"],
                "MaxRecov(d)": dd_i["MaxRecov(d)"],
                "CVaR95_1d(%)": cvar95,
                "Avg_TailDep": float((tail_dep[i].sum() - 1.0) / (N_c - 1)),
                "AvgGross": float(bt["gross_exp"].mean()),
                "AvgNet": float(bt["net_exp"].mean()),
                "Hold_Beta60": port_b60,
                "Hold_VolExp(%)": port_vol_exp,
                "Hold_MomExp(%)": port_mom_exp,
                "MaxAvgSecNet(%)": max_sec_net,
                "Turn_1Way(%)": float(0.5 * bt["traded"].mean() * 100.0),
            }
        )

        # 3-Factor (MKT, VOL, MOM) OLS with Newey-West lag 5
        y = r.to_numpy(dtype=np.float64)
        beta = XtX_inv @ (X_3f.T @ y)
        resid = y - X_3f @ beta
        Xe = X_3f * resid[:, None]
        S_nw = (Xe.T @ Xe) / T_d
        for lag in range(1, 6):
            w_l = 1.0 - lag / 6.0
            G = (Xe[lag:].T @ Xe[:-lag]) / T_d
            S_nw += w_l * (G + G.T)
        se = np.sqrt(np.maximum(np.diag(T_d * (XtX_inv @ S_nw @ XtX_inv)), 1e-18))
        r2 = 1.0 - float(np.sum(resid**2)) / max(
            float(np.sum((y - y.mean()) ** 2)), 1e-18
        )
        fac_attr_ret = float((X_3f[:, 1:] @ beta[1:]).mean() * ANN_DAYS * 100.0)
        fac_rows.append(
            {
                "Candidate": cid,
                "Corr_MKT": float(r.corr(bundle.factors_3f["MKT"])),
                "Corr_VOL": float(r.corr(bundle.factors_3f["VOL"])),
                "Corr_MOM": float(r.corr(bundle.factors_3f["MOM"])),
                "Beta_MKT": float(beta[1]),
                "Beta_VOL": float(beta[2]),
                "Beta_MOM": float(beta[3]),
                "R2(%)": r2 * 100.0,
                "FacAttrRet(%)": fac_attr_ret,
                "PureAlpha(%/yr)": float(beta[0] * ANN_DAYS * 100.0),
                "Alpha_HAC_t": float(beta[0] / se[0]),
            }
        )

    df_rc = pd.DataFrame(rc_rows).set_index("Candidate")
    df_fac = pd.DataFrame(fac_rows).set_index("Candidate")
    df_rc.to_csv(tab_dir / "sec4_risk_contribution_and_exposures_24.csv")
    df_fac.to_csv(tab_dir / "sec4_3factor_attribution_24.csv")

    # 4.4 Regime-Conditioned Performance across 24 candidates (strictly lagged causal regimes)
    mkt_nav = (1.0 + mkt).cumprod()
    is_bear = (
        mkt_nav.shift(1).bfill()
        < mkt_nav.rolling(200, min_periods=40).mean().shift(1).bfill()
    ) & (
        ((1.0 + mkt).rolling(63, min_periods=20).apply(np.prod, raw=True) - 1.0)
        .shift(1)
        .fillna(0.0)
        < 0.0
    )
    vol20_l = mkt.rolling(20, min_periods=5).std().shift(1).bfill()
    br_l = (
        (bundle.ret1d_df > 0)
        .mean(axis=1)
        .rolling(20, min_periods=5)
        .mean()
        .shift(1)
        .bfill()
    )
    mom_l = (
        bundle.factors_3f["MOM"].rolling(63, min_periods=20).sum().shift(1).fillna(0.0)
    )
    idx_d = bundle.dates
    regimes = {
        "Bull": ~is_bear,
        "Bear": is_bear,
        "Vol_Low": vol20_l <= vol20_l.quantile(1 / 3),
        "Vol_High": vol20_l >= vol20_l.quantile(2 / 3),
        "Breadth_Low": br_l <= br_l.quantile(1 / 3),
        "Breadth_High": br_l >= br_l.quantile(2 / 3),
        "Mom_Pos": mom_l > 0.0,
        "Mom_Neg": mom_l <= 0.0,
        "Euro_2011": (idx_d >= "2011-05-01") & (idx_d <= "2011-12-31"),
        "ChinaOil_15_16": (idx_d >= "2015-08-01") & (idx_d <= "2016-03-31"),
        "COVID_2020": (idx_d >= "2020-02-15") & (idx_d <= "2020-04-30"),
        "RateHike_2022": (idx_d >= "2022-01-01") & (idx_d <= "2022-10-31"),
        "AI_Bull_23_24": (idx_d >= "2023-01-01") & (idx_d <= "2024-12-31"),
    }
    reg_sh = pd.DataFrame(index=cids, columns=list(regimes.keys()), dtype=np.float64)
    for rn, mask in regimes.items():
        m = np.asarray(mask, dtype=bool)
        sub = net_df.iloc[m]
        for cid in cids:
            s = sub[cid]
            reg_sh.loc[cid, rn] = float(
                s.mean() / (s.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS)
            )
    reg_sh.to_csv(tab_dir / "sec4_regime_sharpes_24.csv")

    # ---------------- CHARTS ----------------
    # Fig 4.1: Return Correlation vs Position Overlap + Dendrogram
    fig, axes = plt.subplots(
        1, 3, figsize=(24, 7.6), gridspec_kw={"width_ratios": [1.1, 1.1, 0.9]}
    )
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr_p.to_numpy()), 0.0))
    np.fill_diagonal(dist, 0.0)
    Z_lk = linkage(squareform(dist, checks=False), method="ward")
    ord_i = leaves_list(Z_lk)
    ord_c = [cids[k] for k in ord_i]
    short_lbl = [
        c.replace("__top10_", "__")
        .replace("_BetaNeu_InvVol_Buf", "_BN_IV_B")
        .replace("_InvVol_", "_IV_")
        for c in ord_c
    ]

    im0 = axes[0].imshow(
        corr_p.loc[ord_c, ord_c].to_numpy(), cmap="RdBu_r", vmin=-0.3, vmax=1.0
    )
    axes[0].set_xticks(range(N_c))
    axes[0].set_xticklabels(short_lbl, rotation=90, fontsize=7)
    axes[0].set_yticks(range(N_c))
    axes[0].set_yticklabels(short_lbl, fontsize=7)
    axes[0].set_title("Daily Net Return Pearson Correlation (24 Candidates)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)

    im1 = axes[1].imshow(
        df_ov_all.loc[ord_c, ord_c].to_numpy(), cmap="YlGnBu", vmin=0.0, vmax=1.0
    )
    axes[1].set_xticks(range(N_c))
    axes[1].set_xticklabels(short_lbl, rotation=90, fontsize=7)
    axes[1].set_yticks(range(N_c))
    axes[1].set_yticklabels(short_lbl, fontsize=7)
    axes[1].set_title("Holdings Overlap (Jaccard Similarity of Positions)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)

    dendrogram(
        Z_lk,
        labels=[c.replace("__top10_", "__") for c in cids],
        orientation="right",
        ax=axes[2],
        leaf_font_size=7.5,
    )
    axes[2].set_title("24-Candidate Complementarity Dendrogram")
    axes[2].set_xlabel("Correlation Distance")
    axes[2].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig04_01_corr_overlap_clusters.png", dpi=150)
    plt.close(fig)

    # Fig 4.2: Rolling 60d/126d/252d Correlations
    fig, axes = plt.subplots(2, 3, figsize=(21, 9.2), sharex=True, sharey=True)
    for ax, (lbl, (r60, r126, r252)) in zip(axes.flat, roll_figs.items()):
        ax.plot(r60.index, r60.values, color="#aec7e8", lw=0.9, label="60d")
        ax.plot(r126.index, r126.values, color="#1f77b4", lw=1.4, label="126d")
        ax.plot(r252.index, r252.values, color="#d62728", lw=2.0, label="252d")
        ax.axhline(0.0, color="black", ls="--", lw=0.8)
        ax.set_title(lbl, fontsize=8.8)
        ax.set_ylim(-0.6, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5)
    fig.suptitle(
        "Rolling Pairwise Net Return Correlations (60d / 126d / 252d)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "fig04_02_rolling_correlations.png", dpi=150)
    plt.close(fig)

    # Fig 4.3: Risk Contribution & 3-Factor Alpha vs Beta_MKT / Beta_VOL
    fig, axes = plt.subplots(1, 2, figsize=(21, 7.8))
    rc_s = df_rc.sort_values("RiskShare_EW24(%)", ascending=True)
    exp_cols = {"raw_t1": "#1f77b4", "risk_t1": "#2ca02c", "risk_t5": "#ff7f0e"}
    bar_c = [exp_cols[s.split("__")[0]] for s in rc_s.index]
    axes[0].barh(np.arange(N_c), rc_s["RiskShare_EW24(%)"], color=bar_c, alpha=0.88)
    axes[0].axvline(
        100.0 / N_c,
        color="black",
        ls="--",
        lw=1.2,
        label=f"Equal Share ({100.0 / N_c:.2f}%)",
    )
    axes[0].set_yticks(np.arange(N_c))
    axes[0].set_yticklabels(rc_s.index, fontsize=7.8)
    axes[0].set_xlabel("Risk Share in EW-24 Portfolio (%)")
    axes[0].set_title("Component Risk Contribution Across 24 Candidates")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(fontsize=9)

    for exp, col in exp_cols.items():
        m = df_fac.index.str.startswith(exp + "__")
        sub_f = df_fac.loc[m]
        sub_r = df_rc.loc[m]
        axes[1].scatter(
            sub_r["Hold_Beta60"],
            sub_f["PureAlpha(%/yr)"],
            s=np.clip(sub_r["AnnVol2(%)"] * 8.0, 50, 320),
            color=col,
            alpha=0.82,
            edgecolors="black",
            label=exp,
        )
        for cid_f, rf in sub_f.iterrows():
            axes[1].annotate(
                cid_f.split("__")[1].replace("top10_", ""),
                (sub_r.loc[cid_f, "Hold_Beta60"], rf["PureAlpha(%/yr)"]),
                xytext=(4, 2),
                textcoords="offset points",
                fontsize=7.2,
            )
    axes[1].axvline(0.0, color="black", ls="--", lw=0.8)
    axes[1].set_xlabel("Realized Portfolio Holdings Beta (lagged beta_60)")
    axes[1].set_ylabel("3-Factor Pure Alpha (%/yr)")
    axes[1].set_title(
        "Realized Holdings Beta vs 3-Factor Pure Alpha (Note: BetaNeu != Zero Beta!)"
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig04_03_risk_and_beta_alpha.png", dpi=150)
    plt.close(fig)

    # Fig 4.4: Regime Heatmap
    fig, ax = plt.subplots(figsize=(20, 9.0))
    im_r = ax.imshow(
        reg_sh.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=2.5
    )
    ax.set_xticks(range(len(reg_sh.columns)))
    ax.set_xticklabels(reg_sh.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(N_c))
    ax.set_yticklabels(cids, fontsize=8)
    for ri in range(N_c):
        for ci in range(len(reg_sh.columns)):
            ax.text(
                ci,
                ri,
                f"{reg_sh.iloc[ri, ci]:+.1f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    fig.colorbar(im_r, ax=ax, fraction=0.03, pad=0.02, label="Net Sharpe @ 2bps")
    ax.set_title("Regime-Conditioned Net Sharpe (@ 2bps) Across All 24 Candidates")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig04_04_regime_heatmap_24.png", dpi=150)
    plt.close(fig)

    return {
        "corr_pearson": corr_p,
        "corr_spearman": corr_s,
        "df_pairs": df_pairs,
        "df_rc": df_rc,
        "df_fac": df_fac,
        "reg_sh": reg_sh,
        "df_ov_all": df_ov_all,
        "regimes": regimes,
    }
