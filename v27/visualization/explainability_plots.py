"""Visualization charts for alpha explainability and Section 4 strategy complementarity."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, spearmanr

from v27.config import CANDIDATES_24, EXPERIMENTS_V7
from v27.data.bundle import V7DataBundle
from v27.signal_analysis.decile_monotonicity import compute_decile_monotonicity
from v27.visualization.style import (
    CLUSTER_COLORS,
    EXP_COLORS,
    save_fig,
    shade_crisis_eras,
)


def plot_master_gate_clusters(
    cluster_df: pd.DataFrame, name: str, out_path: Path
) -> Path:
    """Plot rolling 21-day MASTER gate cluster multipliers and crisis attribution bars."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )
    roll = cluster_df.rolling(21, min_periods=5).mean()
    for c in roll.columns:
        ax1.plot(roll.index, roll[c], label=c, lw=1.8, color=CLUSTER_COLORS.get(c))
    ax1.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax1.set_title(f"[{name}] MASTER Gate 5-Cluster Feature Importance (21d MA)")
    ax1.set_ylabel("Mean Multiplier")
    ax1.legend(fontsize=8.5)
    ax1.grid(alpha=0.25)
    shade_crisis_eras(ax1, add_labels=True)

    means = cluster_df.mean(axis=0)
    cols_b = [CLUSTER_COLORS.get(c, "#1f77b4") for c in means.index]
    ax2.barh(means.index, means.values, color=cols_b, alpha=0.88)
    ax2.axvline(1.0, color="black", ls="--", lw=0.8)
    ax2.set_xlabel("OOS Mean Gate Multiplier")
    ax2.set_title(f"[{name}] Average Cluster Share")
    ax2.grid(axis="x", alpha=0.25)
    return save_fig(fig, out_path, dpi=150)


def plot_alpha_decay(
    df_dec: pd.DataFrame,
    ic_by_lag: dict[int, float],
    name: str,
    cost_bps: float,
    out_path: Path,
) -> Path:
    """Plot single-day forward IC decay (t+1..t+H) and overlapping HOLD Sharpe vs Turnover."""
    horizons = list(df_dec.index.astype(int))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.bar(
        horizons,
        [ic_by_lag[h] for h in horizons],
        color="#1f77b4",
        alpha=0.85,
        width=0.55,
    )
    ax1.axhline(0.0, color="k", lw=0.8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(horizons)
    ax1.set_xticklabels([f"t+{h}" for h in horizons])
    ax1.set_title(f"[{name}] Single-Day Forward Rank IC Decay")
    ax1.set_ylabel("Mean Daily Rank IC")

    net_col = f"NetSharpe@{cost_bps:g}bps"
    ax2.plot(
        horizons,
        df_dec["GrossSharpe"],
        "o-",
        lw=2,
        color="#1f77b4",
        label="Gross Sharpe",
    )
    ax2.plot(horizons, df_dec[net_col], "s-", lw=2.2, color="#2ca02c", label=net_col)
    ax2.set_xlabel("Holding Period (Overlapping Tranches)")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_xticks(horizons)
    ax2.grid(True, alpha=0.3)
    ax2_r = ax2.twinx()
    ax2_r.plot(
        horizons,
        df_dec["Turnover(%)"],
        "^--",
        lw=1.8,
        color="#d62728",
        label="Turnover (%)",
    )
    ax2_r.set_ylabel("Daily Turnover (%)", color="#d62728")
    ax2.set_title(f"[{name}] Sharpe vs Turnover by Hold Horizon")
    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2_r.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, loc="best", fontsize=8.5)
    return save_fig(fig, out_path, dpi=150)


def plot_corr_overlap_clusters(
    corr_p: pd.DataFrame, df_ov_all: pd.DataFrame, out_path: Path
) -> Path:
    """Fig 04_01: Pearson return correlation, Jaccard overlap, and Ward dendrogram."""
    cids = list(corr_p.index)
    n_c = len(cids)
    fig, axes = plt.subplots(
        1, 3, figsize=(24, 7.6), gridspec_kw={"width_ratios": [1.1, 1.1, 0.9]}
    )
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr_p.to_numpy()), 0.0))
    np.fill_diagonal(dist, 0.0)
    z_lk = linkage(squareform(dist, checks=False), method="ward")
    ord_c = [cids[k] for k in leaves_list(z_lk)]
    short_lbl = [
        c.replace("__top10_", "__")
        .replace("_BetaNeu_InvVol_Buf", "_BN_IV_B")
        .replace("_InvVol_", "_IV_")
        for c in ord_c
    ]

    im0 = axes[0].imshow(
        corr_p.loc[ord_c, ord_c].to_numpy(), cmap="RdBu_r", vmin=-0.3, vmax=1.0
    )
    axes[0].set_xticks(range(n_c))
    axes[0].set_xticklabels(short_lbl, rotation=90, fontsize=7)
    axes[0].set_yticks(range(n_c))
    axes[0].set_yticklabels(short_lbl, fontsize=7)
    axes[0].set_title("Daily Net Return Pearson Correlation (24 Candidates)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)

    im1 = axes[1].imshow(
        df_ov_all.loc[ord_c, ord_c].to_numpy(), cmap="YlGnBu", vmin=0.0, vmax=1.0
    )
    axes[1].set_xticks(range(n_c))
    axes[1].set_xticklabels(short_lbl, rotation=90, fontsize=7)
    axes[1].set_yticks(range(n_c))
    axes[1].set_yticklabels(short_lbl, fontsize=7)
    axes[1].set_title("Holdings Overlap (Jaccard Similarity of Positions)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)

    dendrogram(
        z_lk,
        labels=[c.replace("__top10_", "__") for c in cids],
        orientation="right",
        ax=axes[2],
        leaf_font_size=7.5,
    )
    axes[2].set_title("24-Candidate Complementarity Dendrogram")
    axes[2].set_xlabel("Correlation Distance")
    axes[2].grid(axis="x", alpha=0.25)
    return save_fig(fig, out_path, dpi=150)


def plot_rolling_correlations(net_df: pd.DataFrame, out_path: Path) -> Path:
    """Fig 04_02: Rolling 60d/126d/252d return correlations across key strategy pairs."""
    key_pairs = [
        ("raw_t1__top10_InvVol_Buf", "risk_t1__top10_BetaNeu_InvVol_Buf"),
        ("raw_t1__top10_Buf10_20", "risk_t1__top10_BetaNeu_InvVol_Buf"),
        ("raw_t1__top10_InvVol_Buf", "risk_t1__top10_Buf10_20"),
        ("raw_t1__top10_InvVol_HOLD5", "risk_t1__top10_BetaNeu_InvVol_Buf"),
        ("risk_t5__top10_BetaNeu_InvVol_Buf", "raw_t1__top10_InvVol_Buf"),
        ("risk_t5__rank_RAW", "risk_t1__top10_BetaNeu_InvVol_Buf"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(21, 9.2), sharex=True, sharey=True)
    for ax, (ca, cb) in zip(axes.flat, key_pairs):
        r60 = net_df[ca].rolling(60).corr(net_df[cb])
        r126 = net_df[ca].rolling(126).corr(net_df[cb])
        r252 = net_df[ca].rolling(252).corr(net_df[cb])
        ax.plot(r60.index, r60.values, color="#aec7e8", lw=0.9, label="60d")
        ax.plot(r126.index, r126.values, color="#1f77b4", lw=1.4, label="126d")
        ax.plot(r252.index, r252.values, color="#d62728", lw=2.0, label="252d")
        ax.axhline(0.0, color="black", ls="--", lw=0.8)
        ax.set_title(f"{ca}\nvs {cb}", fontsize=8.8)
        ax.set_ylim(-0.6, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5)
        shade_crisis_eras(ax)
    fig.suptitle(
        "Rolling Pairwise Net Return Correlations (60d / 126d / 252d)",
        fontsize=13,
        fontweight="bold",
    )
    return save_fig(fig, out_path, dpi=150)


def plot_risk_and_beta_alpha(
    df_rc: pd.DataFrame, df_fac: pd.DataFrame, out_path: Path
) -> Path:
    """Fig 04_03: EW-24 component risk share & realized holdings beta vs 3-factor alpha."""
    n_c = len(df_rc)
    fig, axes = plt.subplots(1, 2, figsize=(21, 7.8))
    rc_s = df_rc.sort_values("RiskShare_EW24(%)", ascending=True)
    bar_c = [EXP_COLORS[s.split("__")[0]] for s in rc_s.index]
    axes[0].barh(np.arange(n_c), rc_s["RiskShare_EW24(%)"], color=bar_c, alpha=0.88)
    axes[0].axvline(
        100.0 / n_c,
        color="black",
        ls="--",
        lw=1.2,
        label=f"Equal Share ({100.0 / n_c:.2f}%)",
    )
    axes[0].set_yticks(np.arange(n_c))
    axes[0].set_yticklabels(rc_s.index, fontsize=7.8)
    axes[0].set_xlabel("Risk Share in EW-24 Portfolio (%)")
    axes[0].set_title("Component Risk Contribution Across 24 Candidates")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(fontsize=9)

    for exp, col in EXP_COLORS.items():
        m = df_fac.index.str.startswith(exp + "__")
        sub_f, sub_r = df_fac.loc[m], df_rc.loc[m]
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
    axes[1].set_title("Realized Holdings Beta vs 3-Factor Pure Alpha")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)
    return save_fig(fig, out_path, dpi=150)


def plot_regime_heatmap(reg_sh: pd.DataFrame, out_path: Path) -> Path:
    """Fig 04_04: Regime-conditioned Net Sharpe heatmap across all 24 candidates."""
    cids = list(CANDIDATES_24)
    n_c = len(cids)
    fig, ax = plt.subplots(figsize=(20, 9.0))
    im_r = ax.imshow(
        reg_sh.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=2.5
    )
    ax.set_xticks(range(len(reg_sh.columns)))
    ax.set_xticklabels(reg_sh.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_c))
    ax.set_yticklabels(cids, fontsize=8)
    for ri in range(n_c):
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
    return save_fig(fig, out_path, dpi=150)


def plot_ic_and_decay_profile(
    bundle: V7DataBundle, out_path: Path, ann_days: int = 252
) -> Path:
    """Plot Cumulative Daily Rank IC, Rolling 60d Mean IC, and Execution-Lag Alpha Decay."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 11.5))
    lags = [0, 1, 2, 3, 5, 10]
    for exp in EXPERIMENTS_V7:
        col = EXP_COLORS.get(exp, "#1f77b4")
        sc = bundle.scores[exp]
        act = sc.notna()
        ic_ser = (
            sc.rank(axis=1)
            .corrwith(bundle.ret1d_df.where(act).rank(axis=1), axis=1)
            .dropna()
        )
        mu_ic = float(ic_ser.mean())
        sd_ic = float(ic_ser.std(ddof=1)) + 1e-12
        t_st = mu_ic / (sd_ic / np.sqrt(max(len(ic_ser), 1)))
        icir = mu_ic / sd_ic * np.sqrt(ann_days)

        ax1.plot(
            ic_ser.index,
            ic_ser.cumsum(),
            color=col,
            lw=1.6,
            label=f"{exp} (IC={mu_ic:+.4f}, t={t_st:+.2f}, ICIR={icir:+.2f})",
        )
        r60 = ic_ser.rolling(60, min_periods=20).mean()
        ax2.plot(r60.index, r60, color=col, lw=1.2, label=exp)

        lag_ics = [
            float(
                sc.rank(axis=1)
                .corrwith(bundle.ret1d_df.shift(-lg).where(act).rank(axis=1), axis=1)
                .dropna()
                .mean()
            )
            for lg in lags
        ]
        ax3.plot(lags, lag_ics, marker="o", color=col, lw=1.8, label=exp)

    ax1.axhline(0.0, color="black", ls="--", lw=0.8)
    ax1.set_title("Cumulative Daily Spearman Rank IC (Pooled OOS)")
    ax1.set_ylabel("Cumulative IC")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(alpha=0.25)
    shade_crisis_eras(ax1, add_labels=True)

    ax2.axhline(0.0, color="black", ls="--", lw=0.8)
    ax2.set_title("Rolling 60-Day Mean Spearman Rank IC (Regime Stability)")
    ax2.set_ylabel("60D Mean IC")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(alpha=0.25)
    shade_crisis_eras(ax2)

    ax3.axhline(0.0, color="black", ls="--", lw=0.8)
    ax3.set_title("Alpha Decay Profile: Mean Rank IC vs Execution Lag)")
    ax3.set_xlabel("Execution Lag (Trading Days)")
    ax3.set_ylabel("Mean Spearman IC")
    ax3.set_xticks(lags)
    ax3.legend(loc="upper right", fontsize=8.5)
    ax3.grid(alpha=0.25)
    return save_fig(fig, out_path, dpi=150)


def plot_decile_monotonicity(
    bundle: V7DataBundle, out_path: Path, ann_days: int = 252
) -> Path:
    """Plot D1..D10 annualized return bars and Spearman/Kendall monotonicity across experiments."""
    n_exp = len(EXPERIMENTS_V7)
    fig, axes = plt.subplots(1, n_exp, figsize=(5.8 * n_exp, 5.2), sharey=True)
    if n_exp == 1:
        axes = [axes]
    cmap = plt.get_cmap("coolwarm")
    bar_cols = [cmap(i / 9.0) for i in range(10)]

    for ax, exp in zip(axes, EXPERIMENTS_V7):
        dec_df, st = compute_decile_monotonicity(
            bundle.scores[exp], bundle.ret1d_df, ann_days=ann_days
        )
        vals = dec_df["AnnRet(%)"].to_numpy(dtype=float)
        kt, _ = kendalltau(np.arange(1, 11), vals)
        sp, _ = spearmanr(np.arange(1, 11), vals)
        bars = ax.bar(
            np.arange(1, 11),
            vals,
            color=bar_cols,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v + (0.4 if v >= 0 else -1.2),
                f"{v:+.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xticks(np.arange(1, 11))
        ax.set_xlabel("Prediction Decile (1=Short, 10=Long)")
        ax.set_title(
            f"{exp}\nSp={sp:+.3f} | Kt={kt:+.3f} | Step={st['Adj_Step_Rate(%)']:.0f}%",
            fontsize=9.5,
        )
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean Annualized Return (%)")
    fig.suptitle(
        "Decile Portfolio Monotonicity Spreads (OOS Daily Rebalance)",
        fontsize=12,
        fontweight="bold",
    )
    return save_fig(fig, out_path, dpi=150)
