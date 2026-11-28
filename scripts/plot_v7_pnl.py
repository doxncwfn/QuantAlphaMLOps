"""Generate dedicated $M PnL comparison chart for Stock Ranking v7 ensemble finalists."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.qam.v7_ensemble.combiners import WARMUP_DAYS
from src.qam.v7_ensemble.engine import (
    ANN_DAYS,
    BASELINE_COST_BPS,
    BOOK_SIZE,
    V7DataBundle,
    apply_portfolio_constraints,
    backtest_weights_daily,
    load_v7_data_bundle,
    neutralize_beta,
    strat_top10_buf,
)


def _load_sparse_weights(
    path: Path, dates: pd.DatetimeIndex, sec_ids: np.ndarray
) -> np.ndarray:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    piv = (
        df.pivot(index="date", columns="security_id", values="weight")
        .reindex(index=dates, columns=sec_ids)
        .fillna(0.0)
    )
    return piv.to_numpy(dtype=np.float64)


def _xs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True) - 0.5


def _comp_pnl_m(r: pd.Series, book_m: float = BOOK_SIZE / 1e6) -> pd.Series:
    nav = (1.0 + r.clip(lower=-0.95)).cumprod()
    return book_m * (nav - 1.0)


def _fixed_pnl_m(r: pd.Series, book_m: float = BOOK_SIZE / 1e6) -> pd.Series:
    return book_m * r.cumsum()


def main() -> None:
    qh_dir = Path("outputs/QH_v7")
    ens_dir = qh_dir / "v7_ensemble"
    fig_dir = ens_dir / "figures"
    wgt_dir = ens_dir / "weights"

    bundle: V7DataBundle = load_v7_data_bundle(qh_dir=qh_dir)
    dates = bundle.dates
    sec_ids = bundle.security_ids
    oos_dates = dates[WARMUP_DAYS:]
    r1_np = bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64)
    adv_np = bundle.adv20_df.to_numpy(dtype=np.float64)
    sig_np = bundle.sigma_prev_df.to_numpy(dtype=np.float64)
    rf_np = bundle.ff6_df["rf"].to_numpy(dtype=np.float64)

    def _bt(W: np.ndarray, cost_bps: float = BASELINE_COST_BPS) -> dict[str, pd.Series]:
        return backtest_weights_daily(
            W,
            r1_np,
            dates,
            cost_bps=cost_bps,
            adv20_np=adv_np,
            sigma_np=sig_np,
            rf_np=rf_np,
            inception_zero_trade=False,
        )

    # Load saved flagship & benchmark ensemble weights
    W_hier_g5 = _load_sparse_weights(
        wgt_dir / "weights_G_HierBest_G5_FullConstrainedOverlay.parquet", dates, sec_ids
    )
    W_e3_dyn = _load_sparse_weights(
        wgt_dir / "weights_E3_RollMinVar_126d_Hyst.parquet", dates, sec_ids
    )
    W_ew24 = _load_sparse_weights(
        wgt_dir / "weights_A1_EW_All24.parquet", dates, sec_ids
    )

    # Reconstruct Best Individual (risk_t5__top10_BetaNeu_InvVol_Buf) & B5 / G_ScoreBest_G1
    inv_vol = (1.0 / bundle.sigma_prev_df).fillna(0.0)
    sc_t5 = bundle.scores["risk_t5"]
    W_best_ind = strat_top10_buf(
        neutralize_beta(sc_t5, bundle.beta60_df),
        inv_vol.where(sc_t5.notna()).fillna(0.0),
    ).to_numpy(dtype=np.float64)

    sc_b5 = (
        _xs_rank(bundle.scores["raw_t1"])
        + _xs_rank(bundle.scores["risk_t1"])
        + _xs_rank(bundle.scores["risk_t5"])
    ) / 3.0
    W_b5 = strat_top10_buf(
        neutralize_beta(sc_b5, bundle.beta60_df),
        inv_vol.where(sc_b5.notna()).fillna(0.0),
    ).to_numpy(dtype=np.float64)

    port_r_b5 = np.einsum("ij,ij->i", W_b5, r1_np)
    r_ser = pd.Series(port_r_b5)
    v20 = r_ser.rolling(20, min_periods=10).std().shift(1).to_numpy(
        dtype=np.float64
    ) * math.sqrt(ANN_DAYS)
    v63 = r_ser.rolling(63, min_periods=20).std().shift(1).to_numpy(
        dtype=np.float64
    ) * math.sqrt(ANN_DAYS)
    vhat = np.where(
        np.isfinite(v20) & np.isfinite(v63),
        np.maximum(v20, 0.5 * v20 + 0.5 * v63),
        0.10,
    )
    sc_vt = np.clip(0.09 / np.maximum(vhat, 0.025), 0.35, 1.35)
    sc_vt[:25] = 1.0
    W_score_g1 = apply_portfolio_constraints(W_b5 * sc_vt[:, None], max_single_w=0.035)

    bts = {
        "G_HierBest_G5 (Flagship)": _bt(W_hier_g5),
        "G_ScoreBest_G1 (VolTgt9%)": _bt(W_score_g1),
        "B5_ScoreEW_Rank_All3": _bt(W_b5),
        "E3_RollMinVar_126d_Hyst": _bt(W_e3_dyn),
        "risk_t5__BetaNeu (BestInd)": _bt(W_best_ind),
        "A1_EW_All24": _bt(W_ew24),
    }
    bts_5 = {
        "G_HierBest_G5 (Flagship)": _bt(W_hier_g5, cost_bps=5.0),
        "G_ScoreBest_G1 (VolTgt9%)": _bt(W_score_g1, cost_bps=5.0),
        "B5_ScoreEW_Rank_All3": _bt(W_b5, cost_bps=5.0),
        "E3_RollMinVar_126d_Hyst": _bt(W_e3_dyn, cost_bps=5.0),
        "risk_t5__BetaNeu (BestInd)": _bt(W_best_ind, cost_bps=5.0),
        "A1_EW_All24": _bt(W_ew24, cost_bps=5.0),
    }

    # Plot 2x2 Comprehensive PnL Figure ($M on $20M Equity Book)
    _fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Panel 1: Flagship Gross vs Net @ 2bps vs All-In ($M)
    bt_fl = bts["G_HierBest_G5 (Flagship)"]
    for key, lbl, col, ls in [
        ("gross", "Gross (0 bps)", "#2ca02c", "-"),
        ("net", "Net @ 2 bps", "#1f77b4", "-"),
        ("net_allin", "All-In (2bps+Borrow+Impact)", "#d62728", "--"),
    ]:
        s = _comp_pnl_m(bt_fl[key].iloc[WARMUP_DAYS:])
        ax1.plot(
            oos_dates,
            s,
            label=f"{lbl}: ${s.iloc[-1]:.1f}M",
            color=col,
            linestyle=ls,
            linewidth=1.1,
        )
    ax1.set_title(
        "Flagship G_HierBest_G5: Gross vs Net vs All-In PnL (USD M, 20M Book)"
    )
    ax1.set_ylabel("Compounded PnL ($M)")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(alpha=0.25)

    # Panel 2: Return-Oriented G_ScoreBest_G1 Gross vs Net vs All-In ($M)
    bt_sc = bts["G_ScoreBest_G1 (VolTgt9%)"]
    for key, lbl, col, ls in [
        ("gross", "Gross (0 bps)", "#2ca02c", "-"),
        ("net", "Net @ 2 bps", "#1f77b4", "-"),
        ("net_allin", "All-In (2bps+Borrow+Impact)", "#d62728", "--"),
    ]:
        s = _comp_pnl_m(bt_sc[key].iloc[WARMUP_DAYS:])
        ax2.plot(
            oos_dates,
            s,
            label=f"{lbl}: ${s.iloc[-1]:.1f}M",
            color=col,
            linestyle=ls,
            linewidth=1.1,
        )
    ax2.set_title(
        "Return-Oriented G_ScoreBest_G1: Gross vs Net vs All-In PnL (USD M, 20M Book)"
    )
    ax2.set_ylabel("Compounded PnL ($M)")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(alpha=0.25)

    # Panel 3: Cross-Strategy Net @ 2 bps Compounded PnL ($M)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f"]
    for (nm, btd), col in zip(bts.items(), colors):
        s = _comp_pnl_m(btd["net"].iloc[WARMUP_DAYS:])
        lw = 1.3 if "Flagship" in nm or "G_ScoreBest" in nm or "B5_" in nm else 0.9
        ax3.plot(
            oos_dates, s, label=f"{nm}: +${s.iloc[-1]:.1f}M", color=col, linewidth=lw
        )
    ew_m = _comp_pnl_m(bundle.ew_mkt_1d.iloc[WARMUP_DAYS:])
    ax3.plot(
        oos_dates,
        ew_m,
        color="black",
        linestyle=":",
        linewidth=0.9,
        label=f"EW_Russell1000: +${ew_m.iloc[-1]:.1f}M",
    )
    ax3.set_title(
        "Cumulative Net PnL @ 2 bps Across Ensembles vs Best Ind (USD M, 20M Book)"
    )
    ax3.set_ylabel("Compounded Net PnL ($M)")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(alpha=0.25)

    # Panel 4: Annual Net Dollar PnL ($M per Year) Flagship vs ScoreBest vs BestInd
    ann_df = pd.DataFrame(
        {
            "G_HierBest_G5": (
                1.0 + bts["G_HierBest_G5 (Flagship)"]["net"].iloc[WARMUP_DAYS:]
            )
            .resample("YE")
            .prod()
            - 1.0,
            "G_ScoreBest_G1": (
                1.0 + bts["G_ScoreBest_G1 (VolTgt9%)"]["net"].iloc[WARMUP_DAYS:]
            )
            .resample("YE")
            .prod()
            - 1.0,
            "BestInd_risk_t5_BN": (
                1.0 + bts["risk_t5__BetaNeu (BestInd)"]["net"].iloc[WARMUP_DAYS:]
            )
            .resample("YE")
            .prod()
            - 1.0,
        }
    ) * (BOOK_SIZE / 1e6)  # Stand-alone $20M annual start-of-year PnL
    ann_df.index = ann_df.index.year
    ann_df.plot(
        kind="bar",
        ax=ax4,
        width=0.78,
        color=["#1f77b4", "#ff7f0e", "#8c564b"],
        alpha=0.88,
    )
    ax4.axhline(0.0, color="black", linewidth=0.8)
    ax4.set_title("Calendar-Year Net PnL (USD M per 20M Start-of-Year Book, @ 2 bps)")
    ax4.set_ylabel("Annual Net PnL ($M)")
    ax4.tick_params(axis="x", rotation=45)
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out_path = fig_dir / "fig08_06_pnl_dollar_comparison.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved {out_path}")

    # Second figure: Compounded vs Fixed-20M Uncompounded PnL
    _fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    bx1, bx2, bx3, bx4 = axes.ravel()

    for (nm, btd), col in zip(bts.items(), colors):
        s_c = _comp_pnl_m(btd["net"].iloc[WARMUP_DAYS:])
        s_f = _fixed_pnl_m(btd["net"].iloc[WARMUP_DAYS:])
        lw = 1.3 if "Flagship" in nm or "G_ScoreBest" in nm or "B5_" in nm else 0.9
        bx1.plot(
            oos_dates,
            s_c,
            label=f"{nm}: +${s_c.iloc[-1]:.1f}M",
            color=col,
            linewidth=lw,
        )
        bx2.plot(
            oos_dates,
            s_f,
            label=f"{nm}: +${s_f.iloc[-1]:.1f}M",
            color=col,
            linewidth=lw,
        )

    ew_c = _comp_pnl_m(bundle.ew_mkt_1d.iloc[WARMUP_DAYS:])
    ew_f = _fixed_pnl_m(bundle.ew_mkt_1d.iloc[WARMUP_DAYS:])
    bx1.plot(
        oos_dates,
        ew_c,
        color="black",
        linestyle=":",
        linewidth=0.9,
        label=f"EW_Russell1000: +${ew_c.iloc[-1]:.1f}M",
    )
    bx2.plot(
        oos_dates,
        ew_f,
        color="black",
        linestyle=":",
        linewidth=0.9,
        label=f"EW_Russell1000: +${ew_f.iloc[-1]:.1f}M",
    )

    bx1.set_title("Compounded Net PnL @ 2 bps (USD M, 20M Start, Reinvested)")
    bx1.set_ylabel("Compounded Net PnL ($M)")
    bx1.legend(loc="upper left", fontsize=8)
    bx1.grid(alpha=0.25)

    bx2.set_title("Fixed-20M Uncompounded Net PnL @ 2 bps (USD M, No Reinvestment)")
    bx2.set_ylabel("Uncompounded Net PnL ($M)")
    bx2.legend(loc="upper left", fontsize=8)
    bx2.grid(alpha=0.25)

    bt_fl5 = bts_5["G_HierBest_G5 (Flagship)"]
    for ser_fl, ser_b5, lbl, col, ls in [
        (
            bt_fl["gross"],
            bts["B5_ScoreEW_Rank_All3"]["gross"],
            "Gross (0 bps)",
            "#2ca02c",
            "-",
        ),
        (
            bt_fl["net"],
            bts["B5_ScoreEW_Rank_All3"]["net"],
            "Net @ 2 bps",
            "#1f77b4",
            "-",
        ),
        (
            bt_fl5["net"],
            bts_5["B5_ScoreEW_Rank_All3"]["net"],
            "Net @ 5 bps",
            "#9467bd",
            "-.",
        ),
        (
            bt_fl["net_allin"],
            bts["B5_ScoreEW_Rank_All3"]["net_allin"],
            "All-In (@2bps+Borrow+Impact)",
            "#d62728",
            "--",
        ),
    ]:
        s_fl = _fixed_pnl_m(ser_fl.iloc[WARMUP_DAYS:])
        s_b5 = _fixed_pnl_m(ser_b5.iloc[WARMUP_DAYS:])
        bx3.plot(
            oos_dates,
            s_fl,
            label=f"{lbl}: ${s_fl.iloc[-1]:.1f}M",
            color=col,
            linestyle=ls,
            linewidth=1.1,
        )
        bx4.plot(
            oos_dates,
            s_b5,
            label=f"{lbl}: ${s_b5.iloc[-1]:.1f}M",
            color=col,
            linestyle=ls,
            linewidth=1.1,
        )

    bx3.set_title("Flagship G_HierBest_G5: Fixed-20M Uncompounded PnL (USD M)")
    bx3.set_ylabel("Uncompounded PnL ($M)")
    bx3.legend(loc="upper left", fontsize=8.5)
    bx3.grid(alpha=0.25)

    bx4.set_title("B5_ScoreEW_Rank_All3: Fixed-20M Uncompounded PnL (USD M)")
    bx4.set_ylabel("Uncompounded PnL ($M)")
    bx4.legend(loc="upper left", fontsize=8.5)
    bx4.grid(alpha=0.25)

    plt.tight_layout()
    out_path2 = fig_dir / "fig08_07_pnl_compounded_vs_fixed20m.png"
    plt.savefig(out_path2, dpi=160)
    plt.close()
    print(f"Saved {out_path2}")

    for nm, btd in bts.items():
        btd5 = bts_5[nm]
        g_c = float(_comp_pnl_m(btd["gross"].iloc[WARMUP_DAYS:]).iloc[-1])
        n2_c = float(_comp_pnl_m(btd["net"].iloc[WARMUP_DAYS:]).iloc[-1])
        n5_c = float(_comp_pnl_m(btd5["net"].iloc[WARMUP_DAYS:]).iloc[-1])
        a_c = float(_comp_pnl_m(btd["net_allin"].iloc[WARMUP_DAYS:]).iloc[-1])
        g_f = float(_fixed_pnl_m(btd["gross"].iloc[WARMUP_DAYS:]).iloc[-1])
        n2_f = float(_fixed_pnl_m(btd["net"].iloc[WARMUP_DAYS:]).iloc[-1])
        n5_f = float(_fixed_pnl_m(btd5["net"].iloc[WARMUP_DAYS:]).iloc[-1])
        a_f = float(_fixed_pnl_m(btd["net_allin"].iloc[WARMUP_DAYS:]).iloc[-1])
        print(
            f"  {nm:28s}: Comp(G=${g_c:6.2f}M, N2=${n2_c:6.2f}M, N5=${n5_c:6.2f}M, AI=${a_c:6.2f}M) | Fixed20M(G=${g_f:5.2f}M, N2=${n2_f:5.2f}M, N5=${n5_f:5.2f}M, AI=${a_f:5.2f}M)"
        )


if __name__ == "__main__":
    main()
