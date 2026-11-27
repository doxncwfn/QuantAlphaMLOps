"""Dedicated $20M equity-book dollar PnL charts (Gross, Net@2bps, Net@5bps, All-In)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v27.backtest.simulator import backtest_weights_daily
from v27.config import PortfolioBacktestConfig
from v27.data.bundle import V7DataBundle
from v27.visualization.style import save_fig


def _comp_pnl_m(r: pd.Series, book_m: float) -> pd.Series:
    return book_m * ((1.0 + r.clip(lower=-0.95)).cumprod() - 1.0)


def _fixed_pnl_m(r: pd.Series, book_m: float) -> pd.Series:
    return book_m * r.cumsum()


def plot_pnl_dollar_suite(
    bundle: V7DataBundle,
    all_entries: dict[str, dict[str, Any]],
    fig_dir: Path,
    bt_cfg: PortfolioBacktestConfig | None = None,
) -> None:
    """Generate fig08_06 (dollar PnL + annual breakdown) and fig08_07 (compounded vs fixed-$20M + Net@5bps)."""
    bt_cfg = bt_cfg or PortfolioBacktestConfig()
    wu = bt_cfg.warmup_days
    book_m = bt_cfg.book_size / 1e6
    oos_dates = bundle.dates[wu:]
    r1_np = bundle.ret1d_df.fillna(0.0).to_numpy(dtype=np.float64)
    adv_np = bundle.adv20_df.to_numpy(dtype=np.float64)
    sig_np = bundle.sigma_prev_df.to_numpy(dtype=np.float64)
    rf_np = bundle.ff6_df["rf"].to_numpy(dtype=np.float64)

    candidates = [
        ("G_HierBest_G5 (Flagship)", "G_HierBest_G5_FullConstrainedOverlay"),
        ("G_ScoreBest_G1 (VolTgt9%)", "G_ScoreBest_G1_VolTgt09"),
        ("B5_ScoreEW_Rank_All3", "B5_ScoreEW_Rank_All3__top10_BetaNeu_InvVol_Buf"),
        ("E3_RollMinVar_126d_Hyst", "E3_RollMinVar_126d_Hyst"),
        ("risk_t5__BetaNeu (BestInd)", "risk_t5__top10_BetaNeu_InvVol_Buf"),
        ("A1_EW_All24", "A1_EW_All24"),
    ]

    bts: dict[str, dict[str, pd.Series]] = {}
    bts_5: dict[str, dict[str, pd.Series]] = {}
    for label, key in candidates:
        if key not in all_entries:
            continue
        bts[label] = all_entries[key]["bt"]
        bts_5[label] = backtest_weights_daily(
            all_entries[key]["weights"],
            r1_np,
            bundle.dates,
            cost_bps=5.0,
            adv20_np=adv_np,
            sigma_np=sig_np,
            rf_np=rf_np,
            book_size=bt_cfg.book_size,
            short_borrow_ann=bt_cfg.short_borrow_ann,
            impact_eta=bt_cfg.impact_coeff,
            ann_days=bt_cfg.ann_days,
        )

    if "G_HierBest_G5 (Flagship)" not in bts or "G_ScoreBest_G1 (VolTgt9%)" not in bts:
        return

    # Fig 08_06: Gross vs Net@2bps vs All-In + Cross-Strategy Net + Calendar-Year Bars
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()

    bt_fl = bts["G_HierBest_G5 (Flagship)"]
    for k, lbl, col, ls in [
        ("gross", "Gross (0 bps)", "#2ca02c", "-"),
        ("net", "Net @ 2 bps", "#1f77b4", "-"),
        ("net_allin", "All-In (2bps+Borrow+Impact)", "#d62728", "--"),
    ]:
        s = _comp_pnl_m(bt_fl[k].iloc[wu:], book_m)
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

    bt_sc = bts["G_ScoreBest_G1 (VolTgt9%)"]
    for k, lbl, col, ls in [
        ("gross", "Gross (0 bps)", "#2ca02c", "-"),
        ("net", "Net @ 2 bps", "#1f77b4", "-"),
        ("net_allin", "All-In (2bps+Borrow+Impact)", "#d62728", "--"),
    ]:
        s = _comp_pnl_m(bt_sc[k].iloc[wu:], book_m)
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

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f"]
    for (nm, btd), col in zip(bts.items(), colors):
        s = _comp_pnl_m(btd["net"].iloc[wu:], book_m)
        ax3.plot(
            oos_dates,
            s,
            label=f"{nm}: +${s.iloc[-1]:.1f}M",
            color=col,
            linewidth=1.3
            if "Flagship" in nm or "G_ScoreBest" in nm or "B5_" in nm
            else 0.9,
        )
    ew_m = _comp_pnl_m(bundle.ew_mkt_1d.iloc[wu:], book_m)
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

    ann_df = (
        pd.DataFrame(
            {
                "G_HierBest_G5": (1.0 + bt_fl["net"].iloc[wu:]).resample("YE").prod()
                - 1.0,
                "G_ScoreBest_G1": (1.0 + bt_sc["net"].iloc[wu:]).resample("YE").prod()
                - 1.0,
                "BestInd_risk_t5_BN": (
                    1.0 + bts["risk_t5__BetaNeu (BestInd)"]["net"].iloc[wu:]
                )
                .resample("YE")
                .prod()
                - 1.0,
            }
        )
        * book_m
    )
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
    save_fig(fig, fig_dir / "fig08_06_pnl_dollar_comparison.png")

    # Fig 08_07: Compounded vs Fixed-20M + Net @ 5 bps Sensitivity
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    bx1, bx2, bx3, bx4 = axes.ravel()
    for (nm, btd), col in zip(bts.items(), colors):
        s_c, s_f = (
            _comp_pnl_m(btd["net"].iloc[wu:], book_m),
            _fixed_pnl_m(btd["net"].iloc[wu:], book_m),
        )
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
    ew_c, ew_f = (
        _comp_pnl_m(bundle.ew_mkt_1d.iloc[wu:], book_m),
        _fixed_pnl_m(bundle.ew_mkt_1d.iloc[wu:], book_m),
    )
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
    bx2.set_title("Fixed-20M Uncompounded Net PnL @ 2 bps (USD M, No Reinvestment)")
    bx1.set_ylabel("Compounded Net PnL ($M)")
    bx2.set_ylabel("Uncompounded Net PnL ($M)")
    bx1.legend(loc="upper left", fontsize=8)
    bx2.legend(loc="upper left", fontsize=8)
    bx1.grid(alpha=0.25)
    bx2.grid(alpha=0.25)

    bt_b5 = bts["B5_ScoreEW_Rank_All3"]
    bt_fl5, bt_b5_5 = bts_5["G_HierBest_G5 (Flagship)"], bts_5["B5_ScoreEW_Rank_All3"]
    for s_fl, s_b5, lbl, col, ls in [
        (bt_fl["gross"], bt_b5["gross"], "Gross (0 bps)", "#2ca02c", "-"),
        (bt_fl["net"], bt_b5["net"], "Net @ 2 bps", "#1f77b4", "-"),
        (bt_fl5["net"], bt_b5_5["net"], "Net @ 5 bps", "#9467bd", "-."),
        (
            bt_fl["net_allin"],
            bt_b5["net_allin"],
            "All-In (@2bps+Borrow+Impact)",
            "#d62728",
            "--",
        ),
    ]:
        sf1, sf2 = (
            _fixed_pnl_m(s_fl.iloc[wu:], book_m),
            _fixed_pnl_m(s_b5.iloc[wu:], book_m),
        )
        bx3.plot(
            oos_dates,
            sf1,
            label=f"{lbl}: ${sf1.iloc[-1]:.1f}M",
            color=col,
            linestyle=ls,
            linewidth=1.1,
        )
        bx4.plot(
            oos_dates,
            sf2,
            label=f"{lbl}: ${sf2.iloc[-1]:.1f}M",
            color=col,
            linestyle=ls,
            linewidth=1.1,
        )
    bx3.set_title("Flagship G_HierBest_G5: Fixed-20M Uncompounded PnL (USD M)")
    bx4.set_title("B5_ScoreEW_Rank_All3: Fixed-20M Uncompounded PnL (USD M)")
    bx3.set_ylabel("Uncompounded PnL ($M)")
    bx4.set_ylabel("Uncompounded PnL ($M)")
    bx3.legend(loc="upper left", fontsize=8.5)
    bx4.legend(loc="upper left", fontsize=8.5)
    bx3.grid(alpha=0.25)
    bx4.grid(alpha=0.25)
    save_fig(fig, fig_dir / "fig08_07_pnl_compounded_vs_fixed20m.png")
