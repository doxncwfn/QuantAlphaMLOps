"""End-to-end orchestrator for Stock Ranking v7 Strategy Ensemble & Portfolio Optimization."""

from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd

from src.qam.v7_ensemble.combiners import (
    WARMUP_DAYS,
    _select_orthogonal_subset,
    build_method_a_static,
    build_method_b_score_ensembles,
    build_method_d_risk_and_opt,
    build_method_e_dynamic,
    build_method_f_hierarchical,
    build_method_g_risk_overlays,
    compute_dynamic_persistence_precheck,
    run_section10_ablations,
)
from src.qam.v7_ensemble.diagnostics import run_section3_audit, run_section4_preliminary
from src.qam.v7_ensemble.engine import ANN_DAYS, CANDIDATES_24, load_v7_data_bundle
from src.qam.v7_ensemble.evaluation import run_full_evaluation


def main() -> None:
    t0 = time.perf_counter()
    qh_dir = Path("outputs/QH_v7")
    out_dir = qh_dir / "v7_ensemble"
    tab_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    wgt_dir = out_dir / "weights"
    for d in (tab_dir, fig_dir, wgt_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading v7 data bundle...")
    bundle = load_v7_data_bundle(qh_dir=qh_dir)
    print(
        f"      Panel: {len(bundle.dates)} dates x {len(bundle.security_ids)} securities "
        f"({bundle.dates[0].date()} .. {bundle.dates[-1].date()}), OOS start={bundle.dates[WARMUP_DAYS].date()}"
    )

    print(
        "[2/7] Running Section 3 Backtesting Audit (24 candidates, parity, 1d vs 5d)..."
    )
    cand_weights, cand_bts, audit_df = run_section3_audit(bundle, qh_dir, tab_dir)
    max_d_sh = (audit_df["Legacy_NetSh2"] - audit_df["GS_v7_NetSh2"]).abs().max()
    print(f"      Parity max |Delta_NetSh@2bps| vs GS_v7 = {max_d_sh:.4f}")

    print(
        "[3/7] Running Section 4 Preliminary Analysis (Corr, Overlap, RiskContrib, 3F, Regimes)..."
    )
    run_section4_preliminary(bundle, cand_weights, cand_bts, tab_dir, fig_dir)
    ret_net_df = pd.DataFrame(
        {c: cand_bts[c]["net"] for c in CANDIDATES_24}, index=bundle.dates
    )

    ortho_cols = _select_orthogonal_subset(
        ret_net_df, warmup=WARMUP_DAYS, corr_thresh=0.88
    )
    print(
        f"      Causal warmup orthogonal core ({len(ortho_cols)}/{len(CANDIDATES_24)}): {ortho_cols}"
    )

    print("[4/7] Running Methods A - F Strategy Combinations...")
    res_a = build_method_a_static(bundle, cand_weights, cand_bts, ret_net_df)
    res_b, _decomp_b = build_method_b_score_ensembles(bundle, tab_dir)
    res_d = build_method_d_risk_and_opt(bundle, cand_weights, ret_net_df, ortho_cols)
    persist_df = compute_dynamic_persistence_precheck(ret_net_df, tab_dir)
    res_e = build_method_e_dynamic(bundle, cand_weights, ret_net_df, ortho_cols)
    res_f = build_method_f_hierarchical(bundle, cand_weights, ret_net_df)

    core_ensembles = res_a + res_b + res_d + res_e + res_f

    # Pick flagship Hierarchical / Risk-Controlled base and Return-Oriented Score-Blend base for Method G overlays
    def _oos_sharpe(res_item: dict) -> float:
        r = res_item["bt"]["net"].iloc[WARMUP_DAYS:]
        return float(r.mean() / (r.std(ddof=1) + 1e-12) * math.sqrt(ANN_DAYS))

    f_best = max(res_f, key=_oos_sharpe)
    d_ortho_erc = next(
        (x for x in res_d if x["name"] == "D2_ExpERC_LW_Ortho"), res_d[0]
    )
    b_best = max(res_b, key=_oos_sharpe)
    e_best = max(res_e, key=_oos_sharpe)

    print(
        f"[5/7] Running Method G Incremental Risk Overlays on {f_best['name']}, {d_ortho_erc['name']}, {b_best['name']}, {e_best['name']}..."
    )
    res_g = (
        build_method_g_risk_overlays(f_best, "G_HierBest", bundle)
        + build_method_g_risk_overlays(d_ortho_erc, "G_ERCOrtho", bundle)
        + build_method_g_risk_overlays(b_best, "G_ScoreBest", bundle)
        + build_method_g_risk_overlays(e_best, "G_DynBest", bundle)
    )
    all_ensembles = core_ensembles + res_g

    flagship_full = next(
        (x for x in res_g if x["name"] == "G_HierBest_G5_FullConstrainedOverlay"),
        res_g[-1],
    )

    print("[6/7] Running Section 10 Ablation Study...")
    abl_df = run_section10_ablations(
        bundle, cand_weights, ret_net_df, flagship_full, tab_dir
    )

    print(
        "[7/7] Running Full Evaluation Suite, Significance, Cost Grid, Folds, FF6, Plots & Weights..."
    )
    lb, _ = run_full_evaluation(
        bundle, cand_weights, cand_bts, all_ensembles, abl_df, tab_dir, fig_dir, wgt_dir
    )

    print(
        f"\nCompleted in {time.perf_counter() - t0:.1f}s. Top 12 Leaderboard (OOS Net @ 2 bps):"
    )
    cols_show = [
        "Strategy",
        "Group",
        "CAGR(%)",
        "AnnVol(%)",
        "Net2_Sharpe",
        "AllIn_Sharpe",
        "HAC_Sharpe",
        "Comp_MaxDD(%)",
        "AnnTurn_1Way(%)",
        "Calmar",
    ]
    print(lb[cols_show].head(15).to_string(index=False))

    print("\nDynamic Persistence Pre-check:")
    print(persist_df.to_string(index=False))

    print("\nAblation Summary:")
    print(abl_df.to_string(index=False))


if __name__ == "__main__":
    main()
