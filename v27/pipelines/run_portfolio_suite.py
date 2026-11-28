"""Pipeline 3 CLI: Signal Analysis, Explainability, Strategy Experiments, Optimization, Overlays & Plots."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import polars as pl

from v27.backtest.statistics import evaluate_and_export_tables
from v27.config import (
    CANDIDATES_24,
    EXPERIMENTS_V7,
    PathConfig,
    PortfolioBacktestConfig,
    RiskOverlayConfig,
)
from v27.data.bundle import load_v7_data_bundle
from v27.experiments.ablations import run_section10_ablations
from v27.experiments.method_a_static import build_method_a_static
from v27.experiments.method_b_score_blend import build_method_b_score_ensembles
from v27.experiments.method_e_dynamic import build_method_e_dynamic
from v27.experiments.method_f_hierarchical import build_method_f_hierarchical
from v27.experiments.parity_audit import run_24_candidate_parity_audit
from v27.explainability.complementarity import (
    build_pairwise_complementarity,
    compute_position_overlap_and_tail,
    compute_return_correlations,
)
from v27.explainability.factor_exposure import compute_3f_attribution
from v27.explainability.gate_attribution import summarize_master_gate_clusters
from v27.explainability.regime_analysis import (
    classify_market_regimes,
    compute_regime_sharpes,
)
from v27.explainability.risk_contribution import compute_risk_contribution_and_exposures
from v27.optimization.method_d_opt import build_method_d_risk_and_opt
from v27.portfolio.constructors import select_orthogonal_subset
from v27.risk.method_g_overlay import build_method_g_risk_overlays
from v27.signal_analysis.alpha_decay import compute_alpha_decay
from v27.signal_analysis.decile_monotonicity import audit_decile_monotonicity_all
from v27.signal_analysis.horizon_audit import (
    audit_risk_t5_horizon_accounting,
    audit_signal_1d_vs_5d,
)
from v27.signal_analysis.score_persistence import (
    compute_dynamic_persistence_precheck,
    compute_score_rank_correlations,
)
from v27.visualization.explainability_plots import (
    plot_alpha_decay,
    plot_corr_overlap_clusters,
    plot_decile_monotonicity,
    plot_ic_and_decay_profile,
    plot_master_gate_clusters,
    plot_regime_heatmap,
    plot_risk_and_beta_alpha,
    plot_rolling_correlations,
)
from v27.visualization.pnl_plots import plot_pnl_dollar_suite
from v27.visualization.portfolio_plots import plot_portfolio_evaluation_suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full v27 portfolio, optimization & evaluation suite."
    )
    parser.add_argument("--qh-dir", type=Path, default=PathConfig().qh_v7_dir)
    parser.add_argument("--output-dir", type=Path, default=PathConfig().output_dir)
    args = parser.parse_args()

    t0 = time.perf_counter()
    paths = PathConfig(qh_v7_dir=args.qh_dir, output_dir=args.output_dir)
    bt_cfg = PortfolioBacktestConfig()
    risk_cfg = RiskOverlayConfig()
    for d in (paths.tables_dir, paths.figures_dir, paths.weights_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading v7 data bundle...")
    bundle = load_v7_data_bundle(
        qh_dir=paths.qh_v7_dir, warehouse_path=paths.warehouse_path
    )

    print("[2/7] Auditing 24 candidates, 1d vs 5d signals, and risk_t5 accounting...")
    cand_weights, cand_bts, audit_df = run_24_candidate_parity_audit(
        bundle, paths.qh_v7_dir, paths.tables_dir, bt_cfg
    )
    if audit_df["GS_v7_NetSh2"].notna().any():
        max_d_sh = (audit_df["Legacy_NetSh2"] - audit_df["GS_v7_NetSh2"]).abs().max()
        print(
            f"      Parity max |Delta_NetSh@2bps| vs grand_summary_v7 = {max_d_sh:.4f}"
        )
    audit_signal_1d_vs_5d(bundle, paths.tables_dir, ann_days=bt_cfg.ann_days)
    audit_decile_monotonicity_all(bundle, paths.tables_dir, ann_days=bt_cfg.ann_days)
    plot_ic_and_decay_profile(
        bundle,
        paths.figures_dir / "fig03_ic_and_decay_profile.png",
        ann_days=bt_cfg.ann_days,
    )
    plot_decile_monotonicity(
        bundle,
        paths.figures_dir / "fig03_decile_monotonicity.png",
        ann_days=bt_cfg.ann_days,
    )
    audit_risk_t5_horizon_accounting(bundle, paths.tables_dir, bt_cfg)
    for exp in EXPERIMENTS_V7:
        df_dec, ic_lag = compute_alpha_decay(
            bundle.scores[exp],
            bundle.ret1d_df,
            bundle.sigma_prev_df,
            bt_cfg.baseline_cost_bps,
            bt_cfg.hold_days,
            bt_cfg.ann_days,
        )
        df_dec.to_csv(paths.tables_dir / f"sec3_alpha_decay_{exp}.csv")
        plot_alpha_decay(
            df_dec,
            ic_lag,
            exp,
            bt_cfg.baseline_cost_bps,
            paths.figures_dir / f"fig03_alpha_decay_{exp}.png",
        )
        gate_pq = paths.qh_v7_dir / f"gate_alpha_{exp}.parquet"
        if gate_pq.exists():
            gate_pd = pl.read_parquet(gate_pq).to_pandas()
            if "date" in gate_pd.columns:
                gate_pd = gate_pd.set_index("date")
            cl_df = summarize_master_gate_clusters(
                gate_pd, out_csv=paths.tables_dir / f"sec3_gate_clusters_{exp}.csv"
            )
            plot_master_gate_clusters(
                cl_df, exp, paths.figures_dir / f"fig03_gate_alpha_{exp}.png"
            )

    print("[3/7] Running explainability & complementarity diagnostics...")
    net_df = pd.DataFrame(
        {c: cand_bts[c]["net"] for c in CANDIDATES_24}, index=bundle.dates
    )
    corrs = compute_return_correlations(
        net_df, bundle.factors_3f["MKT"], paths.tables_dir
    )
    compute_score_rank_correlations(bundle, paths.tables_dir)
    overlaps = compute_position_overlap_and_tail(
        bundle, cand_weights, net_df, paths.tables_dir
    )
    build_pairwise_complementarity(corrs, overlaps, paths.tables_dir)
    df_rc = compute_risk_contribution_and_exposures(
        bundle,
        cand_weights,
        cand_bts,
        net_df,
        overlaps["tail_dep"],
        paths.tables_dir,
        bt_cfg.ann_days,
    )
    df_fac = compute_3f_attribution(bundle, net_df, paths.tables_dir, bt_cfg.ann_days)
    reg_sh = compute_regime_sharpes(
        net_df, classify_market_regimes(bundle), paths.tables_dir, bt_cfg.ann_days
    )
    plot_corr_overlap_clusters(
        corrs["pearson"],
        overlaps["ov_all"],
        paths.figures_dir / "fig04_01_corr_overlap_clusters.png",
    )
    plot_rolling_correlations(
        net_df, paths.figures_dir / "fig04_02_rolling_correlations.png"
    )
    plot_risk_and_beta_alpha(
        df_rc, df_fac, paths.figures_dir / "fig04_03_risk_and_beta_alpha.png"
    )
    plot_regime_heatmap(reg_sh, paths.figures_dir / "fig04_04_regime_heatmap_24.png")

    ortho_cols = select_orthogonal_subset(
        net_df, warmup=bt_cfg.warmup_days, corr_thresh=0.88, ann_days=bt_cfg.ann_days
    )
    print(
        f"      Causal warmup orthogonal core ({len(ortho_cols)}/{len(CANDIDATES_24)}): {ortho_cols}"
    )

    print("[4/7] Running Methods A, B, D, E, F...")
    res_a = build_method_a_static(bundle, cand_weights, net_df, bt_cfg)
    res_b, _ = build_method_b_score_ensembles(bundle, paths.tables_dir, bt_cfg)
    res_d = build_method_d_risk_and_opt(
        bundle, cand_weights, net_df, ortho_cols, bt_cfg.warmup_days, bt_cfg.purge_days
    )
    compute_dynamic_persistence_precheck(
        net_df, paths.tables_dir, bt_cfg.purge_days, bt_cfg.ann_days
    )
    res_e = build_method_e_dynamic(bundle, cand_weights, net_df, ortho_cols, bt_cfg)
    res_f = build_method_f_hierarchical(bundle, cand_weights, net_df, bt_cfg)

    def _oos_sh(res_item: dict) -> float:
        r = res_item["bt"]["net"].iloc[bt_cfg.warmup_days :]
        return float(r.mean() / (r.std(ddof=1) + 1e-12) * math.sqrt(bt_cfg.ann_days))

    f_best = max(res_f, key=_oos_sh)
    d_ortho = next((x for x in res_d if x["name"] == "D2_ExpERC_LW_Ortho"), res_d[0])
    b_best = max(res_b, key=_oos_sh)
    e_best = max(res_e, key=_oos_sh)

    print(
        f"[5/7] Applying Method G risk overlays ({f_best['name']}, {d_ortho['name']}, {b_best['name']}, {e_best['name']})..."
    )
    res_g = (
        build_method_g_risk_overlays(f_best, "G_HierBest", bundle, bt_cfg, risk_cfg)
        + build_method_g_risk_overlays(d_ortho, "G_ERCOrtho", bundle, bt_cfg, risk_cfg)
        + build_method_g_risk_overlays(b_best, "G_ScoreBest", bundle, bt_cfg, risk_cfg)
        + build_method_g_risk_overlays(e_best, "G_DynBest", bundle, bt_cfg, risk_cfg)
    )
    all_ensembles = res_a + res_b + res_d + res_e + res_f + res_g
    flagship = next(
        (x for x in res_g if x["name"] == "G_HierBest_G5_FullConstrainedOverlay"),
        res_g[-1],
    )

    print("[6/7] Running Section 10 ablations...")
    abl_df = run_section10_ablations(
        bundle, cand_weights, net_df, flagship, paths.tables_dir, bt_cfg
    )

    print(
        "[7/7] Evaluating significance, cost sensitivity, FF6, weights & full chart suite..."
    )
    eval_out = evaluate_and_export_tables(
        bundle,
        cand_weights,
        cand_bts,
        all_ensembles,
        paths.tables_dir,
        paths.weights_dir,
        bt_cfg,
    )
    plot_portfolio_evaluation_suite(bundle, eval_out, abl_df, paths.figures_dir, bt_cfg)
    plot_pnl_dollar_suite(bundle, eval_out["all_entries"], paths.figures_dir, bt_cfg)

    lb = eval_out["leaderboard"]
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
    print(
        f"\nCompleted in {time.perf_counter() - t0:.1f}s -> outputs saved under {paths.output_dir}"
    )
    print(lb[cols_show].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
