"""Isolated end-to-end runner for Stock Ranking v27 turnover reduction & cost-resilient optimization."""

from __future__ import annotations

import time
from pathlib import Path

from src.qam.v7_cost_opt.pipeline import run_cost_resilient_experiments
from src.qam.v7_cost_opt.reporting import evaluate_and_save_all
from src.qam.v7_ensemble.engine import load_v7_data_bundle


def main() -> None:
    t0 = time.time()
    qh_dir = Path("outputs/QH_v7")
    out_dir = qh_dir / "v7_cost_resilient"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_v7_data_bundle(qh_dir=qh_dir)
    entries, sig_decay_df, ablation_specs = run_cost_resilient_experiments(bundle)
    results = evaluate_and_save_all(bundle, entries, sig_decay_df, ablation_specs, out_dir)

    lb = results["leaderboard"]
    cols = [
        "Strategy",
        "GoalPass",
        "Gross_Sharpe",
        "Net2_Sharpe",
        "Net5_Sharpe",
        "AllIn2_Sharpe",
        "AllIn5_Sharpe",
        "CAGR2(%)",
        "Comp_MaxDD2(%)",
        "Daily1Way(%)",
        "FF6_Alpha_t",
    ]
    print("\n=== TOP 20 STRATEGIES (SORTED BY GOAL PASS -> NET5 SHARPE -> NET2 SHARPE) ===")
    print(lb[cols].head(20).to_string(index=False))
    print("\n=== FROZEN v27 BASELINES ===")
    print(lb[lb["Group"] == "v27_Baseline"][cols].to_string(index=False))
    print(f"\n[Done] Saved all isolated outputs to {out_dir} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
