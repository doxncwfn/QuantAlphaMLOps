"""Pipeline 4 CLI: Turnover Reduction & Cost-Resilient Portfolio Optimization (Pillars 1-3)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from v27.config import PathConfig
from v27.cost_resilient.pipeline import run_cost_resilient_experiments
from v27.cost_resilient.reporting import evaluate_and_save_all
from v27.data.bundle import load_v7_data_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v27 turnover-reduction and cost-resilient optimization suite."
    )
    parser.add_argument("--qh-dir", type=Path, default=PathConfig().qh_v7_dir)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PathConfig().qh_v7_dir / "v7_cost_resilient",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_v7_data_bundle(qh_dir=args.qh_dir)
    entries, sig_decay_df, ablation_specs = run_cost_resilient_experiments(bundle)
    results = evaluate_and_save_all(
        bundle, entries, sig_decay_df, ablation_specs, args.output_dir
    )

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
    print("\n=== TOP 15 COST-RESILIENT STRATEGIES ===")
    print(lb[cols].head(15).to_string(index=False))
    print(
        f"\n[Done] Saved v27 cost-resilient outputs to {args.output_dir} in {time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
