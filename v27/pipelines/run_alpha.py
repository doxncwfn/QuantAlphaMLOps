"""Pipeline 2 CLI: DuckDB/Cache -> Features -> Targets -> HybridLstmTfm WF Training & Explainability."""

from __future__ import annotations

import argparse
from pathlib import Path

from v27.config import EXPERIMENTS_V7, DataConfig, ModelConfig, PathConfig, TrainConfig
from v27.data.warehouse_loader import build_or_load_dataset
from v27.explainability.gate_attribution import summarize_master_gate_clusters
from v27.targets.target_builder import prepare_alpha_dataset
from v27.training.trainer import run_experiment
from v27.visualization.explainability_plots import plot_master_gate_clusters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v27 alpha generation & explainability pipeline."
    )
    parser.add_argument("--qh-dir", type=Path, default=PathConfig().qh_v7_dir)
    parser.add_argument("--output-dir", type=Path, default=PathConfig().output_dir)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS_V7))
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    paths = PathConfig(qh_v7_dir=args.qh_dir, output_dir=args.output_dir)
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    cache = build_or_load_dataset(
        paths=paths, data_cfg=data_cfg, force_rebuild=args.force_rebuild_cache
    )
    ds = prepare_alpha_dataset(cache, data_cfg=data_cfg)
    for exp_name in args.experiments:
        _, alpha_df, _ = run_experiment(
            exp_name,
            ds,
            paths.alpha_dir,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            resume=not args.no_resume,
        )
        if alpha_df is not None:
            cl_df = summarize_master_gate_clusters(
                alpha_df, out_csv=paths.tables_dir / f"gate_clusters_{exp_name}.csv"
            )
            plot_master_gate_clusters(
                cl_df, exp_name, paths.figures_dir / f"gate_alpha_{exp_name}.png"
            )


if __name__ == "__main__":
    main()
