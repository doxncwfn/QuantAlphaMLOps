"""Modal serverless diagnostic runner for v1.3 cross-sectional Russell 1000 ranking model.

Executes deep diagnosis:
- 1.1: Per-date Val vs Test IC time series with rolling 20d mean and autocorrelation.
- 1.2: Train (last 100d) vs Val vs Test Rank IC.
- 1.3: Permutation importance per fold across 19 stock features and 11 context features.
- 1.4: Regime-conditional performance (ctx_mkt_vol_20 quartiles and ctx_mkt_ret_20 sign).
- 1.5: Turnover sensitivity (smoothing alpha in {0.3, 0.5, 0.7, 1.0}, 10d rebalance, position limits).
- 1.6: Strict universe benchmark comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("russell1000-v13-diagnosis")

    # Volume storing the pre-processed feature stores and training artifacts
    volume = modal.Volume.from_name("russell1000-data", create_if_missing=True)

    # Remote GPU environment image
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.2.2",
            "numpy==1.26.4",
            "pandas==2.2.2",
            "polars==0.20.31",
            "scipy==1.12.0",
            "pyarrow==15.0.2",
            "duckdb==1.1.3",
            "wandb>=0.17.0",
        )
        .add_local_dir("src", remote_path="/root/src")
    )

    @app.function(
        image=image,
        gpu="A10",
        volumes={"/data": volume},
        secrets=[modal.Secret.from_dotenv()],
        timeout=7200,
    )
    def diagnose_fold_remote(
        fold_idx: int,
        table_name: str = "silver.features_daily",
        use_wandb: bool = True,
        wandb_project: str = "russell1000-v13-diagnosis",
    ) -> dict:
        """Train and deeply diagnose a single NF-WFCV fold on remote GPU."""
        import sys

        import torch

        sys.path.insert(0, "/root")
        from src.data.loader import (
            FeatureDataStore,
            Russell1000Dataset,
            create_dataloader,
        )
        from src.models.architecture import BaselineArchitectureY
        from src.training.diagnostics import (
            compute_ic_time_series,
            compute_permutation_importance,
            compute_train_val_test_ic,
            evaluate_regimes,
            evaluate_turnover_sensitivity,
        )
        from src.training.evaluate import evaluate_predictions
        from src.training.trainer import collect_predictions, train_one_fold
        from src.training.wfcv import purged_walk_forward_cv

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}"
        )

        # 1. Load Data Store from Modal Volume
        store = FeatureDataStore(
            table_name=table_name,
            cache_dir="/data",
        )
        eval_dates = store.unique_dates[store.eval_date_indices]
        splits = list(
            purged_walk_forward_cv(eval_dates, n_splits=5, horizon=5, embargo=5)
        )

        target_fold = None
        for s in splits:
            if s.fold_idx == fold_idx:
                target_fold = s
                break
        if target_fold is None:
            raise ValueError(f"Fold {fold_idx} not found in splits")

        ckpt_dir = Path("/data/models/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = ckpt_dir / f"fold_{fold_idx}_best.pt"

        # 2. Checkpoint Loading or Training
        if ckpt_file.exists():
            print(f"Found existing checkpoint at {ckpt_file}. Loading directly...")
            ckpt = torch.load(ckpt_file, map_location=device)
            model = BaselineArchitectureY(
                n_stock_features=19, n_context_features=11, d_model=128
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            best_epoch = ckpt.get("best_epoch", 5)
            best_val_ic = ckpt.get("best_val_ic", 0.0)
            # Compute baseline test metrics
            test_ds = Russell1000Dataset(store, target_fold.test_dates)
            test_loader = create_dataloader(test_ds, batch_size=8, shuffle=False)
            test_scores, test_t5, test_mask, test_pn = collect_predictions(
                model, test_loader, device
            )
            test_metrics = evaluate_predictions(
                test_scores, test_t5, test_mask, test_pn, hac_lags=4
            )
        else:
            print(f"No checkpoint found. Training Fold {fold_idx}...")
            res = train_one_fold(
                fold=target_fold,
                store=store,
                device=device,
                batch_size=8,
                grad_accum_steps=8,
                max_epochs=30,
                patience=10,
                warmup_epochs=5,
                checkpoint_dir=str(ckpt_dir),
                seed=42,
                verbose=True,
                use_wandb=use_wandb,
            )
            best_epoch = res["best_epoch"]
            best_val_ic = res["best_val_ic"]
            test_metrics = res["metrics"]
            ckpt = torch.load(ckpt_file, map_location=device)
            model = BaselineArchitectureY(
                n_stock_features=19, n_context_features=11, d_model=128
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])

        model.eval()

        # 3. Diagnostic 1.2: Train (last 100) vs Val vs Test IC
        print(f"--- Diagnosing Fold {fold_idx}: Train/Val/Test IC ---")
        train_val_test = compute_train_val_test_ic(model, store, target_fold, device)

        # 4. Diagnostic 1.1: Per-Date Val & Test IC Time Series
        print(f"--- Diagnosing Fold {fold_idx}: Time Series IC ---")
        val_ds = Russell1000Dataset(store, target_fold.val_dates)
        val_loader = create_dataloader(val_ds, batch_size=8, shuffle=False)
        val_scores, val_t5, val_mask, _val_pn = collect_predictions(
            model, val_loader, device
        )
        val_ic_ts = compute_ic_time_series(
            val_scores, val_t5, val_mask, target_fold.val_dates
        )

        test_ds = Russell1000Dataset(store, target_fold.test_dates)
        test_loader = create_dataloader(test_ds, batch_size=8, shuffle=False)
        test_scores, test_t5, test_mask, test_pn = collect_predictions(
            model, test_loader, device
        )
        test_ic_ts = compute_ic_time_series(
            test_scores, test_t5, test_mask, target_fold.test_dates
        )

        # 5. Diagnostic 1.3: Permutation Importance
        print(f"--- Diagnosing Fold {fold_idx}: Permutation Importance (10 seeds) ---")
        perm_imp = compute_permutation_importance(
            model=model,
            store=store,
            test_dates=target_fold.test_dates,
            device=device,
            n_permutations=10,
            seed=42,
        )

        # 6. Diagnostic 1.4: Regime-Conditional Performance
        print(f"--- Diagnosing Fold {fold_idx}: Regime Performance ---")
        regimes = evaluate_regimes(
            scores=test_scores,
            target_5d=test_t5,
            mask=test_mask,
            permno_matrix=test_pn,
            dates=target_fold.test_dates,
            store=store,
        )

        # 7. Diagnostic 1.5: Turnover Sensitivity
        print(f"--- Diagnosing Fold {fold_idx}: Turnover Sensitivity ---")
        turnover_sens = evaluate_turnover_sensitivity(
            scores=test_scores,
            target_5d=test_t5,
            mask=test_mask,
            permno_matrix=test_pn,
        )

        fold_diag = {
            "fold_idx": fold_idx,
            "table_name": table_name,
            "best_epoch": best_epoch,
            "best_val_ic": best_val_ic,
            "train_val_test_ic": train_val_test,
            "val_ic_time_series": val_ic_ts,
            "test_ic_time_series": test_ic_ts,
            "permutation_importance": perm_imp,
            "regime_performance": regimes,
            "turnover_sensitivity": turnover_sens,
            "test_metrics": test_metrics,
        }

        # Save fold diagnosis to volume
        diag_dir = Path("/data/diagnosis")
        diag_dir.mkdir(parents=True, exist_ok=True)
        out_file = diag_dir / f"{table_name.replace('.', '_')}_fold_{fold_idx}.json"
        with open(out_file, "w") as f:
            clean_diag = json.loads(json.dumps(fold_diag, default=str))
            json.dump(clean_diag, f, indent=2)

        volume.commit()
        return fold_diag

    @app.function(
        image=image,
        gpu="A10",
        volumes={"/data": volume},
        secrets=[modal.Secret.from_dotenv()],
        timeout=7200,
    )
    def train_strict_fold_remote(
        fold_idx: int,
        use_wandb: bool = True,
    ) -> dict:
        """Train a single fold on strict universe table silver.features_daily_v1.3_strict."""
        import sys

        import torch

        sys.path.insert(0, "/root")
        from src.data.loader import FeatureDataStore
        from src.training.trainer import train_one_fold
        from src.training.wfcv import purged_walk_forward_cv

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        store = FeatureDataStore(
            table_name="silver.features_daily_v1.3_strict",
            cache_dir="/data",
        )
        eval_dates = store.unique_dates[store.eval_date_indices]
        splits = list(
            purged_walk_forward_cv(eval_dates, n_splits=5, horizon=5, embargo=5)
        )

        target_fold = next(s for s in splits if s.fold_idx == fold_idx)
        res = train_one_fold(
            fold=target_fold,
            store=store,
            device=device,
            batch_size=8,
            grad_accum_steps=8,
            max_epochs=30,
            patience=10,
            warmup_epochs=5,
            checkpoint_dir="/data/models/checkpoints_strict",
            seed=42,
            verbose=True,
            use_wandb=use_wandb,
        )

        diag_dir = Path("/data/diagnosis")
        diag_dir.mkdir(parents=True, exist_ok=True)
        out_file = diag_dir / f"strict_fold_{fold_idx}.json"
        with open(out_file, "w") as f:
            clean_res = json.loads(json.dumps(res, default=str))
            json.dump(clean_res, f, indent=2)

        volume.commit()
        return res

    @app.local_entrypoint()
    def main(
        folds: str = "all",
        run_strict: bool = True,
    ):
        """CLI entrypoint to launch full parallel diagnosis on Modal."""
        target_folds = (
            [1, 2, 3, 4]
            if folds == "all"
            else [int(f.strip()) for f in folds.split(",")]
        )

        print(
            f"Launching parallel diagnostic run for folds {target_folds} on Modal A10 GPUs..."
        )
        diag_results = list(diagnose_fold_remote.map(target_folds))
        print("Completed base universe diagnosis across all target folds!")

        strict_results = []
        if run_strict:
            print(
                f"Launching parallel strict universe comparison for folds {target_folds} on Modal A10 GPUs..."
            )
            strict_results = list(train_strict_fold_remote.map(target_folds))
            print("Completed strict universe comparison across all target folds!")

        full_output = {
            "base_diagnosis": diag_results,
            "strict_results": strict_results,
        }

        local_out = Path("data/cache/full_diagnosis_results.json")
        with open(local_out, "w") as f:
            clean_out = json.loads(json.dumps(full_output, default=str))
            json.dump(clean_out, f, indent=2)

        print(f"Full diagnostic results saved to {local_out}")

else:

    def main():
        print("Modal client not installed locally.")
