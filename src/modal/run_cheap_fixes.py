"""Modal serverless runner to evaluate the 3 cheap baseline fixes across all 4 folds.

Fix 1: Warmup=15 + 3-epoch EMA validation IC checkpointing + min convergence guard (epoch >= 15).
Fix 2: Lower LR (3e-4 from 1e-3) + Warmup=10 + Cosine decay to 1e-5.
Fix 3: Huber auxiliary (lambda_hub = 0.1) on current ListMLE + Softplus tail loss.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("russell1000-v14-cheap-fixes")

    volume = modal.Volume.from_name("russell1000-data", create_if_missing=True)

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
    def train_and_diagnose_fix_remote(
        fix_id: int,
        fold_idx: int,
        use_wandb: bool = True,
    ) -> dict:
        """Train and evaluate one fold under a specific cheap fix configuration."""
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
        from src.training.trainer import collect_predictions, train_one_fold
        from src.training.wfcv import purged_walk_forward_cv

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"--- Starting Fix {fix_id} Fold {fold_idx} on {torch.cuda.get_device_name(0)} ---"
        )

        # 1. Load Data Store from Modal Volume
        store = FeatureDataStore(
            table_name="silver.features_daily",
            cache_dir="/data",
        )
        eval_dates = store.unique_dates[store.eval_date_indices]
        splits = list(
            purged_walk_forward_cv(eval_dates, n_splits=5, horizon=5, embargo=5)
        )
        target_fold = next(s for s in splits if s.fold_idx == fold_idx)

        ckpt_dir = Path("/data/models/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = f"v1.4_fix{fix_id}_fold{fold_idx}.pt"

        # 2. Configure Fix Parameters
        if fix_id == 1:
            # Fix 1: Warmup=15, 3-epoch moving average validation IC, minimum convergence epoch 15
            warmup_epochs = 15
            val_ma_window = 3
            min_checkpoint_epoch = 15
            learning_rate = 1e-3
            eta_min = None
            lambda_hub = 0.0
            wandb_proj = "russell1000-v14-fix1-warmup"
        elif fix_id == 2:
            # Fix 2: Lower LR=3e-4, Warmup=10, Cosine decay to 1e-5
            warmup_epochs = 10
            val_ma_window = 1
            min_checkpoint_epoch = 0
            learning_rate = 3e-4
            eta_min = 1e-5
            lambda_hub = 0.0
            wandb_proj = "russell1000-v14-fix2-lower-lr"
        elif fix_id == 3:
            # Fix 3: Huber auxiliary lambda_hub=0.1
            warmup_epochs = 5
            val_ma_window = 1
            min_checkpoint_epoch = 0
            learning_rate = 1e-3
            eta_min = None
            lambda_hub = 0.1
            wandb_proj = "russell1000-v14-fix3-huber"
        else:
            raise ValueError(f"Unknown fix_id: {fix_id}")

        print(
            f"Fix {fix_id} Fold {fold_idx} Config: lr={learning_rate}, warmup={warmup_epochs}, "
            f"val_ma_window={val_ma_window}, min_ckpt_epoch={min_checkpoint_epoch}, lambda_hub={lambda_hub}"
        )

        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project=wandb_proj,
                    name=f"fix{fix_id}_fold{fold_idx}",
                    config={
                        "fix_id": fix_id,
                        "fold_idx": fold_idx,
                        "learning_rate": learning_rate,
                        "warmup_epochs": warmup_epochs,
                        "val_ma_window": val_ma_window,
                        "min_checkpoint_epoch": min_checkpoint_epoch,
                        "lambda_hub": lambda_hub,
                    },
                    reinit=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"W&B init skipped: {e}")

        train_res = train_one_fold(
            fold=target_fold,
            store=store,
            device=device,
            batch_size=8,
            grad_accum_steps=8,
            learning_rate=learning_rate,
            weight_decay=1e-4,
            max_epochs=30,
            patience=10,
            warmup_epochs=warmup_epochs,
            val_ma_window=val_ma_window,
            min_checkpoint_epoch=min_checkpoint_epoch,
            eta_min=eta_min,
            lambda_hub=lambda_hub,
            checkpoint_dir=str(ckpt_dir),
            checkpoint_filename=ckpt_name,
            seed=42,
            verbose=True,
            use_wandb=use_wandb,
        )

        best_epoch = train_res["best_epoch"]
        best_val_ic = train_res["best_val_ic"]
        history = train_res["history"]

        # 3. Load Checkpoint Model for Complete Diagnostic Suite
        best_ckpt_file = ckpt_dir / ckpt_name
        ckpt_data = torch.load(best_ckpt_file, map_location=device)
        model = BaselineArchitectureY(
            n_stock_features=19,
            n_context_features=11,
            d_model=128,
            lstm_hidden=128,
            lstm_layers=2,
            context_hidden=32,
            n_heads=4,
            tfm_layers=2,
            ff_dim=256,
            dropout=0.1,
        ).to(device)
        model.load_state_dict(ckpt_data["model_state_dict"])
        model.eval()

        # Collect Test Predictions
        test_ds = Russell1000Dataset(store, target_fold.test_dates)
        test_loader = create_dataloader(test_ds, batch_size=8, shuffle=False)
        test_scores, test_t5, test_mask, test_pn = collect_predictions(
            model, test_loader, device
        )

        # 4. Diagnostic: Train vs Val vs Test IC
        print(
            f"--- Fix {fix_id} Fold {fold_idx}: Computing Train vs Val vs Test IC ---"
        )
        tvt_ic = compute_train_val_test_ic(model, store, target_fold, device)

        # 5. Diagnostic: Val & Test IC Time Series
        val_ds = Russell1000Dataset(store, target_fold.val_dates)
        val_loader = create_dataloader(val_ds, batch_size=8, shuffle=False)
        val_scores, val_t5, val_mask, _ = collect_predictions(model, val_loader, device)

        val_ic_ts = compute_ic_time_series(
            val_scores, val_t5, val_mask, target_fold.val_dates
        )
        test_ic_ts = compute_ic_time_series(
            test_scores, test_t5, test_mask, target_fold.test_dates
        )

        # 6. Diagnostic: Permutation Importance
        print(f"--- Fix {fix_id} Fold {fold_idx}: Computing Permutation Importance ---")
        perm_imp = compute_permutation_importance(
            model=model,
            store=store,
            test_dates=target_fold.test_dates,
            device=device,
            n_permutations=5,
            seed=42,
        )

        # 7. Diagnostic: Regime Performance
        print(f"--- Fix {fix_id} Fold {fold_idx}: Evaluating Regime Performance ---")
        regime_perf = evaluate_regimes(
            scores=test_scores,
            target_5d=test_t5,
            mask=test_mask,
            permno_matrix=test_pn,
            dates=target_fold.test_dates,
            store=store,
        )

        # 8. Diagnostic: Turnover Sensitivity
        print(f"--- Fix {fix_id} Fold {fold_idx}: Evaluating Turnover Sensitivity ---")
        turnover_sens = evaluate_turnover_sensitivity(
            scores=test_scores,
            target_5d=test_t5,
            mask=test_mask,
            permno_matrix=test_pn,
        )

        fold_result = {
            "fix_id": fix_id,
            "fold_idx": fold_idx,
            "best_epoch": best_epoch,
            "best_val_ic": best_val_ic,
            "train_time_sec": train_res.get("train_time_sec", 0.0),
            "metrics_stride5": train_res["metrics_stride5"],
            "metrics_stride10": train_res["metrics_stride10"],
            "train_val_test_ic": tvt_ic,
            "val_ic_time_series": val_ic_ts,
            "test_ic_time_series": test_ic_ts,
            "permutation_importance": perm_imp,
            "regime_performance": regime_perf,
            "turnover_sensitivity": turnover_sens,
            "history": history,
        }

        # Save individual fold result to volume
        diag_dir = Path("/data/diagnosis")
        diag_dir.mkdir(parents=True, exist_ok=True)
        out_file = diag_dir / f"v1.4_fix{fix_id}_fold_{fold_idx}.json"
        with open(out_file, "w") as f:
            clean_res = json.loads(json.dumps(fold_result, default=str))
            json.dump(clean_res, f, indent=2)

        volume.commit()
        if use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:  # noqa: BLE001, S110
                pass
        print(f"Completed Fix {fix_id} Fold {fold_idx} successfully!")
        return fold_result

    @app.local_entrypoint()
    def main(
        fixes: str = "all",
        folds: str = "all",
    ):
        """CLI entrypoint to launch parallel execution of cheap fixes on Modal."""
        target_fixes = (
            [1, 2, 3] if fixes == "all" else [int(x.strip()) for x in fixes.split(",")]
        )
        target_folds = (
            [1, 2, 3, 4]
            if folds == "all"
            else [int(x.strip()) for x in folds.split(",")]
        )

        task_pairs = [(fix, fold) for fix in target_fixes for fold in target_folds]
        print(
            f"Launching {len(task_pairs)} parallel training & diagnostic tasks on Modal A10 GPUs..."
        )
        print(f"Tasks: {task_pairs}")

        # Run all tasks in parallel using starmap
        fix_results = list(train_and_diagnose_fix_remote.starmap(task_pairs))

        # Organize by fix_id
        grouped: dict[str, list[dict]] = {f"fix_{fix}": [] for fix in target_fixes}
        for res in fix_results:
            grouped[f"fix_{res['fix_id']}"].append(res)

        # Sort each fix's folds by fold_idx
        for k, v in grouped.items():
            grouped[k] = sorted(v, key=lambda x: x["fold_idx"])

        local_out = Path("data/cache/cheap_fixes_results.json")
        with open(local_out, "w") as f:
            clean_out = json.loads(json.dumps(grouped, default=str))
            json.dump(clean_out, f, indent=2)

        print(
            f"All {len(task_pairs)} tasks complete! Full results saved locally to {local_out}"
        )

else:

    def main():
        print("Modal client not installed locally.")
