"""Modal serverless runner to evaluate §1 Decile Monotonicity + LS Spread loss ablation across 4 folds.

Variant A (s1a): lambda_hub = 0.0 (no Huber auxiliary)
Variant B (s1b): lambda_hub = 0.2 (with Huber auxiliary)

Both variants evaluated against Fix 1 baseline under identical protocol:
- warmup_epochs = 15
- val_ma_window = 3
- min_checkpoint_epoch = 15
- learning_rate = 1e-3
- cosine decay to 1e-5
- architecture Y
- feature set C (19 stock + 11 context)
- universe 1A
- 4-fold NF-WFCV
"""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("russell1000-v15-s1-ablation")

    volume = modal.Volume.from_name("russell1000-data", create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("build-essential")
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
        .run_commands("pip install --no-build-isolation torchsort==0.1.10")
        .add_local_dir("src", remote_path="/root/src")
    )

    wandb_key = os.getenv(
        "WANDB_API_KEY",
        "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",
    )

    @app.function(
        image=image,
        gpu="A10",
        volumes={"/data": volume},
        secrets=[modal.Secret.from_dict({"WANDB_API_KEY": wandb_key})],
        timeout=7200,
    )
    def train_and_diagnose_s1_remote(
        variant: str,
        fold_idx: int,
        use_wandb: bool = True,
        force_rerun: bool = False,
    ) -> dict:
        """Train and evaluate one fold under §1 loss configuration."""
        import sys

        import numpy as np
        import torch

        sys.path.insert(0, "/root")
        from src.data.loader import (
            FeatureDataStore,
            Russell1000Dataset,
            create_dataloader,
        )
        from src.models.architecture import BaselineArchitectureY
        from src.training.diagnostics import (
            compute_decile_diagnostics,
            compute_ic_time_series,
            compute_train_val_test_ic,
        )
        from src.training.trainer import collect_predictions, train_one_fold
        from src.training.wfcv import purged_walk_forward_cv

        diag_dir = Path("/data/diagnosis")
        diag_dir.mkdir(parents=True, exist_ok=True)
        out_file = diag_dir / f"v1.5_{variant}_fold_{fold_idx}.json"
        if not force_rerun and out_file.exists():
            print(f"Result already exists at {out_file}, loading cached result...")
            try:
                with open(out_file) as f:
                    return json.load(f)
            except Exception as e:  # noqa: BLE001
                print(f"Failed to read cached {out_file}: {e}, proceeding to train...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"--- Starting §1 Variant {variant} Fold {fold_idx} on {torch.cuda.get_device_name(0)} ---"
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
        ckpt_name = f"v1.5_{variant}_fold{fold_idx}.pt"

        # 2. Configure Loss Parameters for Variant A vs B
        if variant == "s1a":
            # Variant A: No Huber auxiliary
            lam_hub = 0.0
            wandb_proj = "russell1000-v15-s1a-no-huber"
        elif variant == "s1b":
            # Variant B: With Huber auxiliary
            lam_hub = 0.2
            wandb_proj = "russell1000-v15-s1b-with-huber"
        else:
            raise ValueError(f"Unknown variant: {variant}")

        m_mono = 0.02
        lam_ls = 0.5
        tau_sort = 0.01
        K = 10

        # Protocol identical to Fix 1 baseline
        warmup_epochs = 15
        val_ma_window = 3
        min_checkpoint_epoch = 15
        learning_rate = 1e-3
        weight_decay = 1e-4
        max_epochs = 30
        patience = 10

        print(
            f"§1 Variant {variant} Fold {fold_idx} Config: K={K}, tau_sort={tau_sort}, "
            f"m_mono={m_mono}, lam_ls={lam_ls}, lam_hub={lam_hub} | lr={learning_rate}, "
            f"warmup={warmup_epochs}, val_ma_window={val_ma_window}, min_ckpt={min_checkpoint_epoch}"
        )

        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project=wandb_proj,
                    name=f"{variant}_fold{fold_idx}",
                    config={
                        "variant": variant,
                        "fold_idx": fold_idx,
                        "loss_type": "decile_mono",
                        "K": K,
                        "tau_sort": tau_sort,
                        "m_mono": m_mono,
                        "lam_ls": lam_ls,
                        "lam_hub": lam_hub,
                        "learning_rate": learning_rate,
                        "warmup_epochs": warmup_epochs,
                        "val_ma_window": val_ma_window,
                        "min_checkpoint_epoch": min_checkpoint_epoch,
                    },
                    reinit=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"W&B init skipped: {e}")

        # 3. Train One Fold with Decile Monotonicity Loss
        train_res = train_one_fold(
            fold=target_fold,
            store=store,
            device=device,
            batch_size=8,
            grad_accum_steps=8,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            warmup_epochs=warmup_epochs,
            val_ma_window=val_ma_window,
            min_checkpoint_epoch=min_checkpoint_epoch,
            loss_type="decile_mono",
            loss_kwargs={
                "K": K,
                "tau_sort": tau_sort,
                "m_mono": m_mono,
                "lam_ls": lam_ls,
                "lam_hub": lam_hub,
                "beta": 0.1,
            },
            checkpoint_dir=str(ckpt_dir),
            checkpoint_filename=ckpt_name,
            seed=42,
            verbose=True,
            use_wandb=use_wandb,
        )

        best_epoch = train_res["best_epoch"]
        best_val_ic = train_res["best_val_ic"]
        history = train_res["history"]

        # 4. Load Checkpoint Model for Complete Diagnostic Suite
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
        test_scores, test_t5, test_mask, _test_pn = collect_predictions(
            model, test_loader, device
        )

        # Score std check for rank collapse
        valid_test_scores = test_scores[test_mask.astype(bool)]
        score_std = float(np.std(valid_test_scores))
        score_mean = float(np.mean(valid_test_scores))

        # 5. Diagnostic: Train vs Val vs Test IC
        print(
            f"--- §1 Variant {variant} Fold {fold_idx}: Computing Train vs Val vs Test IC ---"
        )
        tvt_ic = compute_train_val_test_ic(model, store, target_fold, device)

        # 6. Diagnostic: Val & Test IC Time Series
        val_ds = Russell1000Dataset(store, target_fold.val_dates)
        val_loader = create_dataloader(val_ds, batch_size=8, shuffle=False)
        val_scores, val_t5, val_mask, _ = collect_predictions(model, val_loader, device)

        val_ic_ts = compute_ic_time_series(
            val_scores, val_t5, val_mask, target_fold.val_dates
        )
        test_ic_ts = compute_ic_time_series(
            test_scores, test_t5, test_mask, target_fold.test_dates
        )

        # 7. Diagnostic: Decile Monotonicity on Test Set
        print(
            f"--- §1 Variant {variant} Fold {fold_idx}: Computing Decile Monotonicity Diagnostics ---"
        )
        decile_diag_t5 = compute_decile_diagnostics(
            scores=test_scores,
            target=test_t5,
            mask=test_mask,
            K=10,
        )

        # Compute loss trajectory statistics
        init_loss = history[0]["train_loss"] if history else 0.0
        min_loss = min(h["train_loss"] for h in history) if history else 0.0
        final_loss = history[-1]["train_loss"] if history else 0.0
        loss_drop_pct = (init_loss - min_loss) / (abs(init_loss) + 1e-8) * 100.0

        fold_result = {
            "variant": variant,
            "fold_idx": fold_idx,
            "best_epoch": best_epoch,
            "best_val_ic": best_val_ic,
            "train_time_sec": train_res.get("train_time_sec", 0.0),
            "score_mean": score_mean,
            "score_std": score_std,
            "init_loss": init_loss,
            "min_loss": min_loss,
            "final_loss": final_loss,
            "loss_drop_pct": loss_drop_pct,
            "metrics_stride5": train_res["metrics_stride5"],
            "metrics_stride10": train_res["metrics_stride10"],
            "train_val_test_ic": tvt_ic,
            "decile_diagnostics": decile_diag_t5,
            "val_ic_time_series": {
                "mean": val_ic_ts["mean"],
                "std": val_ic_ts["std"],
                "median": val_ic_ts["median"],
                "autocorr_lag1": val_ic_ts["autocorr_lag1"],
            },
            "test_ic_time_series": {
                "mean": test_ic_ts["mean"],
                "std": test_ic_ts["std"],
                "median": test_ic_ts["median"],
                "autocorr_lag1": test_ic_ts["autocorr_lag1"],
            },
            "history": history,
        }

        # Save individual fold result to volume
        diag_dir = Path("/data/diagnosis")
        diag_dir.mkdir(parents=True, exist_ok=True)
        out_file = diag_dir / f"v1.5_{variant}_fold_{fold_idx}.json"
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
        print(f"Completed §1 Variant {variant} Fold {fold_idx} successfully!")
        return fold_result

    @app.local_entrypoint()
    def main(
        variants: str = "all",
        folds: str = "all",
    ):
        """CLI entrypoint to launch parallel execution of §1 ablation on Modal."""
        target_variants = (
            ["s1a", "s1b"]
            if variants == "all"
            else [x.strip() for x in variants.split(",")]
        )
        target_folds = (
            [1, 2, 3, 4]
            if folds == "all"
            else [int(x.strip()) for x in folds.split(",")]
        )

        task_pairs = [(var, fold) for var in target_variants for fold in target_folds]
        print(
            f"Launching {len(task_pairs)} parallel training & diagnostic tasks on Modal A10 GPUs..."
        )
        print(f"Tasks: {task_pairs}")

        # Run all tasks in parallel using starmap
        s1_results = list(train_and_diagnose_s1_remote.starmap(task_pairs))

        # Organize by variant
        grouped: dict[str, list[dict]] = {f"{var}": [] for var in target_variants}
        for res in s1_results:
            grouped[res["variant"]].append(res)

        # Sort each variant's folds by fold_idx
        for k, v in grouped.items():
            grouped[k] = sorted(v, key=lambda x: x["fold_idx"])

        local_out = Path("data/cache/s1_ablation_results.json")
        local_out.parent.mkdir(parents=True, exist_ok=True)
        with open(local_out, "w") as f:
            clean_out = json.loads(json.dumps(grouped, default=str))
            json.dump(clean_out, f, indent=2)

        print(
            f"All {len(task_pairs)} tasks complete! Full results saved locally to {local_out}"
        )

else:

    def main():
        print("Modal client not installed locally.")
