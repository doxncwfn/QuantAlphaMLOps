"""Standalone runner for §1 loss ablation (Decile Monotonicity + LS Spread).

Can be executed on any machine with GPU (Google Colab, Lambda, RunPod, or local):
    python scripts/run_s1_ablation_standalone.py --variants all --folds all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import FeatureDataStore, Russell1000Dataset, create_dataloader
from src.models.architecture import BaselineArchitectureY
from src.training.diagnostics import (
    compute_decile_diagnostics,
    compute_ic_time_series,
    compute_train_val_test_ic,
)
from src.training.trainer import collect_predictions, train_one_fold
from src.training.wfcv import purged_walk_forward_cv


def run_single_fold(
    variant: str,
    fold_idx: int,
    data_dir: Path,
    ckpt_dir: Path,
    diag_dir: Path,
    device: torch.device,
    use_wandb: bool = False,
    force_rerun: bool = False,
) -> dict:
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_file = diag_dir / f"v1.5_{variant}_fold_{fold_idx}.json"
    if not force_rerun and out_file.exists():
        print(f"Result already exists at {out_file}, loading cached result...")
        try:
            with open(out_file) as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to read cached {out_file}: {e}, re-running...")

    print(f"--- Starting §1 Variant {variant} Fold {fold_idx} on {device} ---")

    store = FeatureDataStore(
        table_name="silver.features_daily",
        cache_dir=str(data_dir),
    )
    eval_dates = store.unique_dates[store.eval_date_indices]
    splits = list(purged_walk_forward_cv(eval_dates, n_splits=5, horizon=5, embargo=5))
    target_fold = next(s for s in splits if s.fold_idx == fold_idx)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"v1.5_{variant}_fold{fold_idx}.pt"

    if variant == "s1a":
        lam_hub = 0.0
        wandb_proj = "russell1000-v15-s1a-no-huber"
    elif variant == "s1b":
        lam_hub = 0.2
        wandb_proj = "russell1000-v15-s1b-with-huber"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    m_mono = 0.02
    lam_ls = 0.5
    tau_sort = 0.01
    K = 10

    # Fix 1 baseline parameters (frozen)
    warmup_epochs = 15
    val_ma_window = 3
    min_checkpoint_epoch = 15
    learning_rate = 1e-3
    weight_decay = 1e-4
    max_epochs = 30
    patience = 10

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

    test_ds = Russell1000Dataset(store, target_fold.test_dates)
    test_loader = create_dataloader(test_ds, batch_size=8, shuffle=False)
    test_scores, test_t5, test_mask, _test_pn = collect_predictions(
        model, test_loader, device
    )

    valid_test_scores = test_scores[test_mask.astype(bool)]
    score_std = float(np.std(valid_test_scores))
    score_mean = float(np.mean(valid_test_scores))

    tvt_ic = compute_train_val_test_ic(model, store, target_fold, device)

    val_ds = Russell1000Dataset(store, target_fold.val_dates)
    val_loader = create_dataloader(val_ds, batch_size=8, shuffle=False)
    val_scores, val_t5, val_mask, _ = collect_predictions(model, val_loader, device)

    val_ic_ts = compute_ic_time_series(
        val_scores, val_t5, val_mask, target_fold.val_dates
    )
    test_ic_ts = compute_ic_time_series(
        test_scores, test_t5, test_mask, target_fold.test_dates
    )

    decile_diag = compute_decile_diagnostics(
        scores=test_scores,
        target=test_t5,
        mask=test_mask,
        K=10,
    )

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
        "decile_diagnostics": decile_diag,
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

    with open(out_file, "w") as f:
        clean_res = json.loads(json.dumps(fold_result, default=str))
        json.dump(clean_res, f, indent=2)

    if use_wandb:
        try:
            import wandb

            wandb.finish()
        except Exception:  # noqa: BLE001, S110
            pass

    print(f"Completed §1 Variant {variant} Fold {fold_idx} successfully!")
    return fold_result


def main():
    parser = argparse.ArgumentParser(description="Run §1 Loss Ablation Standalone")
    parser.add_argument(
        "--variants", type=str, default="all", help="Variants: s1a, s1b, or all"
    )
    parser.add_argument(
        "--folds", type=str, default="all", help="Folds: 1, 2, 3, 4, or all"
    )
    parser.add_argument("--data-dir", type=str, default="data/cache")
    parser.add_argument("--ckpt-dir", type=str, default="models/checkpoints")
    parser.add_argument("--diag-dir", type=str, default="data/cache")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)

    target_variants = (
        ["s1a", "s1b"]
        if args.variants == "all"
        else [x.strip() for x in args.variants.split(",")]
    )
    target_folds = (
        [1, 2, 3, 4]
        if args.folds == "all"
        else [int(x.strip()) for x in args.folds.split(",")]
    )

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    diag_dir = Path(args.diag_dir)

    all_results: dict[str, list[dict]] = {var: [] for var in target_variants}

    for var in target_variants:
        for f_idx in target_folds:
            res = run_single_fold(
                variant=var,
                fold_idx=f_idx,
                data_dir=data_dir,
                ckpt_dir=ckpt_dir,
                diag_dir=diag_dir,
                device=dev,
                use_wandb=args.use_wandb,
                force_rerun=args.force,
            )
            all_results[var].append(res)

    out_file = Path("data/cache/s1_ablation_results.json")
    with open(out_file, "w") as f:
        clean_res = json.loads(json.dumps(all_results, default=str))
        json.dump(clean_res, f, indent=2)

    print(f"\nAll requested tasks completed! Consolidated results saved to {out_file}")


if __name__ == "__main__":
    main()
