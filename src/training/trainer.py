"""Training pipeline for v1.3 cross-sectional Russell 1000 ranking model.

Bug fixes applied:
- BUG-01: Fix dead import and restore generate_baseline_1a_report integration.
- BUG-02: Add per-date score centering to remove Transformer common-mode bias.
- BUG-06: Training protocol alignment (patience=10, effective batch size=64, early stopping warmup guard at epoch 5, checkpoint disk persistence).
- BUG-07: Hardware determinism enforcement (reject non-deterministic MPS, require CUDA or CPU; monotonic active compute time tracking).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

from src.data.loader import (
    FeatureDataStore,
    Russell1000Dataset,
    create_dataloader,
)
from src.data.normalize_schema import seed_everything
from src.losses.combined import CombinedRankingLoss
from src.models.architecture import BaselineArchitectureY
from src.training.evaluate import evaluate_predictions, generate_baseline_1a_report
from src.training.wfcv import FoldSplit, purged_walk_forward_cv


def center_scores_per_date(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Subtract per-date cross-sectional mean of valid-stock scores.

    Removes the common-mode bias in scores that dominates non-shift-invariant
    losses. ListMLE is invariant to this; Softplus tail is not.

    Score centering removes the per-date common mode in the Cross-Sectional Transformer's
    output, which would otherwise dominate the gradient of any non-shift-invariant loss
    (Softplus tail, MSE). The common mode is not present in ListMLE's gradient because
    ListMLE is shift-invariant. Centering makes the two loss components comparable in
    gradient magnitude without changing the rank ordering of stocks.
    """
    mask_f = mask.float()
    n_valid = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
    mean_per_date = (scores * mask_f).sum(dim=1, keepdim=True) / n_valid  # [B, 1]
    return (
        scores - mean_per_date * mask_f
    )  # only subtract on valid, leave padded unchanged


def get_default_device() -> torch.device:
    """Select best available hardware acceleration device (CUDA / CPU; rejects MPS per BUG-07)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not deterministic for this model. Use CUDA (Modal A10G) or CPU."
        )
    return torch.device("cpu")


def evaluate_rank_ic(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean Spearman Rank IC on a dataloader split."""
    model.eval()
    rank_ics: list[float] = []
    use_autocast = device.type == "cuda"

    with torch.inference_mode():
        for batch in dataloader:
            x_stock = batch["x_stock"].to(device)
            x_ctx = batch["x_ctx"].to(device)
            mask = batch["mask"].to(device)
            t5 = batch["target_5d"].numpy()
            mask_np = batch["mask"].numpy()

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
            ):
                scores_t = model(x_stock, x_ctx, mask)
            scores = scores_t.detach().cpu().float().numpy()
            del scores_t
            B = scores.shape[0]

            for b in range(B):
                valid = mask_np[b]
                if int(valid.sum()) < 10:
                    continue
                s = scores[b][valid]
                y = t5[b][valid]
                res = spearmanr(s, y)
                ic = (
                    float(res.statistic) if hasattr(res, "statistic") else float(res[0])
                )
                if np.isfinite(ic):
                    rank_ics.append(ic)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return float(np.mean(rank_ics)) if rank_ics else 0.0


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect predictions, targets, masks, and permnos across an evaluation split."""
    model.eval()
    all_scores = []
    all_targets = []
    all_masks = []
    all_permnos = []
    use_autocast = device.type == "cuda"

    with torch.inference_mode():
        for batch in dataloader:
            x_stock = batch["x_stock"].to(device)
            x_ctx = batch["x_ctx"].to(device)
            mask = batch["mask"].to(device)

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
            ):
                scores_t = model(x_stock, x_ctx, mask)
            scores = scores_t.detach().cpu().float().numpy()
            del scores_t

            all_scores.append(scores)
            all_targets.append(batch["target_5d"].numpy())
            all_masks.append(batch["mask"].numpy())
            all_permnos.append(batch["permno"].numpy())

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return (
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_masks, axis=0),
        np.concatenate(all_permnos, axis=0),
    )


def train_one_fold(
    fold: FoldSplit,
    store: FeatureDataStore,
    device: torch.device | None = None,
    batch_size: int = 8,
    grad_accum_steps: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 30,
    patience: int = 10,
    warmup_epochs: int = 15,
    checkpoint_dir: str | Path | None = "models/checkpoints",
    seed: int = 42,
    verbose: bool = True,
    use_wandb: bool = False,
    rebalance_stride: int = 5,
    horizon: int = 5,
    allow_mismatched_stride: bool = False,
    val_ma_window: int = 3,
    min_checkpoint_epoch: int | None = None,
    eta_min: float | None = None,
    lambda_hub: float = 0.0,
    checkpoint_filename: str | None = None,
    loss_type: str = "combined",
    loss_kwargs: dict | None = None,
    save_checkpoint: bool = True,
) -> dict[str, Any]:
    """Train Architecture Y on a single NF-WFCV fold with early stopping.

    Parameters
    ----------
    fold : FoldSplit
        Train, validation, and test date partitions.
    store : FeatureDataStore
        Pre-processed feature tensor store.
    device : torch.device | None
        Target computing device.
    batch_size : int
        Micro-batch size in dates (default 8).
    grad_accum_steps : int
        Gradient accumulation steps (default 8 -> effective batch size 64 per §9.7).
    learning_rate : float
        Initial AdamW learning rate (1e-3 per §9.7).
    weight_decay : float
        AdamW weight decay (1e-4 per §9.7).
    max_epochs : int
        Maximum training epochs (30).
    patience : int
        Early stopping patience on validation Rank IC (10 per §9.7).
    warmup_epochs : int
        Warmup guard before early stopping can trigger (5 per BUG-06).
    checkpoint_dir : str | Path | None
        Directory to persist best fold model checkpoints.
    seed : int
        Base random seed.
    verbose : bool
        Whether to print epoch progress.
    use_wandb : bool
        Whether to log training metrics to W&B.

    Returns
    -------
    dict[str, Any]
        Dictionary with test metrics, best epoch, and validation history.
    """
    fold_seed = seed + fold.fold_idx * 1000
    seed_everything(fold_seed)

    eff_min_ckpt = 0 if min_checkpoint_epoch is None else int(min_checkpoint_epoch)
    if eff_min_ckpt > warmup_epochs:
        raise ValueError(
            f"min_checkpoint_epoch ({eff_min_ckpt}) must be <= warmup_epochs ({warmup_epochs})"
        )

    if device is None:
        device = get_default_device()

    train_ds = Russell1000Dataset(store, fold.train_dates)
    val_ds = Russell1000Dataset(store, fold.val_dates)
    test_ds = Russell1000Dataset(store, fold.test_dates)

    train_loader = create_dataloader(
        train_ds, batch_size=batch_size, shuffle=True, seed=fold_seed
    )
    val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_ds, batch_size=batch_size, shuffle=False)

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

    if loss_type == "decile_mono":
        from src.losses.decile_mono import DecileMonotonicityLoss

        lk = loss_kwargs or {}
        loss_fn = DecileMonotonicityLoss(
            K=lk.get("K", 10),
            tau_sort=lk.get("tau_sort", 0.01),
            tau_assign=lk.get("tau_assign", None),
            m_mono=lk.get("m_mono", 0.02),
            lam_ls=lk.get("lam_ls", 0.5),
            lam_hub=lk.get("lam_hub", 0.0),
            beta=lk.get("beta", 1.0),
        )
    else:
        loss_fn = CombinedRankingLoss(
            lambda_tail=1.5, tail_threshold=0.4, lambda_hub=lambda_hub
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=learning_rate * 0.01,
    )

    best_val_ic = -float("inf")
    best_weights = None
    best_epoch = 0
    fallback_val_ic = -float("inf")
    fallback_weights = None
    fallback_epoch = 0
    patience_counter = 0
    history: list[dict[str, float]] = []

    t_wall_start = time.time()
    t_mono_start = time.monotonic()
    effective_bs = batch_size * grad_accum_steps
    if verbose:
        print(
            f"--- Starting Fold {fold.fold_idx} --- "
            f"Train dates: {len(train_ds)} ({str(fold.train_start)[:10]} to {str(fold.train_end)[:10]}), "
            f"Val dates: {len(val_ds)} ({str(fold.val_start)[:10]} to {str(fold.val_end)[:10]}), "
            f"Test dates: {len(test_ds)} ({str(fold.test_start)[:10]} to {str(fold.test_end)[:10]}) | "
            f"Batch size: {batch_size}, Grad accum: {grad_accum_steps} (Effective batch size: {effective_bs}) | "
            f"Patience: {patience}, Warmup epochs: {warmup_epochs}"
        )

    use_autocast = device.type == "cuda"
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_mle_accum = 0.0
        train_tail_accum = 0.0
        train_hub_accum = 0.0
        n_batches = 0

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            x_stock = batch["x_stock"].to(device)
            x_ctx = batch["x_ctx"].to(device)
            mask = batch["mask"].to(device)
            t5 = batch["target_5d"].to(device)
            ys = batch["y_star"].to(device)
            yn = batch["y_norm"].to(device) if "y_norm" in batch else ys

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
            ):
                scores = model(x_stock, x_ctx, mask)

            with torch.autocast(device_type=device.type, enabled=False):
                scores = scores.float()
                t5 = t5.float()
                ys = ys.float()
                yn = yn.float()
                mask = mask.float()
                scores = center_scores_per_date(scores, mask)
                loss, l_mle, l_tail = loss_fn(scores, t5, ys, mask, y_norm=yn)
                scaled_loss = loss / grad_accum_steps

            scaled_loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(
                train_loader
            ):
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                if not torch.isfinite(total_norm):
                    logger.warning(
                        "Non-finite gradient norm (%s) at date-batch index %d; skipping optimizer step.",
                        total_norm,
                        batch_idx,
                    )
                    optimizer.zero_grad()
                    continue
                optimizer.step()
                optimizer.zero_grad()

            train_loss_accum += float(loss.item())
            train_mle_accum += float(l_mle.item())
            train_tail_accum += float(l_tail.item())
            if hasattr(loss_fn, "last_diag") and "L_hub" in loss_fn.last_diag:
                train_hub_accum += float(loss_fn.last_diag["L_hub"].item())
            n_batches += 1

        scheduler.step()
        train_loss = train_loss_accum / max(1, n_batches)
        train_mle = train_mle_accum / max(1, n_batches)
        train_tail = train_tail_accum / max(1, n_batches)
        train_hub = train_hub_accum / max(1, n_batches)

        # Clear device memory caches
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

        # Validation step
        val_rank_ic = evaluate_rank_ic(model, val_loader, device)
        val_history = [h["val_rank_ic"] for h in history] + [val_rank_ic]
        if val_ma_window > 1:
            val_metric = float(np.mean(val_history[-val_ma_window:]))
        else:
            val_metric = val_rank_ic

        hist_entry: dict[str, float] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_rank_ic": val_rank_ic,
            "val_metric": val_metric,
        }
        if loss_type == "decile_mono":
            hist_entry["L_mono"] = train_mle
            hist_entry["L_ls"] = train_tail
            hist_entry["L_hub"] = train_hub
            hist_entry["train_l_mono"] = train_mle
            hist_entry["train_l_ls"] = train_tail
            hist_entry["train_l_hub"] = train_hub
        else:
            hist_entry["L_mle"] = train_mle
            hist_entry["L_tail"] = train_tail
            hist_entry["train_mle"] = train_mle
            hist_entry["train_tail"] = train_tail
            if hasattr(loss_fn, "last_diag") and loss_fn.last_diag:
                for k_diag, v_diag in loss_fn.last_diag.items():
                    if k_diag != "mu" and torch.is_tensor(v_diag):
                        hist_entry[f"train_{k_diag.lower()}"] = float(v_diag.item())
        history.append(hist_entry)

        if verbose:
            comp_str = (
                f"L_mono: {train_mle:.4f}, L_ls: {train_tail:.4f}, L_hub: {train_hub:.4f}"
                if loss_type == "decile_mono"
                else f"MLE: {train_mle:.4f}, Tail: {train_tail:.4f}"
            )
            print(
                f"  Epoch {epoch:2d}/{max_epochs:2d} | "
                f"Train Loss: {train_loss:.4f} ({comp_str}) | "
                f"Val Rank IC: {val_rank_ic:+.4f} (Metric: {val_metric:+.4f}) | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        if use_wandb:
            try:
                import wandb

                log_payload = {
                    f"fold_{fold.fold_idx}/train_loss": train_loss,
                    f"fold_{fold.fold_idx}/val_rank_ic": val_rank_ic,
                    f"fold_{fold.fold_idx}/val_metric": val_metric,
                    f"fold_{fold.fold_idx}/lr": float(scheduler.get_last_lr()[0]),
                    f"fold_{fold.fold_idx}/epoch": epoch,
                }
                if loss_type == "decile_mono":
                    log_payload[f"fold_{fold.fold_idx}/L_mono"] = train_mle
                    log_payload[f"fold_{fold.fold_idx}/L_ls"] = train_tail
                    log_payload[f"fold_{fold.fold_idx}/L_hub"] = train_hub
                else:
                    log_payload[f"fold_{fold.fold_idx}/L_mle"] = train_mle
                    log_payload[f"fold_{fold.fold_idx}/L_tail"] = train_tail
                    log_payload[f"fold_{fold.fold_idx}/train_mle"] = train_mle
                    log_payload[f"fold_{fold.fold_idx}/train_tail"] = train_tail
                wandb.log(log_payload)
            except Exception:  # noqa: BLE001, S110
                pass

        is_finite_val = bool(np.isfinite(val_metric))
        if is_finite_val and val_metric > fallback_val_ic:
            fallback_val_ic = val_metric
            fallback_weights = copy.deepcopy(model.state_dict())
            fallback_epoch = epoch

        can_checkpoint = epoch >= eff_min_ckpt
        if can_checkpoint and is_finite_val and val_metric > best_val_ic:
            best_val_ic = val_metric
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        elif can_checkpoint:
            patience_counter += 1
            if patience_counter >= patience and epoch >= warmup_epochs:
                if verbose:
                    print(
                        f"  Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})."
                    )
                break

    if best_weights is None and fallback_weights is not None:
        best_val_ic = fallback_val_ic
        best_weights = fallback_weights
        best_epoch = fallback_epoch

    # Restore best weights and evaluate on test partition
    if best_weights is not None:
        model.load_state_dict(best_weights)

    test_scores, test_t5, test_mask, test_permno = collect_predictions(
        model, test_loader, device
    )
    test_metrics = evaluate_predictions(
        scores=test_scores,
        target_5d=test_t5,
        mask=test_mask,
        permno_matrix=test_permno,
        hac_lags=4,
        rebalance_stride=rebalance_stride,
        horizon=horizon,
        allow_mismatched_stride=allow_mismatched_stride,
    )
    test_metrics_stride5 = (
        test_metrics
        if rebalance_stride == 5
        else evaluate_predictions(
            scores=test_scores,
            target_5d=test_t5,
            mask=test_mask,
            permno_matrix=test_permno,
            hac_lags=4,
            rebalance_stride=5,
            horizon=5,
        )
    )

    t_wall_end = time.time()
    t_mono_end = time.monotonic()
    active_compute_sec = t_mono_end - t_mono_start
    wall_clock_sec = t_wall_end - t_wall_start

    # BUG-06 Step 6.4: Persist best checkpoints to disk
    if save_checkpoint and checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        if checkpoint_filename is not None:
            save_file = ckpt_path / checkpoint_filename
        else:
            save_file = ckpt_path / f"fold_{fold.fold_idx}_best.pt"
        torch.save(
            {
                "fold_idx": fold.fold_idx,
                "best_epoch": best_epoch,
                "best_val_ic": best_val_ic,
                "model_state_dict": best_weights
                if best_weights is not None
                else model.state_dict(),
                "history": history,
            },
            save_file,
        )
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:  # noqa: BLE001
            git_sha = "unknown"

        hasher = hashlib.sha256()
        date_targets = getattr(store, "date_targets", {})
        if isinstance(date_targets, dict):
            target_arrays = (date_targets[k] for k in sorted(date_targets.keys()))
        else:
            target_arrays = date_targets
        for t_arr in target_arrays:
            hasher.update(np.ascontiguousarray(t_arr, dtype=np.float32).tobytes())
        target_hash = hasher.hexdigest()

        eff_lam_hub = (
            float((loss_kwargs or {}).get("lam_hub", 0.0))
            if loss_type == "decile_mono"
            else float(lambda_hub)
        )
        metadata = {
            "loss_type": loss_type,
            "lam_hub": eff_lam_hub,
            "device": str(device),
            "dtype": "bfloat16+float32_loss" if use_autocast else "float32",
            "lr": float(learning_rate),
            "weight_decay": float(weight_decay),
            "warmup_epochs": int(warmup_epochs),
            "min_checkpoint_epoch": eff_min_ckpt,
            "seed": int(seed),
            "batch_size": int(batch_size),
            "git_sha": git_sha,
            "target_hash": target_hash,
        }
        meta_path = ckpt_path / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f_meta:
            json.dump(metadata, f_meta, indent=2)
        stem_meta_path = save_file.with_suffix(".metadata.json")
        with open(stem_meta_path, "w", encoding="utf-8") as f_stem:
            json.dump(metadata, f_stem, indent=2)
        if verbose:
            print(f"Saved best checkpoint to {save_file}")

    if verbose:
        print(
            f"Fold {fold.fold_idx} Complete in {active_compute_sec:.1f}s (active compute, wall clock: {wall_clock_sec:.1f}s) | "
            f"Best Epoch: {best_epoch} | "
            f"Val Rank IC: {best_val_ic:+.4f} | "
            f"Test Rank IC: {test_metrics['rank_ic']:+.4f} | "
            f"Test IC_IR: {test_metrics['ic_ir']:+.3f} | "
            f"Net Sharpe: {test_metrics['ls_sharpe_net']:+.3f}"
        )

    if use_wandb:
        try:
            import wandb

            fold_log = {
                f"fold_{fold.fold_idx}/test_{k}": v for k, v in test_metrics.items()
            }
            fold_log[f"fold_{fold.fold_idx}/best_epoch"] = best_epoch
            fold_log[f"fold_{fold.fold_idx}/best_val_ic"] = best_val_ic
            fold_log[f"fold_{fold.fold_idx}/train_time_sec"] = active_compute_sec
            fold_log[f"fold_{fold.fold_idx}/active_compute_seconds"] = (
                active_compute_sec
            )
            fold_log[f"fold_{fold.fold_idx}/wall_clock_seconds"] = wall_clock_sec
            wandb.log(fold_log)
        except Exception:  # noqa: BLE001, S110
            pass

    return {
        "fold_idx": fold.fold_idx,
        "best_epoch": best_epoch,
        "best_val_ic": best_val_ic,
        "train_time_sec": active_compute_sec,
        "active_compute_seconds": active_compute_sec,
        "wall_clock_seconds": wall_clock_sec,
        "metrics": test_metrics,
        "metrics_stride5": test_metrics_stride5,
        "history": history,
    }


def run_full_walk_forward_pipeline(
    warehouse_path: str | Path = "data/warehouse/russell1000.duckdb",
    table_name: str = "silver.features_daily",
    cache_dir: str | Path | None = "data/cache",
    device: torch.device | None = None,
    batch_size: int = 8,
    grad_accum_steps: int = 8,
    n_splits: int = 5,
    max_epochs: int = 30,
    patience: int = 10,
    warmup_epochs: int = 15,
    val_ma_window: int = 3,
    min_checkpoint_epoch: int | None = None,
    checkpoint_dir: str | Path | None = "models/checkpoints",
    seed: int = 42,
    target_folds: list[int] | None = None,
    use_wandb: bool = True,
    wandb_project: str = "russell1000-v13-baseline",
    wandb_run_name: str | None = None,
) -> dict[str, Any]:
    """Execute complete 4-fold purged walk-forward cross-validation pipeline."""
    import os

    t0_wall = time.time()
    t0_mono = time.monotonic()

    wandb_url = ""
    if use_wandb:
        if "WANDB_API_KEY" not in os.environ and Path(".env").exists():
            with open(".env") as f:
                for line in f:
                    if line.startswith("WANDB_API_KEY="):
                        os.environ["WANDB_API_KEY"] = line.strip().split("=", 1)[1]
        try:
            import wandb

            run_name = wandb_run_name or f"baseline_1a_{table_name.replace('.', '_')}"
            run = wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "model": "Architecture Y (v1.3 Baseline v1)",
                    "universe": "full silver.russell (~1,000 stocks/year)",
                    "table_name": table_name,
                    "target": "target_5d (5-session forward return)",
                    "y_star": "cross-sectional rank in [-0.5, +0.5]",
                    "loss": "ListMLE + 1.5 * SoftplusTail(threshold=0.4, mult=3.0, soft_tanh=4.0)",
                    "seq_len": 60,
                    "n_splits": n_splits,
                    "horizon": 5,
                    "embargo": 5,
                    "batch_size": batch_size,
                    "grad_accum_steps": grad_accum_steps,
                    "effective_batch_size": batch_size * grad_accum_steps,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "warmup_epochs": warmup_epochs,
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "seed": seed,
                    "device": str(device or get_default_device()),
                },
            )
            wandb_url = run.url if hasattr(run, "url") else ""
        except Exception as e:  # noqa: BLE001
            print(f"Warning: W&B initialization failed: {e}")
            use_wandb = False

    store = FeatureDataStore(
        warehouse_path=warehouse_path,
        table_name=table_name,
        n_max_stocks=1200,
        seq_len=60,
        cache_dir=cache_dir,
    )

    eval_dates = store.unique_dates[store.eval_date_indices]
    splits = list(
        purged_walk_forward_cv(eval_dates, n_splits=n_splits, horizon=5, embargo=5)
    )

    fold_results = []
    for split in splits:
        if target_folds is not None and split.fold_idx not in target_folds:
            continue
        res = train_one_fold(
            fold=split,
            store=store,
            device=device,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            max_epochs=max_epochs,
            patience=patience,
            warmup_epochs=warmup_epochs,
            val_ma_window=val_ma_window,
            min_checkpoint_epoch=min_checkpoint_epoch,
            checkpoint_dir=checkpoint_dir,
            seed=seed,
            verbose=True,
            use_wandb=use_wandb,
        )
        fold_results.append(res)

    total_time_wall = time.time() - t0_wall
    total_time_active = time.monotonic() - t0_mono

    # Summary metrics across folds
    if fold_results:
        metrics_keys = list(fold_results[0]["metrics"].keys())
        summary_stats: dict[str, float] = {}
        for k in metrics_keys:
            vals = [f["metrics"][k] for f in fold_results if k in f["metrics"]]
            summary_stats[f"mean_{k}"] = float(np.mean(vals))
            summary_stats[f"std_{k}"] = float(np.std(vals))

        pos_ic_folds = sum(1 for f in fold_results if f["metrics"]["rank_ic"] > 0)
        pos_sharpe_folds = sum(
            1 for f in fold_results if f["metrics"]["ls_sharpe_net"] > 0
        )
        acceptance_passed = (
            len(fold_results) == 4 and pos_ic_folds >= 3 and pos_sharpe_folds >= 2
        )

        summary_stats["pos_rank_ic_folds"] = float(pos_ic_folds)
        summary_stats["pos_net_sharpe_folds"] = float(pos_sharpe_folds)
        summary_stats["acceptance_gate_passed"] = 1.0 if acceptance_passed else 0.0

        if use_wandb:
            try:
                import wandb

                for sk, sv in summary_stats.items():
                    wandb.summary[sk] = sv

                table_cols = [
                    "Fold",
                    "Best Epoch",
                    "Best Val Rank IC",
                    "Test Rank IC",
                    "IC IR",
                    "Top Decile Prec",
                    "Bottom Decile Prec",
                    "LS Spread Gross",
                    "LS Spread Net",
                    "LS Sharpe Gross",
                    "LS Sharpe Net",
                    "Turnover",
                    "Max Drawdown",
                    "Active Return",
                ]
                table_data = []
                for f_res in fold_results:
                    f_idx = f_res["fold_idx"]
                    m = f_res["metrics"]
                    table_data.append(
                        [
                            f_idx,
                            f_res["best_epoch"],
                            round(f_res["best_val_ic"], 4),
                            round(m["rank_ic"], 4),
                            round(m["ic_ir"], 3),
                            round(m["top_decile_precision"], 4),
                            round(m["bottom_decile_precision"], 4),
                            round(m["ls_spread_gross"], 4),
                            round(m["ls_spread_net"], 4),
                            round(m["ls_sharpe_gross"], 3),
                            round(m["ls_sharpe_net"], 3),
                            round(m["turnover"], 4),
                            round(m["max_drawdown"], 4),
                            round(m["active_return"], 4),
                        ]
                    )
                wandb.log(
                    {"results_table": wandb.Table(columns=table_cols, data=table_data)}
                )
                wandb.finish()
            except Exception as e:  # noqa: BLE001
                print(f"Warning: W&B summary logging failed: {e}")

        # Automatically write acceptance report
        generate_baseline_1a_report(
            fold_results=fold_results,
            output_path="docs/baseline_1a.md",
            wandb_url=wandb_url,
            total_time_sec=total_time_active,
        )

    return {
        "table_name": table_name,
        "n_folds_completed": len(fold_results),
        "total_time_sec": total_time_active,
        "active_compute_seconds": total_time_active,
        "wall_clock_seconds": total_time_wall,
        "wandb_url": wandb_url,
        "folds": fold_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run v1.3 Walk-Forward Model Pipeline on Russell 1000"
    )
    parser.add_argument(
        "--table", default="silver.features_daily", help="Target table name"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=30, help="Max training epochs per fold"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10 per v1.3 §9.7)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Warmup epochs before early stopping (default: 5)",
    )
    parser.add_argument(
        "--fold", type=int, default=None, help="Specific fold to execute (default: all)"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Micro-batch size in dates"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8 -> effective batch size 64)",
    )
    args = parser.parse_args()

    target = [args.fold] if args.fold is not None else None
    print(
        f"Starting Walk-Forward Cross-Validation Pipeline (Folds: {'all' if target is None else target})..."
    )
    res = run_full_walk_forward_pipeline(
        table_name=args.table,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_epochs=args.max_epochs,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        target_folds=target,
        use_wandb=not args.no_wandb,
    )
    print("Pipeline Execution Completed!")
    if res.get("wandb_url"):
        print(f"W&B Run URL: {res['wandb_url']}")
