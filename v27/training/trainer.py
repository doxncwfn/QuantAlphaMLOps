"""Walk-forward trainer, checkpointing, and OOS prediction/gate inference using Polars Parquet I/O."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import nn

from v27.config import EXPERIMENT_SPECS, DataConfig, ModelConfig, TrainConfig
from v27.models.hybrid_lstm_tfm import HybridLstmTfm
from v27.models.losses import compute_spearman_loss
from v27.targets.batch_sampler import iter_batches
from v27.targets.target_builder import PreparedAlphaDataset
from v27.training.folds import build_walk_forward_folds
from v27.training.metrics import daily_rank_ic, daily_tail_ic


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def cross_sectional_zscore(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero-mean unit-variance cross-sectional score normalization per date."""
    mf = mask.float()
    n = mf.sum(1, keepdim=True).clamp(min=1.0)
    mu = (scores * mf).sum(1, keepdim=True) / n
    sd = torch.sqrt((((scores - mu) * mf) ** 2).sum(1, keepdim=True) / n + 1e-8)
    return (scores - mu) / sd * mf


def make_day_weights(
    train_idx: np.ndarray,
    total_days: int,
    halflife: int | None,
    device: torch.device,
) -> torch.Tensor:
    """Exponential recency weights favoring more recent training dates."""
    w = torch.ones(total_days, device=device)
    if halflife:
        age = torch.as_tensor(
            train_idx[-1] - train_idx, dtype=torch.float32, device=device
        )
        w[torch.as_tensor(train_idx, dtype=torch.long, device=device)] = 0.5 ** (
            age / halflife
        )
    return w


@torch.no_grad()
def evaluate_fold_metrics(
    model: HybridLstmTfm,
    ds: PreparedAlphaDataset,
    indices: np.ndarray,
    exp_cfg: dict[str, Any],
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    use_amp: bool,
) -> tuple[float, float, float]:
    """Compute validation mean RankIC, TailIC, and composite early-stopping score."""
    model.eval()
    ics, tics = [], []
    for b in iter_batches(
        ds,
        indices,
        target_key=str(exp_cfg["target"]),
        eval_key=str(exp_cfg["eval_target"]),
        batch_size=train_cfg.batch_dates,
        lookback=data_cfg.lookback,
        max_missing=data_cfg.max_missing,
    ):
        with torch.amp.autocast("cuda", enabled=use_amp):
            sc = model(b["x"], b["mask"], b["mkt"])
        scf = sc.float()
        ics.append(
            daily_rank_ic(
                scf, b["y_eval"], b["y_eval_mask"], min_stocks=train_cfg.min_stocks
            )
        )
        tics.append(
            daily_tail_ic(
                scf,
                b["y_eval"],
                b["y_eval_mask"],
                q=train_cfg.tail_q,
                min_stocks=train_cfg.min_stocks,
            )
        )
    ic_cat = torch.cat(ics) if ics else torch.empty(0)
    tic_cat = torch.cat(tics) if tics else torch.empty(0)
    ic_m = float(ic_cat.mean()) if len(ic_cat) else -1.0
    tic_m = float(tic_cat.mean()) if len(tic_cat) else -1.0
    return ic_m, tic_m, ic_m + train_cfg.tail_ic_lambda * tic_m


def train_one_fold(
    ds: PreparedAlphaDataset,
    fold: dict[str, Any],
    exp_cfg: dict[str, Any],
    seed: int,
    day_weights: torch.Tensor,
    data_cfg: DataConfig | None = None,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> tuple[HybridLstmTfm, float, float, float, int, list[dict[str, float]]]:
    """Train HybridLstmTfm on one walk-forward fold with early stopping on Val RankIC + TailIC."""
    data_cfg = data_cfg or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    seed_all(seed)
    dev = ds.Xn_g.device
    use_amp = dev.type == "cuda"
    model = HybridLstmTfm(ds.n_feat, ds.m_dim, cfg=model_cfg).to(dev)
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score, best_ic, best_tic = -np.inf, -np.inf, -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_ep, wait, hist = 0, 0, []

    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        tot = torch.zeros((), device=dev)
        nb = 0
        for b in iter_batches(
            ds,
            fold["train"],
            target_key=str(exp_cfg["target"]),
            eval_key=str(exp_cfg["eval_target"]),
            batch_size=train_cfg.batch_dates,
            lookback=data_cfg.lookback,
            max_missing=data_cfg.max_missing,
            shuffle=True,
            sub_k=train_cfg.train_subsample,
        ):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                scores = model(b["x"], b["mask"], b["mkt"])
            loss = compute_spearman_loss(
                scores.float(),
                b["y_train"],
                b["y_train_mask"],
                day_weights[b["date_index"]],
                min_stocks=train_cfg.min_stocks,
                tau=train_cfg.soft_tau,
            )
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.detach()
            nb += 1

        val_ic, val_tic, val_score = evaluate_fold_metrics(
            model, ds, fold["val"], exp_cfg, data_cfg, train_cfg, use_amp
        )
        hist.append(
            {
                "epoch": float(epoch),
                "train_loss": float(tot) / max(nb, 1),
                "val_ic": val_ic,
                "val_tail_ic": val_tic,
                "val_score": val_score,
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )
        sched.step(val_score)
        if val_score > best_score:
            best_score, best_ic, best_tic, best_ep, wait = (
                val_score,
                val_ic,
                val_tic,
                epoch,
                0,
            )
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            wait += 1
            if wait >= train_cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ic, best_tic, best_score, best_ep, hist


@torch.no_grad()
def predict_with_alpha(
    model: HybridLstmTfm,
    ds: PreparedAlphaDataset,
    indices: np.ndarray,
    exp_cfg: dict[str, Any],
    data_cfg: DataConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> tuple[torch.Tensor, ...]:
    """Generate raw cross-sectional scores and MASTER gate weights across target indices."""
    data_cfg = data_cfg or DataConfig()
    train_cfg = train_cfg or TrainConfig()
    model.eval()
    use_amp = ds.Xn_g.device.type == "cuda"
    S, M_, Y1, YM1, YE, YME, T, A = [], [], [], [], [], [], [], []
    for b in iter_batches(
        ds,
        indices,
        target_key=str(exp_cfg["target"]),
        eval_key=str(exp_cfg["eval_target"]),
        batch_size=train_cfg.batch_dates,
        lookback=data_cfg.lookback,
        max_missing=data_cfg.max_missing,
    ):
        with torch.amp.autocast("cuda", enabled=use_amp):
            sc, al = model(b["x"], b["mask"], b["mkt"], return_alpha=True)
        S.append(sc.float().cpu())
        M_.append(b["mask"].cpu())
        Y1.append(b["y_1d"].cpu())
        YM1.append(b["y_1d_mask"].cpu())
        YE.append(b["y_eval"].cpu())
        YME.append(b["y_eval_mask"].cpu())
        T.append(b["date_index"].cpu())
        A.append(al.float().cpu())
    return (
        torch.cat(S),
        torch.cat(M_),
        torch.cat(Y1),
        torch.cat(YM1),
        torch.cat(YE),
        torch.cat(YME),
        torch.cat(T).numpy(),
        torch.cat(A).numpy(),
    )


def run_experiment(
    name: str,
    ds: PreparedAlphaDataset,
    out_dir: Path,
    data_cfg: DataConfig | None = None,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    n_seeds: int = 1,
    resume: bool = True,
) -> tuple[pl.DataFrame, pd.DataFrame | None, list[dict[str, Any]]]:
    """Run all 14 walk-forward folds for one target config (raw_t1, risk_t1, or risk_t5)."""
    data_cfg = data_cfg or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    out_dir.mkdir(parents=True, exist_ok=True)
    exp_cfg = EXPERIMENT_SPECS[name]
    folds = build_walk_forward_folds(
        ds.decision_indices,
        ds.calendar,
        test_start=data_cfg.test_start,
        test_end=data_cfg.test_end,
        n_folds=data_cfg.n_folds,
        purge=int(exp_cfg["purge"]),
        embargo=train_cfg.embargo,
        val_days=train_cfg.val_days,
    )
    dev = ds.Xn_g.device
    frames, alpha_frames = [], []
    for f in folds:
        path = out_dir / f"preds_{name}_{f['name']}.parquet"
        apath = out_dir / f"alpha_{name}_{f['name']}.parquet"
        if resume and path.exists():
            frames.append(pl.read_parquet(path))
            if apath.exists():
                alpha_frames.append(pl.read_parquet(apath))
            continue
        t_f = time.time()
        day_w = make_day_weights(f["train"], ds.D, train_cfg.time_decay_halflife, dev)
        acc, alpha_acc, hist0 = None, None, None
        m = y1 = ym1 = t_idx = None
        for s in range(n_seeds):
            model, _, _, _, _, hist = train_one_fold(
                ds,
                f,
                exp_cfg,
                train_cfg.seed + s,
                day_w,
                data_cfg,
                model_cfg,
                train_cfg,
            )
            if s == 0:
                hist0 = hist
            sc, m, y1, ym1, _ye, _yme, t_idx, al = predict_with_alpha(
                model, ds, f["test"], exp_cfg, data_cfg, train_cfg
            )
            sc_z = cross_sectional_zscore(sc, m)
            acc = sc_z if acc is None else acc + sc_z
            alpha_acc = al if alpha_acc is None else alpha_acc + al
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        assert acc is not None and alpha_acc is not None
        assert (
            m is not None and y1 is not None and ym1 is not None and t_idx is not None
        )
        acc /= n_seeds
        alpha_acc /= n_seeds
        mm, yy, yym = m.numpy(), y1.numpy(), ym1.numpy()
        rows, sl = np.nonzero(mm)
        t_rows = t_idx[rows]
        df_pl = pl.DataFrame(
            {
                "date": ds.calendar[t_rows],
                "security_id": ds.active_ids[ds.slots[t_rows, sl]],
                "score": acc.numpy()[rows, sl],
                "label": np.where(yym[rows, sl], yy[rows, sl], np.nan),
                "fold": [str(f["name"])] * len(rows),
            }
        )
        df_pl.write_parquet(path)
        adf_pl = (
            pl.DataFrame(alpha_acc, schema=list(ds.features))
            .with_columns(pl.Series("date", ds.calendar[t_idx]))
            .select(["date", *list(ds.features)])
        )
        adf_pl.write_parquet(apath)
        if hist0 is not None:
            with open(out_dir / f"hist_{name}_{f['name']}.json", "w") as fp:
                json.dump(hist0, fp)
        frames.append(df_pl)
        alpha_frames.append(adf_pl)
        print(f"  [{name}] {f['name']} saved in {time.time() - t_f:.1f}s")

    preds_pl = pl.concat(frames, how="vertical")
    preds_pl.write_parquet(out_dir / f"predictions_{name}.parquet")
    alpha_df = None
    if alpha_frames:
        alpha_pl = (
            pl.concat(alpha_frames, how="vertical").unique(subset=["date"]).sort("date")
        )
        alpha_pl.write_parquet(out_dir / f"gate_alpha_{name}.parquet")
        alpha_df = alpha_pl.to_pandas().set_index("date")
    return preds_pl, alpha_df, folds
