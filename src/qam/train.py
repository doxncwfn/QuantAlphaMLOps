"""Train stage: walk-forward folds, one generic Lightning module, out-of-sample predictions.

The model and loss come from the registry; this file only wires them together.
Output: <run_dir>/predictions.parquet with (date, security_id, slot, score, fold) for every
real slot of every test date, and one best checkpoint per fold under <run_dir>/checkpoints/.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from qam import registry
from qam.config import RunConfig
from qam.data import DatasetStore, DatasetView, WindowDataset
from qam.tracking import RunContext

log = logging.getLogger("qam.train")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def spearman_per_date(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, min_n: int = 2) -> list[float]:
    out = []
    for b in range(scores.shape[0]):
        valid = mask[b]
        if int(valid.sum()) < min_n:
            continue
        s = torch.argsort(torch.argsort(scores[b][valid])).double()
        y = torch.argsort(torch.argsort(labels[b][valid])).double()
        s, y = s - s.mean(), y - y.mean()
        denom = s.norm() * y.norm()
        if denom > 0:
            out.append(float((s * y).sum() / denom))
    return out


class RankingModule(pl.LightningModule):
    def __init__(self, cfg: RunConfig, model: torch.nn.Module, loss):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss
        self._val_ic: list[float] = []

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        loss = self.loss_fn(self(batch), batch["label"], batch["label_mask"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["x"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        scores = self(batch)
        loss = self.loss_fn(scores, batch["label"], batch["label_mask"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["x"].shape[0])
        self._val_ic += spearman_per_date(scores.detach(), batch["label"], batch["label_mask"])

    def on_validation_epoch_end(self):
        self.log("val_ic", float(np.mean(self._val_ic)) if self._val_ic else 0.0, prog_bar=True)
        self._val_ic = []

    def predict_step(self, batch, batch_idx):
        return {"scores": self(batch), "mask": batch["mask"], "ids": batch["ids"], "date_index": batch["date_index"]}

    def configure_optimizers(self):
        t = self.cfg.train_spec
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
        mode = "max" if t.monitor == "val_ic" else "min"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=t.lr_factor, patience=t.lr_patience)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": t.monitor}}


def build_model(cfg: RunConfig, view: DatasetView) -> torch.nn.Module:
    dims = dict(
        n_features=view.features.shape[2],
        n_mkt_ts=view.mkt_ts.shape[1],
        n_mkt_static=view.mkt_static.shape[1],
        lookback=cfg.window_spec.lookback,
        k=view.k,
    )
    return registry.build("model", cfg.model_spec, **dims)


def _labelled(view: DatasetView, indices: np.ndarray, min_valid: int) -> np.ndarray:
    """Keep decision dates with at least min_valid real slots that have a label."""
    keep = []
    for i in indices:
        cols = view.slots[i]
        real = cols >= 0
        if (~np.isnan(view.labels[i, cols[real]])).sum() >= min_valid:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def train(cfg: RunConfig, run: RunContext, tables: dict[str, str]) -> pd.DataFrame:
    """tables: table -> version hash to train on (recorded in meta.json by pipeline.run)."""
    store = DatasetStore.for_config(cfg)
    missing = [t for t, h in tables.items() if not store.has(t, h)]
    if missing:
        raise RuntimeError(f"dataset tables {missing} are not built for this config; run `qam build-dataset` first")

    seed_everything(cfg.seed, cfg.train_spec.deterministic)
    view = DatasetView.load(store, cfg, tables)
    decision = view.decision_indices()
    folds = registry.build("splitter", cfg.split_spec).split(decision)
    log.info("%d decision dates, k=%d, %d features, %d folds", len(decision), view.k, view.features.shape[2], len(folds))

    h = cfg.label_spec.kwargs.get("h")
    purge = cfg.split_spec.kwargs.get("purge")
    if h is not None and purge is not None and purge < h:
        log.warning("split purge (%s) is shorter than the label horizon h (%s): labels overlap across segments", purge, h)

    t = cfg.train_spec
    loader = lambda idx: DataLoader(WindowDataset(view, idx), batch_size=t.batch_dates, shuffle=False, num_workers=t.num_workers)
    rows = []
    for fold in folds:
        train_idx = _labelled(view, fold.train, t.min_valid_slots)
        val_idx = _labelled(view, fold.val, t.min_valid_slots)
        log.info("%s: train %d, val %d, test %d dates", fold.name, len(train_idx), len(val_idx), len(fold.test))
        seed_everything(cfg.seed, t.deterministic)
        module = RankingModule(cfg, build_model(cfg, view), registry.build("loss", cfg.loss_spec))
        mode = "max" if t.monitor == "val_ic" else "min"
        checkpoint = ModelCheckpoint(dirpath=run.run_dir / "checkpoints" / fold.name, monitor=t.monitor, mode=mode, save_top_k=1)
        callbacks = [checkpoint]
        if t.early_stopping_patience:
            callbacks.append(EarlyStopping(monitor=t.monitor, mode=mode, patience=t.early_stopping_patience))
        trainer = pl.Trainer(
            max_epochs=t.max_epochs,
            accelerator=t.accelerator,
            deterministic=t.deterministic,
            logger=run.lightning_loggers(fold.name),
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
        )
        trainer.fit(module, loader(train_idx), loader(val_idx))
        best = checkpoint.best_model_score
        if best is not None:
            run.log_metrics({f"best_{t.monitor}": float(best)}, prefix=fold.name)

        outputs = trainer.predict(module, loader(fold.test), ckpt_path="best")
        for out in outputs:
            scores, mask, ids, dates = (out[key].cpu().numpy() for key in ("scores", "mask", "ids", "date_index"))
            for b in range(scores.shape[0]):
                real = np.flatnonzero(mask[b])
                rows.append(
                    pd.DataFrame(
                        {
                            "date": view.calendar[dates[b]],
                            "security_id": ids[b, real],
                            "slot": real,
                            "score": scores[b, real],
                            "fold": fold.name,
                        }
                    )
                )
    predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "security_id", "slot", "score", "fold"])
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions.to_parquet(run.run_dir / "predictions.parquet", index=False)
    log.info("wrote %d predictions", len(predictions))
    return predictions
