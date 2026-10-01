import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import warnings
from pathlib import Path

# Ensure src/ is on sys.path for local module resolution
src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import optuna
import pandas as pd
import pytorch_lightning as pl_trainer
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

import yaml

from src.data.residualizer import UniversalDataProcessor, UniversalDataset
from src.data.splitters import PurgedWalkForwardSplitter
from src.training.trainer import AlphaEngineTrainer, seed_everything


def load_config(config_path: Path | str | None = None) -> dict:
    """Load configuration from config/config.yaml."""
    if config_path is None:
        candidates = [
            Path(__file__).resolve().parents[1] / "config" / "config.yaml",
            Path("config/config.yaml"),
            Path("config.yaml"),
        ]
        for c in candidates:
            if c.exists():
                config_path = c
                break
    if config_path is None or not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if cfg.get("device") in (None, "auto"):
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.setdefault("tickers", [])
    return cfg


CONFIG = load_config()


def create_dataloaders(train_df, val_df, feature_cols, num_stocks, batch_size=32):
    train_dataset = UniversalDataset(train_df, feature_cols, CONFIG["seq_len"], num_stocks)
    val_dataset = UniversalDataset(val_df, feature_cols, CONFIG["seq_len"], num_stocks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader


def objective(trial, inner_train, inner_val, feature_cols, num_stocks, input_dim):
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    margin = trial.suggest_categorical("margin", [0.1, 0.25])

    train_loader, val_loader = create_dataloaders(
        inner_train, inner_val, feature_cols, num_stocks, CONFIG["batch_size"]
    )

    model = AlphaEngineTrainer(
        num_stocks=num_stocks,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        margin=margin,
        decile=0.10,
    )

    early_stop = EarlyStopping(monitor="val_ic", mode="max", patience=3)
    trainer = pl_trainer.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        callbacks=[early_stop],
        enable_progress_bar=False,
        logger=False,
        precision="16-mixed",
    )
    trainer.fit(model, train_loader, val_loader)
    return early_stop.best_score.item() if early_stop.best_score is not None else 0.0


def main(config: dict | None = None):
    if config is not None:
        CONFIG.update(config)

    seed_everything(42)
    torch.set_float32_matmul_precision("high")

    ckpt_dir = Path(CONFIG.get("ckpt_dir", "/mnt/data/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("--- 1. Initializing Data Gateway ---")
    processor = UniversalDataProcessor(
        data_dir=CONFIG["data_dir"], ff_dir=CONFIG["fama_french_dir"], tickers=CONFIG["tickers"]
    )
    big_df = processor.load_and_process()
    feature_cols = processor.feature_cols
    input_dim = len(feature_cols)
    num_stocks = len(processor.valid_tickers)

    splitter = PurgedWalkForwardSplitter(n_folds=5, purge_days=5, embargo_days=20)
    folds = splitter.split(big_df)

    # 2. Inner Split for Optuna (Fold 1)
    print("\n--- 2. Executing Optuna Capacity Ablation ---")
    fold1_train = folds[0][0]
    dates = sorted(fold1_train["Date"].unique())
    split_idx = int(len(dates) * 0.70)
    seq_len = CONFIG["seq_len"]

    inner_train_dates = dates[: split_idx - 5]
    inner_val_dates = dates[split_idx - seq_len + 1 :]

    inner_train = fold1_train[fold1_train["Date"].isin(inner_train_dates)]
    inner_val = fold1_train[fold1_train["Date"].isin(inner_val_dates)]

    study = optuna.create_study(direction="maximize", study_name="Phase1_Ablation")
    study.optimize(
        lambda trial: objective(trial, inner_train, inner_val, feature_cols, num_stocks, input_dim),
        n_trials=5,
    )
    best_params = study.best_params
    print(f"\nOptimal Hyperparameters: {best_params}")

    # 3. 5-Fold Walk-Forward Evaluation & Artifact Export
    print("\n--- 3. Executing 5-Fold Out-of-Sample Walk-Forward ---")
    wandb.init(project="HCMUT-Specialized-Project", name="OOS_Walk_Forward", config=best_params)

    oos_records = []

    for i, (train_df, test_df) in enumerate(folds):
        print(f"\n>>> Training Fold {i + 1}/5...")
        train_loader, test_loader = create_dataloaders(
            train_df, test_df, feature_cols, num_stocks, CONFIG["batch_size"]
        )

        model = AlphaEngineTrainer(
            num_stocks=num_stocks, input_dim=input_dim, decile=0.10, **best_params
        )

        fold_ckpt_path = ckpt_dir / f"fold_{i + 1}_best"
        checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"fold_{i + 1}_best",
            monitor="val_ic",
            mode="max",
            save_top_k=1,
        )

        trainer = pl_trainer.Trainer(
            max_epochs=15,
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint_cb],
            logger=WandbLogger(project="HCMUT-Specialized-Project"),
            precision="16-mixed",
        )

        trainer.fit(model, train_loader, test_loader)

        # Load best weights and extract predictions
        best_model_path = checkpoint_cb.best_model_path or f"{fold_ckpt_path}.ckpt"
        if os.path.exists(best_model_path):
            model = AlphaEngineTrainer.load_from_checkpoint(
                best_model_path,
                num_stocks=num_stocks,
                input_dim=input_dim,
                decile=0.10,
                **best_params,
            )

        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Extract out-of-sample predictions
        val_dataset = test_loader.dataset
        dates_in_test = val_dataset.valid_dates
        id_to_ticker = {v: k for k, v in processor.stock_to_id.items()}

        with torch.no_grad():
            for idx, d in enumerate(dates_in_test):
                x, s_ids, target = val_dataset[idx]
                x = x.unsqueeze(0).to(model.device)
                s_ids = s_ids.to(model.device)
                pred = model(x, s_ids).squeeze(0).cpu().numpy()

                day_df = test_df[test_df["Date"] == d]
                for s_idx, ticker_str in id_to_ticker.items():
                    sub = day_df[day_df["Stock_ID"] == s_idx]
                    if not sub.empty:
                        oos_records.append(
                            {
                                "Fold": i + 1,
                                "Date": d,
                                "Ticker": ticker_str,
                                "Pred_Alpha": float(pred[s_idx]),
                                "Target_5D_Z": float(target[s_idx].item()),
                                "Log_Ret": float(sub["Log_Ret"].values[0]),
                            }
                        )

    oos_df = pd.DataFrame(oos_records)
    out_path = Path("/mnt/data/oos_predictions.parquet")
    oos_df.to_parquet(out_path, index=False)
    print(f"\n[SUCCESS] OOS Predictions saved to {out_path} ({len(oos_df)} rows).")
    wandb.finish()


if __name__ == "__main__":
    main()
