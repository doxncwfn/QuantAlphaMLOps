import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum.attr import IntegratedGradients
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import CONFIG
from src.data.residualizer import UniversalDataProcessor, UniversalDataset
from src.training.trainer import AlphaEngineTrainer

root_dir = Path().resolve()
parquet_path = root_dir / "artifacts" / "oos_predictions_3.parquet"

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")


class StockScorer(torch.nn.Module):
    """
    This wrapper isolates the Alpha Engine's prediction for a specific stock index (expects scalar output for attribution).
    """

    def __init__(self, model, stock_idx):
        super().__init__()
        self.model = model
        self.stock_idx = stock_idx

    def forward(self, x):
        scores = self.model(x, stock_ids=None)
        return scores[:, self.stock_idx]


def run_xai_diagnostics(oos_parquet_path=parquet_path):
    print("TASK 1.4: REGIME-CONDITIONED XAI DIAGNOSTICS")

    oos_df = pd.read_parquet(oos_parquet_path)
    oos_df["Date"] = pd.to_datetime(oos_df["Date"])

    print("\n[1/4] Reconstructing Feature Tensors...")
    processor = UniversalDataProcessor(
        data_dir=CONFIG["data_dir"], ff_dir=CONFIG["fama_french_dir"], tickers=CONFIG["tickers"]
    )
    big_df = processor.load_and_process()
    feature_cols = processor.feature_cols
    num_stocks = len(processor.valid_tickers)
    seq_len = CONFIG["seq_len"]

    print("[2/4] Segmenting Market Volatility Regimes...")
    market_vol = big_df.groupby("Date")["Vol_60"].median().reset_index()
    market_vol["Trailing_60D_Med_Vol"] = market_vol["Vol_60"].rolling(60).median()

    oos_df = pd.merge(oos_df, market_vol[["Date", "Trailing_60D_Med_Vol"]], on="Date", how="left")
    global_median_vol = oos_df["Trailing_60D_Med_Vol"].median()
    oos_df["Regime"] = np.where(
        oos_df["Trailing_60D_Med_Vol"] > global_median_vol, "High_Vol", "Low_Vol"
    )

    high_vol_dates = oos_df[oos_df["Regime"] == "High_Vol"]["Date"].unique()
    low_vol_dates = oos_df[oos_df["Regime"] == "Low_Vol"]["Date"].unique()

    print(f"      High Volatility Trading Days: {len(high_vol_dates)}")
    print(f"      Low Volatility Trading Days : {len(low_vol_dates)}")

    artifacts_dir = Path(oos_parquet_path).resolve().parent
    ckpt_dir = artifacts_dir / "checkpoints"
    ckpt_path = ckpt_dir / "fold_5_best-v2.ckpt"

    print(f"Artifacts directory: {artifacts_dir}")
    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"Looking for checkpoint: {ckpt_path}")

    if not ckpt_path.exists():
        available = sorted(ckpt_dir.glob("fold_5_best*.ckpt")) if ckpt_dir.exists() else []
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Available checkpoints: {[p.name for p in available]}"
        )

    print(f"Loading checkpoint: {ckpt_path}")
    pl_module = AlphaEngineTrainer.load_from_checkpoint(str(ckpt_path))
    model = pl_module.model

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("\n[3/4] Executing PyTorch Integrated Gradients (Sampling 5000 stock-days)...")

    torch.backends.cudnn.enabled = False

    dataset = UniversalDataset(big_df, feature_cols, seq_len, num_stocks)

    sample_high_dates = np.random.choice(
        high_vol_dates, size=min(50, len(high_vol_dates)), replace=False
    )
    sample_low_dates = np.random.choice(
        low_vol_dates, size=min(50, len(low_vol_dates)), replace=False
    )

    attributions = {"High_Vol": np.zeros(len(feature_cols)), "Low_Vol": np.zeros(len(feature_cols))}

    def compute_regime_attributions(sample_dates, regime_key):
        regime_attr = np.zeros(len(feature_cols))
        total_samples = 0

        for d in sample_dates:
            try:
                idx = dataset.valid_dates.index(d)
                x_tensor, _, _ = dataset[idx]
                x_tensor = x_tensor.unsqueeze(0).to(device)
                x_baseline = torch.zeros_like(x_tensor).to(device)

                sampled_stocks = np.random.choice(num_stocks, size=50, replace=False)

                for s_idx in sampled_stocks:
                    scorer = StockScorer(model, s_idx).to(device)
                    ig = IntegratedGradients(scorer)
                    attr, _ = ig.attribute(
                        inputs=x_tensor,
                        baselines=x_baseline,
                        internal_batch_size=10,
                        return_convergence_delta=True,
                    )
                    stock_attr = attr[0, s_idx, :, :].mean(dim=0).cpu().detach().numpy()
                    regime_attr += np.abs(stock_attr)
                    total_samples += 1

                    if total_samples % 250 == 0:
                        print(
                            f"      [{regime_key}] Processed {total_samples} / 2500 stock-days..."
                        )

                    del attr, scorer, ig
                    torch.cuda.empty_cache()

            except ValueError:
                continue

        return (regime_attr / total_samples) if total_samples > 0 else regime_attr

    attributions["High_Vol"] = compute_regime_attributions(sample_high_dates, "High_Vol")
    attributions["Low_Vol"] = compute_regime_attributions(sample_low_dates, "Low_Vol")

    high_attr_pct = (attributions["High_Vol"] / attributions["High_Vol"].sum()) * 100
    low_attr_pct = (attributions["Low_Vol"] / attributions["Low_Vol"].sum()) * 100

    print("\n[4/4] Training LightGBM Surrogate to test Black-Box Fidelity...")
    surrogate_data = oos_df.merge(big_df, on=["Date", "Ticker"], how="inner")

    X_surrogate = surrogate_data[feature_cols].values
    y_target = surrogate_data["Pred_Alpha"].values

    lgbm = LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    lgbm.fit(X_surrogate, y_target)
    surrogate_preds = lgbm.predict(X_surrogate)
    fidelity_score, _ = spearmanr(surrogate_preds, y_target)

    print("XAI DIAGNOSTIC RESULTS")
    print(f"LightGBM Surrogate Fidelity (Spearman Rank) : {fidelity_score:.4f}")
    if fidelity_score > 0.75:
        print("-> SUCCESS: Deep Learning logic is highly interpretable and structurally sound.")
    else:
        print(
            "-> WARNING: Neural network logic cannot be mapped. High risk of unexplainable black-box behavior."
        )

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(feature_cols))
    width = 0.35
    ax.bar(
        x - width / 2,
        high_attr_pct,
        width,
        label="High Volatility Regime",
        color="#d62728",
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        low_attr_pct,
        width,
        label="Low Volatility Regime",
        color="#1f77b4",
        edgecolor="black",
    )
    ax.set_ylabel("Relative Feature Importance (%)")
    ax.set_title(
        f"Integrated Gradients: Feature Attribution Shift Across Regimes\n(Surrogate Fidelity: {fidelity_score:.4f})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/mnt/data/xai_regime_diagnostics.png")
    print("\nVisual diagnostic saved to /mnt/data/xai_regime_diagnostics.png")


if __name__ == "__main__":
    run_xai_diagnostics()
