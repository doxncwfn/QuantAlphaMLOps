import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import spearmanr

from models.hybrid import HybridAlphaRanker
from models.losses import HybridAlphaLoss


def seed_everything(seed: int = 42):
    """
    Enforces strict deterministic algorithmic execution where supported by the CUDA backend,
    as mathematically required by Phase 1 Risk 2.2 mitigations.
    """
    import os
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AlphaEngineTrainer(pl.LightningModule):
    def __init__(
        self,
        num_stocks: int = 482,
        input_dim: int = 15,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        margin: float = 0.1,
        decile: float = 0.10,
        lambda_mse: float = 1.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = HybridAlphaRanker(
            num_stocks=num_stocks, input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout
        )

        self.criterion = HybridAlphaLoss(margin=margin, decile=decile, lambda_mse=lambda_mse)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x, stock_ids):
        return self.model(x, stock_ids)

    def training_step(self, batch, batch_idx):
        x, stock_ids, targets = batch
        preds = self(x, stock_ids)
        loss = self.criterion(preds, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, stock_ids, targets = batch

        preds = self(x, stock_ids)
        loss = self.criterion(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Calculate the Information Coefficient (Spearman Rank Correlation)
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        batch_ic = []
        for b in range(preds_np.shape[0]):
            if np.isnan(preds_np[b]).any() or np.std(preds_np[b]) < 1e-6:
                batch_ic.append(0.0)
            else:
                corr, _ = spearmanr(preds_np[b], targets_np[b])
                batch_ic.append(0.0 if np.isnan(corr) else corr)

        mean_ic = np.mean(batch_ic) if batch_ic else 0.0
        self.log("val_ic", float(mean_ic), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


if __name__ == "__main__":
    seed_everything(42)
    print("Deterministic flags and seeds set successfully.")

    trainer_module = AlphaEngineTrainer()
    print("PyTorch Lightning AlphaEngineTrainer initialized successfully.")
    print(f"Tracking IC and Hybrid AlphaLoss for {trainer_module.hparams.num_stocks} equities.")
