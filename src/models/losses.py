import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridAlphaLoss(nn.Module):
    """
    Implements the Hybrid AlphaLoss: L_{total,t} = L_{rank,t} + lambda * L_{MSE,t}.
    Combines pointwise Mean Squared Error with a pairwise hinge loss that
    penalizes rank inversions between the top and bottom deciles.
    """

    def __init__(self, margin: float = 0.1, decile: float = 0.10, lambda_mse: float = 1.0):
        super().__init__()
        # Margin alpha must be in {0.1, 0.25} per Phase 1 requirements
        self.margin = margin
        self.decile = decile
        self.lambda_mse = lambda_mse

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        predictions: [Batch_Dates, Stocks] - Predicted alpha scores
        targets: [Batch_Dates, Stocks] - True cross-sectionally z-scored residuals
        """
        batch_size, num_stocks = predictions.shape

        # Determine the number of stocks in the top/bottom decile
        k = max(1, int(num_stocks * self.decile))

        total_loss = 0.0

        # We loop over the batch (time) dimension to strictly enforce the constraint:
        # "Ensure deciles are never pooled across dates during loss computation"
        for b in range(batch_size):
            pred_t = predictions[b]
            target_t = targets[b]

            # 1. Pointwise Component: L_{MSE,t}
            mse_loss_t = F.mse_loss(pred_t, target_t)

            # 2. Pairwise Component: L_{rank,t}
            # Sort targets to identify true top (Pt) and bottom (Qt) deciles for this specific date
            sorted_indices = torch.argsort(target_t)

            true_shorts_idx = sorted_indices[:k]
            true_longs_idx = sorted_indices[-k:]

            pred_shorts = pred_t[true_shorts_idx]
            pred_longs = pred_t[true_longs_idx]

            # Vectorized pairwise differences: pred_longs[i] - pred_shorts[j]
            diff = pred_longs.unsqueeze(1) - pred_shorts.unsqueeze(0)

            # Hinge penalty: max(0, margin - difference)
            rank_loss_t = torch.clamp(self.margin - diff, min=0.0).mean()

            # 3. Hybrid Combination: L_{total,t} = L_{rank,t} + lambda * L_{MSE,t}
            total_loss += rank_loss_t + self.lambda_mse * mse_loss_t

        # Average the hybrid loss across the time dimension
        return total_loss / batch_size


if __name__ == "__main__":
    # Sanity Check
    batch_dates = 32
    stocks = 482

    # Initialize with the lower bound of the required margin constraint
    loss_fn = HybridAlphaLoss(margin=0.1, decile=0.10, lambda_mse=1.0)

    dummy_preds = torch.randn(batch_dates, stocks, requires_grad=True)
    dummy_targets = torch.randn(batch_dates, stocks)

    loss = loss_fn(dummy_preds, dummy_targets)
    print(f"Calculated Hybrid AlphaLoss: {loss.item():.6f}")

    loss.backward()
    print("Backward pass successful. Gradients attached.")
