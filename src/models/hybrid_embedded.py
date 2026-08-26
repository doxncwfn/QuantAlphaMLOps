# Data leak Modal -> Aborted

import torch
import torch.nn as nn

class HybridAlphaRanker(nn.Module):
    """
    Hybrid LSTM-Transformer architecture for cross-sectional equity ranking,
    strictly matching the Phase 1 engineering specifications.
    """
    def __init__(self, num_stocks: int = 482, input_dim: int = 15, 
                 hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.num_stocks = num_stocks
        self.hidden_dim = hidden_dim
        self.embed_dim = 32
        
        # 1. Temporal Feature Extractor
        # Explicitly set to 2 layers per Task 1.2 requirements
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=dropout
        )
        
        # 2. Static Stock ID Embedding
        # 32-dimensional embedding concatenated to the LSTM state
        self.stock_embedding = nn.Embedding(
            num_embeddings=num_stocks, 
            embedding_dim=self.embed_dim
        )
        
        # 3. Cross-Sectional Transformer
        # Transformer dimension becomes hidden_dim (e.g. 128) + embed_dim (32)
        transformer_dim = hidden_dim + self.embed_dim
        
        # Task 1.2 explicitly mandates 4 heads, 2 layers, and a feedforward dim of 128
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=4, 
            dim_feedforward=128, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=2
        )
        
        # 4. Scoring Head
        self.scorer = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, 1)
        )

    def forward(self, x, stock_ids):
        """
        x shape: [Batch_Dates, Stocks, Seq_Len, Features]
        stock_ids shape: [Batch_Dates, Stocks] or [Stocks]
        """
        B, S, L, F = x.shape
        
        # --- 1. Temporal Processing (LSTM) ---
        # Fold batch and stocks together to process all 60-day sequences in parallel
        x_flat = x.view(B * S, L, F)
        
        # h_n shape for 2-layer LSTM: [2, Batch*Stocks, hidden_dim]
        _, (h_n, _) = self.lstm(x_flat)
        
        # Extract the hidden state from the FINAL (top) LSTM layer: h_n[-1]
        # Reshape to [Batch, Stocks, hidden_dim]
        lstm_out = h_n[-1].view(B, S, -1) 
        
        # --- 2. Stock Embedding ---
        if stock_ids.dim() == 1:
            stock_ids = stock_ids.unsqueeze(0).expand(B, -1)
            
        # embeds shape: [Batch, Stocks, 32]
        embeds = self.stock_embedding(stock_ids) 
        
        # --- 3. Feature Fusion ---
        # Concatenate LSTM temporal features with static stock identities
        # Shape: [Batch, Stocks, hidden_dim + 32]
        combined_features = torch.cat([lstm_out, embeds], dim=-1) 
        
        # --- 4. Cross-Sectional Attention ---
        # The Transformer treats 'Stocks' as the sequence dimension
        # z shape: [Batch, Stocks, hidden_dim + 32]
        z = self.transformer(combined_features) 
        
        # --- 5. Scoring ---
        # Compress down to a single scalar per stock per date
        # scores shape: [Batch, Stocks]
        scores = self.scorer(z).squeeze(-1) 
        
        return scores

if __name__ == "__main__":
    # Sanity Check to verify tensor dimensions and math (divisibility for heads)
    batch_size = 32
    num_stocks = 482
    seq_len = 60
    input_dim = 15
    
    # Testing with hidden_dim=64 to ensure (64+32) % 4 heads == 0
    model = HybridAlphaRanker(num_stocks=num_stocks, input_dim=input_dim, hidden_dim=64)
    
    dummy_x = torch.randn(batch_size, num_stocks, seq_len, input_dim)
    dummy_ids = torch.arange(num_stocks)
    
    out = model(dummy_x, dummy_ids)
    print(f"Network Output Shape: {out.shape} - Expected: [{batch_size}, {num_stocks}]")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")