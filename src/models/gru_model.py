"""
GRU baseline with bidirectional + attention, mirroring the BiLSTM Strategy A
architecture but with GRU cells.

Key differences vs LSTM:
  - GRU has 3 gates (update, reset, output) vs LSTM's 4 (input, forget, output, candidate)
  - Roughly 25% fewer parameters per layer
  - Often comparable accuracy on small datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over a sequence."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs):
        # encoder_outputs: (B, T, H)
        scores = self.v(torch.tanh(self.W(encoder_outputs)))  # (B, T, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, H)
        return context, weights


class WatershedGRU(nn.Module):
    """
    BiGRU + multi-head self-attention + additive attention pool + dense head.

    Same input shape as WatershedLSTM: (batch, seq_len, n_features).
    Output: (batch, 1) discharge prediction in normalized space.
    """

    def __init__(
        self, input_dim, hidden_dim=128, num_layers=2, n_heads=4, dropout=0.30
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_norm = nn.LayerNorm(input_dim)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        gru_out_dim = hidden_dim * 2  # bidirectional

        # Multi-head self-attention on top of GRU outputs
        self.self_attn = nn.MultiheadAttention(
            embed_dim=gru_out_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(gru_out_dim)

        # Additive attention pooling to compress sequence into single vector
        self.pool_attn = AdditiveAttention(gru_out_dim)

        # Dense head
        self.head = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_norm(x)

        # BiGRU
        gru_out, _ = self.gru(x)  # (B, T, 2H)

        # Self-attention with residual
        attn_out, _ = self.self_attn(gru_out, gru_out, gru_out)
        gru_out = self.attn_norm(gru_out + attn_out)

        # Pool to single vector
        context, _ = self.pool_attn(gru_out)

        # Dense head → discharge prediction
        return self.head(context)
