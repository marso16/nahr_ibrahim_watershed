import torch
import torch.nn as nn


class WatershedMLP(nn.Module):
    """
    Simple feed-forward network. Flattens the lookback window into one long
    feature vector, then passes through stacked dense + SiLU + dropout layers.

    Parameter count is comparable to a small LSTM; the comparison is fair
    because the model has access to the same information (full lookback × features),
    just without any temporal inductive bias.
    """

    def __init__(
        self,
        input_dim: int,  # features per timestep
        seq_len: int,  # lookback length
        hidden_dims=(512, 256, 128),
        dropout: float = 0.30,
    ):
        super().__init__()
        flat_dim = input_dim * seq_len

        layers = []
        prev = flat_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, T, F) → flatten → (B, T*F)
        b = x.size(0)
        return self.net(x.reshape(b, -1))
