import torch
import torch.nn as nn
from typing import Optional


class LSTMClassifier(nn.Module):
    """
    LSTM Classifier for time-series patient data.

    Supports both unidirectional and bidirectional LSTMs with configurable
    pooling strategies. Matches the architecture of pre-trained model files.

    Args:
        input_dim: Number of input features.
        hidden_dim: Hidden state size for LSTM layers.
        num_layers: Number of stacked LSTM layers.
        bidirectional: If True, use bidirectional LSTM.
        dropout: Dropout probability between LSTM layers.
        pooling: Pooling strategy - 'mean', 'max', or 'last'.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pooling: str = "mean"
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling

        num_directions = 2 if bidirectional else 1
        lstm_output_dim = hidden_dim * num_directions

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # CRITICAL: Must use Sequential to match pre-trained model state dict keys
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, input_dim].

        Returns:
            Logit tensor of shape [batch, 1].
        """
        lstm_out, _ = self.lstm(x)

        if self.pooling == "mean":
            pooled = lstm_out.mean(dim=1)
        elif self.pooling == "max":
            pooled = lstm_out.max(dim=1).values
        elif self.pooling == "last":
            pooled = lstm_out[:, -1, :]
        else:
            pooled = lstm_out.mean(dim=1)

        return self.fc(pooled)