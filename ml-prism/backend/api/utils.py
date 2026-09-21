"""
Shared utilities for PRISM Backend.

This module provides common functions and classes used across the codebase,
eliminating code duplication and improving maintainability.
"""

import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import torch
import torch.nn as nn


# =============================================================================
# Risk Classification
# =============================================================================

def _classify_risk(prob: float) -> str:
    """
    Classify a probability score into a risk category.

    Args:
        prob: Probability score between 0 and 1.

    Returns:
        Risk category string: 'low_risk', 'moderate_risk', or 'high_risk'.

    Risk Thresholds:
        - low_risk:      prob < 0.50
        - moderate_risk: 0.50 <= prob < 0.75
        - high_risk:     prob >= 0.75
    """
    if prob >= 0.75:
        return 'high_risk'
    elif prob >= 0.50:
        return 'moderate_risk'
    else:
        return 'low_risk'


# =============================================================================
# LSTM Model Architecture
# =============================================================================

class LSTMClassifier(nn.Module):
    """
    LSTM Classifier for time-series patient data.

    This is the canonical definition used by both torch_models.py and
    inference_engine.py. The model supports both unidirectional and
    bidirectional LSTMs with configurable pooling strategies.

    IMPORTANT: The `fc` layer uses nn.Sequential(Dropout, Linear) to match
    the architecture of the pre-trained model files. Changing this will
    break model loading.

    Args:
        input_dim: Number of input features.
        hidden_dim: Hidden state size for LSTM layers.
        num_layers: Number of stacked LSTM layers.
        bidirectional: If True, use bidirectional LSTM.
        dropout: Dropout probability between LSTM layers.
        pooling: Pooling strategy - 'mean', 'max', or 'last'.

    Example:
        >>> model = LSTMClassifier(input_dim=34, hidden_dim=64, num_layers=2)
        >>> x = torch.randn(1, 24, 34)  # [batch, seq_len, features]
        >>> output = model(x)
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
        # State dict expects: fc.1.weight, fc.1.bias (not fc.weight, fc.bias)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM classifier.

        Args:
            x: Input tensor of shape [batch, seq_len, input_dim].

        Returns:
            Logit tensor of shape [batch, 1].
        """
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim * num_dirs]

        if self.pooling == "mean":
            pooled = lstm_out.mean(dim=1)
        elif self.pooling == "max":
            pooled = lstm_out.max(dim=1).values
        elif self.pooling == "last":
            pooled = lstm_out[:, -1, :]
        else:
            pooled = lstm_out.mean(dim=1)

        out = self.fc(pooled)
        return out


# =============================================================================
# Time Utilities
# =============================================================================

def get_current_timestamp() -> str:
    """
    Get the current UTC timestamp as an ISO 8601 string.

    Uses timezone-aware datetime to avoid deprecation warnings from
    datetime.utcnow().

    Returns:
        ISO 8601 formatted timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def get_current_datetime() -> datetime:
    """
    Get the current timezone-aware UTC datetime.

    Returns:
        datetime object in UTC timezone.
    """
    return datetime.now(timezone.utc)


# =============================================================================
# Path Utilities
# =============================================================================

def get_base_dir() -> str:
    """
    Get the project base directory (parent of the api/ directory).

    Returns:
        Absolute path to the project root directory.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_models_dir() -> str:
    """
    Get the models directory path.

    Returns:
        Absolute path to the models/ directory.
    """
    return os.path.join(get_base_dir(), 'models')


def get_api_data_dir() -> str:
    """
    Get the api_data directory path.

    Returns:
        Absolute path to the api_data/ directory.
    """
    return os.path.join(get_base_dir(), 'api_data')


def ensure_directory(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure.

    Returns:
        The (potentially created) directory path.
    """
    os.makedirs(path, exist_ok=True)
    return path


# =============================================================================
# Model Configuration
# =============================================================================

MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    'mimic': {
        'input_dim': 34,
        'hidden_dim': 4,
        'num_layers': 2,
        'bidirectional': False,
        'dropout': 0.3,
        'pooling': 'mean',
        'max_seq_len': 48,
        'min_seq_len': 6,
        'model_path': 'models/mimic/lstm_model_mimic.pth',
        'stats_path': 'models/mimic/normalization_stats.json',
    },
    'sepsisexp': {
        'input_dim': 27,
        'hidden_dim': 8,
        'num_layers': 4,
        'bidirectional': False,
        'dropout': 0.3,
        'pooling': 'max',
        'max_seq_len': 48,
        'min_seq_len': 6,
        'model_path': 'models/sepsisexp/lstm_model_sepsisexp.pth',
        'stats_path': 'models/sepsisexp/normalization_stats.json',
    },
}


def get_model_config(model_name: str) -> Optional[Dict[str, object]]:
    """
    Get configuration for a specific model.

    Args:
        model_name: Model identifier ('mimic' or 'sepsisexp').

    Returns:
        Configuration dictionary, or None if model not found.
    """
    return MODEL_CONFIGS.get(model_name.lower())


def is_valid_model_name(model_name: str) -> bool:
    """
    Check if a model name is valid.

    Args:
        model_name: Model identifier to check.

    Returns:
        True if the model name is recognized.
    """
    return model_name.lower() in MODEL_CONFIGS
