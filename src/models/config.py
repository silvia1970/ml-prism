from typing import Dict

# Risk classification thresholds
RISK_THRESHOLDS = {
    'low': 0.50,
    'moderate': 0.75,
}


def classify_risk(prob: float) -> str:
    """Classify a probability score into a risk category."""
    if prob >= 0.75:
        return 'high_risk'
    elif prob >= 0.50:
        return 'moderate_risk'
    else:
        return 'low_risk'


MODEL_CONFIGS: Dict[str, Dict] = {
    'mimic': {
        'input_dim': 34,
        'hidden_dim': 4,
        'num_layers': 2,
        'bidirectional': False,
        'dropout': 0.3,
        'pooling': 'mean',
        'max_seq_len': 48,
        'min_seq_len': 6,
        'model_path': 'models/mimic_24-6_None.pth',
        'stats_path': 'datasets/MIMIC/carry_forward/mean/backend/temporal_signature_info_split_0 (1).json',
    },
    'sepsisexp': {
        'input_dim': 27,
        'hidden_dim': 8,
        'num_layers': 4,
        'bidirectional': False,
        'dropout': 0.2,
        'pooling': 'max',
        'max_seq_len': 48,
        'min_seq_len': 6,
        'model_path': 'models/sepsisexp_24-6_None_29features_norm.pth',
        'stats_path': 'datasets/SepsisExp/original_sepsisexp/sepsisexp_normalization_global.json',
    },
}


def get_model_config(model_name: str) -> Dict:
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model_name.lower())


def is_valid_model_name(model_name: str) -> bool:
    """Check if a model name is valid."""
    return model_name.lower() in MODEL_CONFIGS