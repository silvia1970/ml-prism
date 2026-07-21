import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

from src.models.lstm import LSTMClassifier
from src.models.config import MODEL_CONFIGS, classify_risk
from src.utils.preprocessing import (
    normalize_single_patient_mimic,
    normalize_single_patient_sepsisexp,
    is_standardized,
    pad_or_truncate_sequence,
)
from src.utils.field_mappings import (
    map_json_to_pytorch, MIMIC_FEATURES, SEPSIEXP_FEATURES
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class ModelLoader:
    """Manages loading and inference for PRISM models."""

    MIMIC_METADATA = ['label', 'icustay_id', 'chart_time']
    SEPSIEXP_METADATA = ['id', 'timestep', 'severity', 'sepsis']

    def __init__(self, models_dir: str = 'models'):
        project_root = PROJECT_ROOT
        self.models_dir = project_root / models_dir
        self.models = {}
        self.device = torch.device('cpu')

        self.mimic_normalization = None
        self.sepsisexp_normalization = None

        # Load MIMIC normalization stats
        mimic_stats_path = project_root / 'datasets/MIMIC/carry_forward/mean/backend/temporal_signature_info_split_0 (1).json'
        try:
            with open(mimic_stats_path, 'r') as f:
                stats = json.load(f)
                self.mimic_normalization = {
                    name: (mean_val, std_val)
                    for name, mean_val, std_val in zip(stats['names'], stats['mean'], stats['std'])
                }
            logger.info(f"Loaded MIMIC normalization stats: {len(self.mimic_normalization)} features")
        except Exception as e:
            logger.warning(f"Could not load MIMIC normalization stats: {e}")

        # Load SepsisExp normalization stats
        sepsisexp_stats_path = project_root / 'datasets/SepsisExp/original_sepsisexp/scaler_stats_final.json'
        try:
            with open(sepsisexp_stats_path, 'r') as f:
                stats = json.load(f)
                self.sepsisexp_normalization = {
                    name: (feat_stats['mean'], feat_stats['std'])
                    for name, feat_stats in stats.items()
                }
            logger.info(f"Loaded SepsisExp normalization stats: {len(self.sepsisexp_normalization)} features")
        except Exception as e:
            logger.warning(f"Could not load SepsisExp normalization stats: {e}")

    @property
    def loaded_models(self):
        return list(self.models.keys())

    def load_model(self, model_name: str) -> bool:
        if model_name not in MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_name}")
            return False

        if model_name in self.models:
            return True

        config = MODEL_CONFIGS[model_name]
        model_path = self.models_dir / config['model_path']

        try:
            model = LSTMClassifier(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                bidirectional=config.get('bidirectional', False),
                pooling=config['pooling'],
                dropout=config['dropout']
            ).to(self.device)

            if model_path.exists():
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                model.load_state_dict(state_dict)
                model.eval()
                self.models[model_name] = {'model': model, 'config': config}
                logger.info(f"Model {model_name} loaded from {model_path}")
                return True
            else:
                logger.warning(f"Model path not found: {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False

    def load_all_models(self):
        for model_name in MODEL_CONFIGS.keys():
            self.load_model(model_name)

    def _extract_features_from_record(self, record: Dict, model_name: str) -> List[float]:
        pytorch_record = map_json_to_pytorch(record, model_name)
        features = MIMIC_FEATURES if model_name == 'mimic' else SEPSIEXP_FEATURES

        values = []
        for feat in features:
            val = pytorch_record.get(feat)
            if val is None or val == '':
                val = 0.0
            val = float(val)

            if model_name == 'mimic' and self.mimic_normalization and feat in self.mimic_normalization:
                mean_val, std_val = self.mimic_normalization[feat]
                if std_val > 0:
                    val = (val - mean_val) / std_val
            elif model_name == 'sepsisexp' and self.sepsisexp_normalization and feat in self.sepsisexp_normalization:
                mean_val, std_val = self.sepsisexp_normalization[feat]
                if std_val > 0:
                    val = (val - mean_val) / std_val

            values.append(val)
        return values

    def prepare_input(self, record: Dict, model_name: str) -> Optional[torch.Tensor]:
        values = self._extract_features_from_record(record, model_name)
        single_step = torch.tensor([values], dtype=torch.float32)
        padded = pad_or_truncate_sequence(single_step, target_length=24, padding_value=-100)
        return padded.unsqueeze(0).to(self.device)

    def predict(self, record: Dict, model_name: str) -> Dict:
        if model_name not in self.models:
            self.load_model(model_name)
            if model_name not in self.models:
                return {'score': 0.0, 'class': 'unknown', 'error': f'Model {model_name} not available'}

        try:
            model = self.models[model_name]['model']
            x = self.prepare_input(record, model_name)

            with torch.no_grad():
                logits = model(x).squeeze()
                prob = torch.sigmoid(logits).item()

            return {
                'score': round(prob, 4),
                'class': classify_risk(prob),
                'probability': round(prob, 4),
                'prediction': 1 if prob >= 0.5 else 0,
                'model_used': model_name,
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'score': 0.0, 'class': 'error', 'error': str(e)}

    def predict_sequence(self, records: List[Dict], model_name: str,
                          target_len: int = 24, stride: int = 6) -> Dict:
        if model_name not in self.models:
            self.load_model(model_name)
            if model_name not in self.models:
                return {'windows': [], 'summary': {'error': f'Model {model_name} not available'}}

        try:
            model = self.models[model_name]['model']
            all_values = [self._extract_features_from_record(r, model_name) for r in records]
            features_tensor = torch.tensor(all_values, dtype=torch.float32)
            total_len = len(records)

            n_slices = max(1, (total_len - target_len) // stride + 1) if total_len >= target_len else 1

            results = []
            total_prob = 0.0
            positive_count = 0

            for i in range(n_slices):
                start_idx = i * stride
                end_idx = start_idx + target_len
                frame = features_tensor[start_idx:end_idx] if end_idx <= total_len else features_tensor[start_idx:]
                frame = pad_or_truncate_sequence(frame, target_len, padding_value=-100)

                with torch.no_grad():
                    logits = model(frame.unsqueeze(0).to(self.device)).squeeze()
                    prob = torch.sigmoid(logits).item()

                pred = 1 if prob >= 0.5 else 0
                total_prob += prob
                if pred == 1:
                    positive_count += 1

                results.append({
                    'window_index': i,
                    'start_index': start_idx,
                    'end_index': min(end_idx - 1, total_len - 1),
                    'score': round(prob, 4),
                    'class': classify_risk(prob),
                    'probability': round(prob, 4),
                    'prediction': pred,
                    'model_used': model_name,
                })

            n_windows = len(results)
            avg_prob = total_prob / n_windows if n_windows > 0 else 0.0

            return {
                'windows': results,
                'summary': {
                    'total_records': len(records),
                    'n_windows': n_windows,
                    'avg_score': round(avg_prob, 4),
                    'at_risk_windows': positive_count,
                    'final_class': classify_risk(avg_prob),
                    'model_used': model_name,
                }
            }
        except Exception as e:
            logger.error(f"Sequence prediction error: {e}")
            return {'windows': [], 'summary': {'error': str(e)}}