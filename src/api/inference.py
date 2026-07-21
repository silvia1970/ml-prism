"""
Inference Engine for PRISM API.
Wraps model loading and sliding window predictions.
"""
import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from collections import OrderedDict
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
from src.utils.field_mappings import MIMIC_FEATURES, SEPSIEXP_FEATURES

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def prepare_sepsisexp_data_inference(hour_df: pd.DataFrame, target_len: int = 24,
                                      padding_value: float = -100, stride: int = 6) -> Dict:
    """Prepare SepsisExp data for inference with sliding windows."""
    data_by_patient = {}
    y = hour_df["sepsis"].iloc[0] if "sepsis" in hour_df.columns else 0
    y_tensor = torch.tensor(y, dtype=torch.long)

    sepsis_onset_index = -1
    if y == 1 and "severity" in hour_df.columns:
        sepsis_onset_index = int(np.argmax(hour_df['severity']))

    total_len = len(hour_df)
    n_slices = max(1, (total_len - target_len) // stride + 1) if total_len >= target_len else 1

    drop_cols = ["id", "sepsis", "timestep", "severity"]
    feature_cols = [c for c in hour_df.columns if c not in drop_cols]
    features = hour_df[feature_cols].values

    for i in range(n_slices):
        start_idx = i * stride
        end_idx = start_idx + target_len
        frame = features[start_idx:end_idx]
        frame_tensor = pad_or_truncate_sequence(
            torch.tensor(frame, dtype=torch.float), target_length=target_len, padding_value=padding_value
        )
        data_by_patient[start_idx] = {
            "x": frame_tensor, "y": y_tensor,
            "start_index": start_idx, "end_index": min(end_idx - 1, total_len - 1),
            "sepsis_onset_index": sepsis_onset_index,
        }
    return data_by_patient


def prepare_mimic_data_inference(mimic_df: pd.DataFrame, target_len: int = 24,
                                  stride: int = 6, padding_value: float = -100) -> Dict:
    """Prepare MIMIC data for inference with sliding windows."""
    available_features = [f for f in MIMIC_FEATURES if f in mimic_df.columns]
    if not available_features:
        raise ValueError(f"No MIMIC features found. Expected: {MIMIC_FEATURES[:5]}...")

    if not is_standardized(mimic_df[available_features]):
        normalize_df = normalize_single_patient_mimic(mimic_df, available_features)
        for col in ['Index', 'timestep', 'sepsis', 'icustay_id']:
            if col in mimic_df.columns:
                normalize_df[col] = mimic_df[col].values
        mimic_df = normalize_df.copy()

    data_by_patient = {}
    y = mimic_df["sepsis"].iloc[0] if "sepsis" in mimic_df.columns else 0
    y_tensor = torch.tensor(y, dtype=torch.long)
    total_len = len(mimic_df)
    n_slices = max(1, (total_len - target_len) // stride + 1) if total_len >= target_len else 1

    drop_cols = ["Index", "sepsis", "timestep", "icustay_id", "label", "chart_time"]
    feature_cols = [c for c in mimic_df.columns if c not in drop_cols and c in available_features]
    features = mimic_df[feature_cols].values

    for i in range(n_slices):
        start_idx = i * stride
        end_idx = start_idx + target_len
        frame = features[start_idx:end_idx]
        frame_tensor = pad_or_truncate_sequence(
            torch.tensor(frame, dtype=torch.float), target_length=target_len, padding_value=padding_value
        )
        data_by_patient[start_idx] = {
            "x": frame_tensor, "y": y_tensor,
            "start_index": start_idx, "end_index": min(end_idx - 1, total_len - 1),
            "sepsis_onset_index": total_len,
        }

    return data_by_patient


INFER_MODEL_CONFIGS = {
    "sepsisexp": {
        "path": PROJECT_ROOT / "models/sepsisexp_24-6_None_29features_norm.pth",
        "input_dim": 27, "hidden_dim": 8, "num_layers": 4,
        "pooling": "max", "dropout": 0.2,
        "data_func": prepare_sepsisexp_data_inference,
    },
    "mimic": {
        "path": PROJECT_ROOT / "models/mimic_24-6_None.pth",
        "input_dim": 34, "hidden_dim": 4, "num_layers": 2,
        "pooling": "mean", "dropout": 0.3,
        "data_func": prepare_mimic_data_inference,
    },
}


class InferenceEngine:
    """Main inference engine for PRISM predictions."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self._load_models()

    def _load_models(self):
        for model_name, cfg in INFER_MODEL_CONFIGS.items():
            try:
                model = LSTMClassifier(
                    input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
                    num_layers=cfg["num_layers"], bidirectional=False,
                    pooling=cfg["pooling"], dropout=cfg["dropout"]
                ).to(self.device)
                if cfg["path"].exists():
                    model.load_state_dict(torch.load(cfg["path"], map_location=self.device, weights_only=False))
                    model.eval()
                    self.models[model_name] = model
                    logger.info(f"Loaded model '{model_name}' from {cfg['path']}")
                else:
                    logger.warning(f"Model file not found: {cfg['path']}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")

    def predict_from_csv(self, csv_data: pd.DataFrame, model_name: str,
                          target_len: int = 24, stride: int = 6) -> Dict:
        """Run sliding window predictions on CSV data."""
        if model_name not in INFER_MODEL_CONFIGS:
            return {"error": f"Unknown model: {model_name}", "windows": [], "summary": {}}
        if model_name not in self.models:
            return {"error": f"Model {model_name} not loaded", "windows": [], "summary": {}}

        cfg = INFER_MODEL_CONFIGS[model_name]
        model = self.models[model_name]

        try:
            data_by_patient = cfg["data_func"](csv_data, target_len=target_len, stride=stride)

            results = []
            total_prob = 0.0
            positive_count = 0

            for frame_n, data in data_by_patient.items():
                x = data["x"].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = model(x).squeeze()
                    prob = torch.sigmoid(logits).item()
                    pred = int(prob > 0.5)

                total_prob += prob
                if pred == 1:
                    positive_count += 1

                results.append(OrderedDict([
                    ("window_index", len(results)), ("n_frame", int(frame_n)),
                    ("prediction", pred), ("score", round(prob, 4)),
                    ("probability", round(prob, 4)), ("class", classify_risk(prob)),
                    ("start_index", data["start_index"]), ("end_index", data["end_index"]),
                ]))

            n_windows = len(results)
            avg_prob = total_prob / n_windows if n_windows > 0 else 0.0

            patient_id = None
            for col in ['id', 'icustay_id', 'Index']:
                if col in csv_data.columns:
                    patient_id = csv_data[col].iloc[0]
                    break

            return {
                "patient_id": patient_id, "model_used": model_name, "windows": results,
                "summary": {
                    "total_records": len(csv_data), "n_windows": n_windows,
                    "target_len": target_len, "stride": stride,
                    "avg_score": round(avg_prob, 4),
                    "at_risk_windows": positive_count,
                    "final_class": classify_risk(avg_prob),
                    "final_prediction": 1 if avg_prob >= 0.5 else 0,
                }
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return {"error": str(e), "windows": [], "summary": {}}

    def get_model_info(self, model_name: str) -> Dict:
        if model_name not in INFER_MODEL_CONFIGS:
            return {"error": f"Unknown model: {model_name}"}
        cfg = INFER_MODEL_CONFIGS[model_name]
        return {
            "model_name": model_name, "loaded": model_name in self.models,
            **{k: cfg[k] for k in ["input_dim", "hidden_dim", "num_layers", "pooling", "dropout"]},
        }

    def list_models(self) -> List[Dict]:
        return [self.get_model_info(name) for name in INFER_MODEL_CONFIGS.keys()]


_engine = None

def get_inference_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine