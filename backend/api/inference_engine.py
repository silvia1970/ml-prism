"""
Inference Engine for PRISM API

Wraps functions from run_Prism.py for sliding window predictions on CSV data.
Uses shared utilities from api.utils for LSTMClassifier and risk classification.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent

# Import shared utilities (LSTMClassifier, _classify_risk)
from api.utils import LSTMClassifier, _classify_risk


def pad_or_truncate_sequence_inference(seq: torch.Tensor, target_length: int, 
                                        padding_value: float = -100) -> torch.Tensor:
    """
    Pad or truncate sequence to target length
    From run_Prism.py
    """
    # Padding se troppo corta
    if len(seq) < target_length:
        pad_len = target_length - len(seq)
        padding = torch.full((pad_len, *seq.shape[1:]), padding_value)
        seq = torch.cat([padding, seq], dim=0)

    # Troncamento finale (per sicurezza)
    if len(seq) > target_length:
        seq = seq[-target_length:]

    return seq


def is_standardized(df: pd.DataFrame, tol_mean: float = 1.0, tol_std: float = 2.0) -> bool:
    """Check if data is already standardized"""
    numeric_df = df.select_dtypes(include=[np.number])
    means = numeric_df.mean()
    stds = numeric_df.std()
    
    mean_check = (means.abs() < tol_mean).all()
    std_check = ((stds - 1).abs() < tol_std).all()
    
    return mean_check and std_check


def normalize_single_patient_mimic(single_patient_df: pd.DataFrame, 
                                   features_selected: List[str],
                                   path_to_json: str = None) -> pd.DataFrame:
    """
    Normalize MIMIC patient data using stored statistics
    From run_Prism.py
    """
    if path_to_json is None:
        path_to_json = PROJECT_ROOT / "datasets/MIMIC/carry_forward/mean/backend/temporal_signature_info_split_0 (1).json"
    
    with open(path_to_json) as f:
        mimic_dict = json.load(f)
    
    normalize_dict = {}
    
    for name, mean, std in zip(mimic_dict['names'], mimic_dict['mean'], mimic_dict['std']):
        if name in features_selected and name in single_patient_df.columns:
            if std > 0:
                normalize_dict[name] = (single_patient_df[name] - mean) / std
            else:
                normalize_dict[name] = single_patient_df[name] - mean
    
    return pd.DataFrame(normalize_dict)


def normalize_single_patient_sepsisexp(single_patient_df: pd.DataFrame,
                                        path_to_json: str = None) -> pd.DataFrame:
    """
    Normalize SepsisExp patient data using stored statistics
    From run_Prism.py
    """
    if path_to_json is None:
        path_to_json = PROJECT_ROOT / "datasets/SepsisExp/original_sepsisexp/sepsisexp_normalization_global.json"
    
    with open(path_to_json) as f:
        sepsisexp_dict = json.load(f)
    
    normalize_dict = {}
    
    for col in single_patient_df.columns:
        if col not in ['id', 'sepsis', 'severity', 'timestep']:
            if col in sepsisexp_dict:
                mean_val = sepsisexp_dict[col]["mean"]
                std_val = sepsisexp_dict[col]["std"]
                if std_val > 0:
                    normalize_dict[col] = (single_patient_df[col] - mean_val) / std_val
                else:
                    normalize_dict[col] = single_patient_df[col] - mean_val
    
    normalize_df = pd.DataFrame(normalize_dict)
    
    # Preserve metadata columns
    for col in ['id', 'sepsis', 'severity', 'timestep']:
        if col in single_patient_df.columns:
            normalize_df[col] = single_patient_df[col].values
    
    return normalize_df


# Feature lists from run_Prism.py
MIMIC_FEATURES = [
    "meanbp", "resprate", "heartrate", "spo2_pulsoxy", "tempc",
    "cardiacoutput", "o2flow", "fio2", "albumin", "bands",
    "bicarbonate", "bilirubin", "creatinine", "chloride", "glucose",
    "hemoglobin", "lactate", "platelet", "potassium", "ptt",
    "inr", "sodium", "wbc", "creatinekinase", "ck_mb",
    "fibrinogen", "ldh", "magnesium", "calcium_free", "po2_bloodgas",
    "ph_bloodgas", "pco2_bloodgas", "so2_bloodgas", "troponin_t"
]  # 34 features

SEPSIEXP_FEATURES = [
    'age', 'heart_rate', 'svri', 'mean_bp', 'heart_time_volume',
    'oxygen_saturation', 'delta-temperature', 'mixed_venous_oxygen_saturation',
    'lactate', 'creatinine', 'bilirubin', 'sodium', 'potassium',
    'hemoglobin', 'chloride', 'leukocytes', 'bicarbonate',
    'pancreatic_lipase', 'blood_urea_nitrogen', 'procalcitonin',
    'bun/creatinine_ratio', 'aspartate_transaminase',
    'c-reactive_protein', 'respiratory_minute_volume',
    'fraction_of_inspired_o2', 'arterial_ph', 'partial_pressure_art._o2'
]  # 27 features


def prepare_sepsisexp_data_inference(hour_df: pd.DataFrame, 
                                      target_len: int = 24, 
                                      padding_value: float = -100, 
                                      stride: int = 6) -> Dict:
    """
    Prepare SepsisExp data for inference with sliding window
    From run_Prism.py
    """
    data_by_patient = {}

    # Get label if available
    y = hour_df["sepsis"].iloc[0] if "sepsis" in hour_df.columns else 0
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Find sepsis onset index
    sepsis_onset_index = -1
    if y == 1 and "severity" in hour_df.columns:
        sepsis_onset_index = int(np.argmax(hour_df['severity']))
    else:
        sepsis_onset_index = 0

    total_len = len(hour_df)

    # Calculate number of windows
    if total_len < target_len:
        n_slices = 1
    else:
        n_slices = (total_len - target_len) // stride + 1

    # Select feature columns
    drop_cols = ["id", "sepsis", "timestep", "severity"]
    feature_cols = [c for c in hour_df.columns if c not in drop_cols]
    features = hour_df[feature_cols].values

    # Create sliding windows
    for i in range(n_slices):
        start_idx = i * stride
        end_idx = start_idx + target_len
        frame = features[start_idx:end_idx]

        frame_tensor = pad_or_truncate_sequence_inference(
            torch.tensor(frame, dtype=torch.float),
            target_length=target_len,
            padding_value=padding_value
        )

        last_index_window = min(end_idx - 1, total_len - 1)

        data_by_patient[start_idx] = {
            "x": frame_tensor,
            "y": y_tensor,
            "start_index": start_idx,
            "end_index": last_index_window,
            "sepsis_onset_index": sepsis_onset_index
        }

    return data_by_patient


def prepare_mimic_data_inference(mimic_df: pd.DataFrame, 
                                  target_len: int = 24, 
                                  stride: int = 6, 
                                  padding_value: float = -100) -> Dict:
    """
    Prepare MIMIC data for inference with sliding window
    From run_Prism.py
    """
    features_selected = MIMIC_FEATURES
    
    # Check which features are available
    available_features = [f for f in features_selected if f in mimic_df.columns]
    
    if not available_features:
        raise ValueError(f"No MIMIC features found in dataframe. Expected: {features_selected[:5]}...")
    
    # Normalize if not already standardized
    if not is_standardized(mimic_df[available_features]):
        normalize_df = normalize_single_patient_mimic(mimic_df, available_features)
        
        # Add metadata columns back
        if 'Index' in mimic_df.columns:
            normalize_df['Index'] = mimic_df['Index'].values
        if 'timestep' in mimic_df.columns:
            normalize_df['timestep'] = mimic_df['timestep'].values
        if 'sepsis' in mimic_df.columns:
            normalize_df['sepsis'] = mimic_df['sepsis'].values
        if 'icustay_id' in mimic_df.columns:
            normalize_df['icustay_id'] = mimic_df['icustay_id'].values
        
        mimic_df = normalize_df.copy()
    
    data_by_patient = {}

    # Get label if available
    y = mimic_df["sepsis"].iloc[0] if "sepsis" in mimic_df.columns else 0
    y_tensor = torch.tensor(y, dtype=torch.long)
    sepsis_onset_index = mimic_df.shape[0]

    total_len = len(mimic_df)

    # Calculate number of windows
    if total_len < target_len:
        n_slices = 1
    else:
        n_slices = (total_len - target_len) // stride + 1

    # Get feature columns only (drop metadata)
    drop_cols = ["Index", "sepsis", "timestep", "icustay_id", "label", "chart_time"]
    feature_cols = [c for c in mimic_df.columns if c not in drop_cols and c in available_features]
    features = mimic_df[feature_cols].values

    # Create sliding windows
    for i in range(n_slices):
        start_idx = i * stride
        end_idx = start_idx + target_len
        frame = features[start_idx:end_idx]

        frame_tensor = pad_or_truncate_sequence_inference(
            torch.tensor(frame, dtype=torch.float),
            target_length=target_len,
            padding_value=padding_value
        )

        last_index_window = min(end_idx - 1, total_len - 1)

        data_by_patient[start_idx] = {
            "x": frame_tensor,
            "y": y_tensor,
            "start_index": start_idx,
            "end_index": last_index_window,
            "sepsis_onset_index": sepsis_onset_index
        }

    return data_by_patient


# Model configurations from run_Prism.py
MODEL_CONFIGS = {
    "sepsisexp": {
        "path": PROJECT_ROOT / "models/sepsisexp_24-6_None_29features_norm.pth",
        "input_dim": 27,
        "hidden_dim": 8,
        "num_layers": 4,
        "pooling": "max",
        "dropout": 0.2,
        "data_func": prepare_sepsisexp_data_inference
    },
    "mimic": {
        "path": PROJECT_ROOT / "models/mimic_24-6_None.pth",
        "input_dim": 34,
        "hidden_dim": 4,
        "num_layers": 2,
        "pooling": "mean",
        "dropout": 0.3,
        "data_func": prepare_mimic_data_inference
    }
}


class InferenceEngine:
    """
    Main inference engine for PRISM predictions
    Wraps run_Prism.py functionality for API use
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all available models"""
        for model_name, cfg in MODEL_CONFIGS.items():
            try:
                model = LSTMClassifier(
                    input_dim=cfg["input_dim"],
                    hidden_dim=cfg["hidden_dim"],
                    num_layers=cfg["num_layers"],
                    pooling=cfg["pooling"],
                    dropout=cfg["dropout"]
                ).to(self.device)
                
                if cfg["path"].exists():
                    model.load_state_dict(
                        torch.load(cfg["path"], map_location=self.device, weights_only=False)
                    )
                    model.eval()
                    self.models[model_name] = model
                    logger.info(f"✓ Loaded model '{model_name}' from {cfg['path']}")
                else:
                    logger.warning(f"Model file not found: {cfg['path']}")
                    
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
    
    def predict_from_csv(self, csv_data: pd.DataFrame, model_name: str,
                         target_len: int = 24, stride: int = 6) -> Dict:
        """
        Run sliding window predictions on CSV data
        Replicates the full flow from run_Prism.py /predict endpoint
        
        Args:
            csv_data: DataFrame with patient data
            model_name: 'mimic' or 'sepsisexp'
            target_len: Window size (default 24)
            stride: Sliding stride (default 1)
            
        Returns:
            Dict with windows, summary, and metadata
        """
        if model_name not in MODEL_CONFIGS:
            return {
                "error": f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}",
                "windows": [],
                "summary": {}
            }
        
        if model_name not in self.models:
            return {
                "error": f"Model {model_name} not loaded",
                "windows": [],
                "summary": {}
            }
        
        cfg = MODEL_CONFIGS[model_name]
        prepare_func = cfg["data_func"]
        model = self.models[model_name]
        
        try:
            # Prepare data with sliding windows
            data_by_patient = prepare_func(
                csv_data,
                target_len=target_len,
                padding_value=-100,
                stride=stride
            )
            
            # Run predictions on each window
            results = []
            total_prob = 0.0
            positive_count = 0
            
            for frame_n, data in data_by_patient.items():
                x = data["x"].unsqueeze(0).to(self.device)  # [1, T, F]
                
                with torch.no_grad():
                    logits = model(x).squeeze()
                    prob = torch.sigmoid(logits).item()
                    pred = int(prob > 0.5)
                
                score = round(prob, 4)
                total_prob += prob
                if pred == 1:
                    positive_count += 1
                
                results.append(OrderedDict([
                    ("window_index", len(results)),
                    ("n_frame", int(frame_n)),
                    ("prediction", pred),
                    ("score", score),
                    ("probability", round(prob, 4)),
                    ("class", _classify_risk(prob)),
                    ("start_index", data["start_index"]),
                    ("end_index", data["end_index"]),
                    ("sepsis_onset_index", data["sepsis_onset_index"])
                ]))
            
            n_windows = len(results)
            avg_prob = total_prob / n_windows if n_windows > 0 else 0.0
            avg_score = round(avg_prob, 4)
            
            # Get patient ID if available
            patient_id = None
            if 'id' in csv_data.columns:
                patient_id = csv_data['id'].iloc[0]
            elif 'icustay_id' in csv_data.columns:
                patient_id = csv_data['icustay_id'].iloc[0]
            elif 'Index' in csv_data.columns:
                patient_id = csv_data['Index'].iloc[0]
            
            # Get true label if available
            true_label = None
            if 'sepsis' in csv_data.columns:
                true_label = int(csv_data['sepsis'].iloc[0])
            
            return {
                "patient_id": patient_id,
                "true_label": true_label,
                "model_used": model_name,
                "windows": results,
                "summary": {
                    "total_records": len(csv_data),
                    "n_windows": n_windows,
                    "target_len": target_len,
                    "stride": stride,
                    "avg_score": avg_score,
                    "avg_probability": round(avg_prob, 4),
                    "at_risk_windows": positive_count,
                    "low_risk_windows": n_windows - positive_count,
                    "at_risk_ratio": round(positive_count / n_windows, 4) if n_windows > 0 else 0.0,
                    "final_class": _classify_risk(avg_prob),
                    "final_prediction": 1 if avg_prob >= 0.5 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "windows": [],
                "summary": {
                    "total_records": len(csv_data),
                    "n_windows": 0,
                    "error": str(e)
                }
            }
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a model"""
        if model_name not in MODEL_CONFIGS:
            return {"error": f"Unknown model: {model_name}"}
        
        cfg = MODEL_CONFIGS[model_name]
        loaded = model_name in self.models
        
        return {
            "model_name": model_name,
            "loaded": loaded,
            "input_dim": cfg["input_dim"],
            "hidden_dim": cfg["hidden_dim"],
            "num_layers": cfg["num_layers"],
            "pooling": cfg["pooling"],
            "dropout": cfg["dropout"],
            "model_path": str(cfg["path"]),
            "features": MIMIC_FEATURES if model_name == "mimic" else SEPSIEXP_FEATURES
        }
    
    def list_models(self) -> List[Dict]:
        """List all available models"""
        return [self.get_model_info(name) for name in MODEL_CONFIGS.keys()]


# Singleton instance
_engine = None

def get_inference_engine() -> InferenceEngine:
    """Get or create the inference engine singleton"""
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine
