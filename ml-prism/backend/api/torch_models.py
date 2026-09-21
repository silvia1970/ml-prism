"""
PyTorch Model Loader for PRISM

Loads and manages LSTM models for MIMIC and SepsisExp predictions.
Uses shared utilities from api.utils for LSTMClassifier and risk classification.
"""

import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Import field mappings
from api.field_mappings import map_json_to_pytorch
# Import shared utilities
from api.utils import LSTMClassifier, _classify_risk

logger = logging.getLogger(__name__)


class ModelLoader:
    """Manages loading and inference for PRISM models"""
    
    # Model configurations
    MODEL_CONFIGS = {
        'mimic': {
            'input_dim': 34,  # Number of features for MIMIC (actual trained model size)
            'hidden_dim': 4,
            'num_layers': 2,
            'pooling': 'mean',
            'dropout': 0.3,
            'path': 'mimic_24-6_None.pth'
        },
        'sepsiexp': {
            'input_dim': 27,  # Model trained with age + 26 clinical features (timestamp excluded)
            'hidden_dim': 8,  # Hidden dimension (weights show 32 = 8*4 bidirectional)
            'num_layers': 4,
            'pooling': 'max',
            'dropout': 0.2,
            'path': 'sepsisexp_24-6_None_29features_norm.pth'
        }
    }
    
    # MIMIC: Metadata fields (NOT used in model)
    # icustay_id: Primary identifier for MIMIC (maps to sample_id in API)
    # label: Target variable (not used for prediction input)
    # chart_time: Timestamp (maps to timestamp in API)
    MIMIC_METADATA = ['label', 'icustay_id', 'chart_time']
    
    # MIMIC: 34 features (all used in model)

    #sysbp	diabp NOT USED BY MODEL (36 features)
    # Feature mappings for each model (from training scripts)
    MIMIC_FEATURES = [
        'meanbp', 'resprate', 'heartrate', 'spo2_pulsoxy', 'tempc',
        'cardiacoutput', 'o2flow', 'fio2', 'albumin', 'bands',
        'bicarbonate', 'bilirubin', 'creatinine', 'chloride', 'glucose',
        'hemoglobin', 'lactate', 'platelet', 'potassium', 'ptt',
        'inr', 'sodium', 'wbc', 'creatinekinase', 'ck_mb',
        'fibrinogen', 'ldh', 'magnesium', 'calcium_free', 'po2_bloodgas',
        'ph_bloodgas', 'pco2_bloodgas', 'so2_bloodgas', 'troponin_t'
    ]  # 34 features
    
    # SepsisExp: 31 total columns (4 metadata + 27 features for model)
    # Metadata fields (NOT used in model, but saved for retrieval):
    # id: Primary identifier for SepsisExp (maps to sample_id in API)
    # timestep: Timestep number (integer)
    # severity: Severity level (metadata)
    # sepsis: Target variable (0 or 1)
    SEPSIEXP_METADATA = ['id', 'timestep', 'severity', 'sepsis']
    # SepsisExp: 27 features (age + 26 clinical features) - USED IN MODEL
    SEPSIEXP_FEATURES = [
        'age',  # Age is used as model input
        'heart_rate', 'svri', 'mean_bp', 'heart_time_volume',
        'oxygen_saturation', 'delta-temperature', 'mixed_venous_oxygen_saturation',
        'lactate', 'creatinine', 'bilirubin', 'sodium', 'potassium',
        'hemoglobin', 'chloride', 'leukocytes', 'bicarbonate',
        'pancreatic_lipase', 'blood_urea_nitrogen', 'procalcitonin',
        'bun/creatinine_ratio', 'aspartate_transaminase',
        'c-reactive_protein', 'respiratory_minute_volume',
        'fraction_of_inspired_o2', 'arterial_ph', 'partial_pressure_art._o2'
    ]  # 27 features (age + 26 clinical)
    
    # All SepsisExp columns (metadata + features) - Total 31
    SEPSIEXP_ALL_COLUMNS = SEPSIEXP_METADATA + SEPSIEXP_FEATURES

    MIMIC_ALL_COLUMNS = MIMIC_METADATA + MIMIC_FEATURES
    
    def __init__(self, models_dir: str = 'models'):
        """Initialize model loader"""
        # Get project root directory (parent of api/)
        project_root = Path(__file__).parent.parent
        self.models_dir = project_root / models_dir
        self.models = {}
        self.device = torch.device('cpu')  # Use CPU for API
        
        # Load MIMIC normalization statistics (MIMIC requires normalization)
        mimic_stats_path = project_root / 'datasets/MIMIC/carry_forward/mean/backend/temporal_signature_info_split_0 (1).json'
        self.mimic_normalization = None

        sepsiexp_stats_path = project_root / 'datasets/SepsisExp/original_sepsisexp/scaler_stats_final.json'
        self.sepsiexp_normalization = None
        try:
            with open(mimic_stats_path, 'r') as f:
                stats = json.load(f)
                # Create dict mapping feature name -> (mean, std)
                self.mimic_normalization = {
                    name: (mean_val, std_val)
                    for name, mean_val, std_val in zip(stats['names'], stats['mean'], stats['std'])
                }
            logger.info(f"✓ Loaded MIMIC normalization stats: {len(self.mimic_normalization)} features")
        except Exception as e:
            logger.warning(f"Could not load MIMIC normalization stats: {e}")
            logger.warning("MIMIC predictions may be inaccurate without normalization!")
        try:
            with open(sepsiexp_stats_path, 'r') as f:
                stats = json.load(f)
                # SepsisExp stats format: {feature_name: {mean: X, std: Y}}
                self.sepsiexp_normalization = {
                    name: (feat_stats['mean'], feat_stats['std'])
                    for name, feat_stats in stats.items()
                }
            logger.info(f"✓ Loaded SepsisExp normalization stats: {len(self.sepsiexp_normalization)} features")
        except Exception as e:
            logger.warning(f"Could not load SepsisExp normalization stats: {e}")
            logger.warning("SepsisExp predictions may be inaccurate without normalization!")
        
    @property
    def loaded_models(self):
        """Return list of loaded model names"""
        return list(self.models.keys())
        
    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model
        
        Args:
            model_name: 'mimic' or 'sepsiexp'
            
        Returns:
            True if loaded successfully
        """
        if model_name not in self.MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_name}")
            return False
            
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded")
            return True
            
        config = self.MODEL_CONFIGS[model_name]
        model_path = self.models_dir / config['path']
        
        try:
            # Initialize model with config
            model = LSTMClassifier(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                pooling=config['pooling'],
                dropout=config['dropout']
            ).to(self.device)
            
            # Load state dict direttamente dalla directory PyTorch
            # PyTorch salva i modelli come directory con struttura specifica
            # Possiamo caricare direttamente usando torch.load sulla directory
            if model_path.exists():
                try:
                    # Carica state dict direttamente dalla directory del modello
                    state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                    model.load_state_dict(state_dict)
                    model.eval()
                    
                    self.models[model_name] = {
                        'model': model,
                        'config': config
                    }
                    
                    logger.info(f"✓ Model {model_name} loaded successfully from {model_path}")
                    return True
                    
                except Exception as load_error:
                    logger.error(f"Error loading state dict for {model_name}: {load_error}")
                    return False
            else:
                logger.warning(f"Model path not found: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return False
    
    def load_all_models(self):
        """Load all available models"""
        for model_name in self.MODEL_CONFIGS.keys():
            self.load_model(model_name)
    
    def _pad_or_truncate_sequence(self, seq: torch.Tensor, target_length: int = 24, 
                                   padding_value: float = -100) -> torch.Tensor:
        """
        Pad or truncate sequence to target length (aligned with run_Prism.py)
        
        Args:
            seq: Tensor of shape [T, F]
            target_length: Target sequence length (default 24)
            padding_value: Value for padding (default -100)
            
        Returns:
            Tensor of shape [target_length, F]
        """
        # Padding if too short
        if len(seq) < target_length:
            pad_len = target_length - len(seq)
            padding = torch.full((pad_len, seq.shape[1]), padding_value, dtype=seq.dtype)
            seq = torch.cat([padding, seq], dim=0)
        
        # Truncate if too long (keep last target_length)
        if len(seq) > target_length:
            seq = seq[-target_length:]
        
        return seq
    
    def _extract_features_from_record(self, record: Dict, model_name: str) -> List[float]:
        """
        Extract normalized feature values from a single record
        
        Args:
            record: Patient data dictionary
            model_name: 'mimic' or 'sepsiexp'
            
        Returns:
            List of normalized feature values
        """
        # Map JSON schema field names to PyTorch model field names
        pytorch_record = map_json_to_pytorch(record, model_name)
        
        # Select features based on model
        features = self.MIMIC_FEATURES if model_name == 'mimic' else self.SEPSIEXP_FEATURES
        
        values = []
        for feat in features:
            val = pytorch_record.get(feat)
            
            # Handle missing values
            if val is None or val == '':
                val = 0.0
            # Handle timestamp - convert ISO8601 to unix timestamp
            elif feat == 'timestamp' and isinstance(val, str):
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(val.replace('Z', '+00:00'))
                    val = float(dt.timestamp())
                except:
                    val = 0.0
            # Handle categorical values (sex)
            elif feat == 'sex':
                if isinstance(val, str):
                    val = {'M': 1.0, 'F': 0.0, 'X': 0.5}.get(val.upper(), 0.0)
            
            # Convert to float
            val = float(val)
            
            # Apply normalization
            if model_name == 'mimic' and self.mimic_normalization and feat in self.mimic_normalization:
                mean_val, std_val = self.mimic_normalization[feat]
                if std_val > 0:
                    val = (val - mean_val) / std_val
            elif model_name == 'sepsiexp' and self.sepsiexp_normalization and feat in self.sepsiexp_normalization:
                mean_val, std_val = self.sepsiexp_normalization[feat]
                if std_val > 0:
                    val = (val - mean_val) / std_val
            
            values.append(val)
        
        return values
    
    def prepare_input(self, record: Dict, model_name: str) -> Optional[torch.Tensor]:
        """
        Prepare input tensor from a single patient record (single timestep with padding)
        
        Args:
            record: Patient data dictionary (with JSON schema field names)
            model_name: 'mimic' or 'sepsiexp'
            
        Returns:
            Tensor of shape [1, 24, num_features] (padded to window size)
        """
        values = self._extract_features_from_record(record, model_name)
        
        # Create tensor [1, F] for single timestep
        single_step = torch.tensor([values], dtype=torch.float32)
        
        # Pad to target length [24, F]
        padded = self._pad_or_truncate_sequence(single_step, target_length=24, padding_value=-100)
        
        # Add batch dimension [1, 24, F]
        tensor = padded.unsqueeze(0)
        return tensor.to(self.device)
    
    def prepare_input_sequence(self, records: List[Dict], model_name: str, 
                                target_len: int = 24, stride: int = 6) -> List[Dict]:
        """
        Prepare input tensors from a sequence of patient records using sliding window
        (Aligned with run_Prism.py prepare_*_data_inference functions)
        
        Args:
            records: List of patient data dictionaries (ordered by time)
            model_name: 'mimic' or 'sepsiexp'
            target_len: Window size (default 24)
            stride: Sliding window stride (default 6)
            
        Returns:
            List of dicts with 'x' tensor, 'start_index', 'end_index'
        """
        if not records:
            return []
        
        # Extract features from all records
        all_values = []
        for record in records:
            values = self._extract_features_from_record(record, model_name)
            all_values.append(values)
        
        # Convert to tensor [T, F]
        features_tensor = torch.tensor(all_values, dtype=torch.float32)
        total_len = len(records)
        
        # Calculate number of windows
        if total_len < target_len:
            n_slices = 1
        else:
            n_slices = (total_len - target_len) // stride + 1
        
        windows = []
        for i in range(n_slices):
            start_idx = i * stride
            end_idx = start_idx + target_len
            
            # Extract window
            if end_idx <= total_len:
                frame = features_tensor[start_idx:end_idx]
            else:
                # Pad if needed
                frame = features_tensor[start_idx:]
                frame = self._pad_or_truncate_sequence(frame, target_len, padding_value=-100)
            
            last_index_window = min(end_idx - 1, total_len - 1)
            
            windows.append({
                'x': frame.unsqueeze(0).to(self.device),  # [1, T, F]
                'start_index': start_idx,
                'end_index': last_index_window,
                'window_index': i
            })

        logger.debug(f"Prepared {len(windows)} windows from {total_len} records (target_len={target_len}, stride={stride})")
        
        return windows
    
    def predict(self, record: Dict, model_name: str) -> Dict:
        """
        Make prediction for a single patient record (padded to window size)
        
        Args:
            record: Patient data dictionary
            model_name: 'mimic' or 'sepsiexp'
            
        Returns:
            Prediction dictionary with score, class, window info, and details
        """
        # Try to load model if not already loaded
        if model_name not in self.models:
            success = self.load_model(model_name)
            if not success:
                return {
                    'score': 0.0,
                    'class': 'unknown',
                    'probability': 0.0,
                    'prediction': 0,
                    'method': 'unavailable',
                    'error': f'Model {model_name} not available'
                }
        
        try:
            model_info = self.models[model_name]
            model = model_info['model']
            
            # Prepare input [1, 24, F] - single record padded to window size
            x = self.prepare_input(record, model_name)
            
            if x is None:
                return {
                    'score': 0.0,
                    'class': 'error',
                    'error': 'Failed to prepare input'
                }
            
            # Make prediction
            with torch.no_grad():
                logits = model(x).squeeze()
                prob = torch.sigmoid(logits).item()

            # Score is raw probability 0.0-1.0
            class_label = _classify_risk(prob)

            return {
                'score': round(prob, 4),
                'class': class_label,
                'probability': round(prob, 4),
                'prediction': 1 if prob >= 0.5 else 0,
                'method': f'pytorch_{model_name}',
                'confidence': round(abs(prob - 0.5) * 2, 4),
                'model_used': model_name,
                # Window info for single record (padded)
                'window': {
                    'target_len': 24,
                    'actual_records': 1,
                    'padding_applied': True
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'score': 0.0,
                'class': 'error',
                'probability': 0.0,
                'prediction': 0,
                'method': 'error',
                'error': str(e)
            }
    
    def predict_sequence(self, records: List[Dict], model_name: str,
                          target_len: int = 24, stride: int = 6) -> Dict:
        """
        Make predictions for a sequence of patient records using sliding window
        (Aligned with run_Prism.py prediction flow)
        
        Args:
            records: List of patient data dictionaries (ordered by time)
            model_name: 'mimic' or 'sepsiexp'
            target_len: Window size (default 24)
            stride: Sliding window stride (default 6)
            
        Returns:
            Dict with 'windows' (list of predictions) and 'summary' (aggregated stats)
        """
        # Try to load model if not already loaded
        if model_name not in self.models:
            success = self.load_model(model_name)
            if not success:
                return {
                    'windows': [],
                    'summary': {
                        'error': f'Model {model_name} not available',
                        'total_records': len(records),
                        'n_windows': 0
                    }
                }
        
        try:
            model_info = self.models[model_name]
            model = model_info['model']
            
            # Prepare windows
            windows = self.prepare_input_sequence(records, model_name, target_len, stride)
            
            results = []
            total_prob = 0.0
            positive_count = 0
            
            for window in windows:
                x = window['x']  # [1, target_len, F]
                
                with torch.no_grad():
                    logits = model(x).squeeze()
                    prob = torch.sigmoid(logits).item()

                class_label = _classify_risk(prob)
                prediction = 1 if prob >= 0.5 else 0

                total_prob += prob
                if prediction == 1:
                    positive_count += 1

                results.append({
                    'window_index': window['window_index'],
                    'start_index': window['start_index'],
                    'end_index': window['end_index'],
                    'score': round(prob, 4),
                    'class': class_label,
                    'probability': round(prob, 4),
                    'prediction': prediction,
                    'method': f'pytorch_{model_name}',
                    'model_used': model_name
                })
            
            # Calculate summary statistics
            n_windows = len(results)
            avg_prob = total_prob / n_windows if n_windows > 0 else 0.0

            # Capture the last (most recent) window as the valid window
            valid_window = results[-1] if results else None

            return {
                'windows': results,
                'valid_window': valid_window,
                'summary': {
                    'total_records': len(records),
                    'n_windows': n_windows,
                    'target_len': target_len,
                    'stride': stride,
                    'avg_score': round(avg_prob, 4),
                    'avg_probability': round(avg_prob, 4),
                    'at_risk_windows': positive_count,
                    'low_risk_windows': n_windows - positive_count,
                    'at_risk_ratio': round(positive_count / n_windows, 4) if n_windows > 0 else 0.0,
                    'final_class': _classify_risk(avg_prob),
                    'model_used': model_name
                }
            }
            
        except Exception as e:
            logger.error(f"Sequence prediction error: {str(e)}")
            return {
                'windows': [],
                'summary': {
                    'error': str(e),
                    'total_records': len(records),
                    'n_windows': 0
                }
            }
    
    def get_feature_importance(self, model_name: str) -> List[Dict]:
        """
        Get feature importance for a model
        Note: This is a placeholder - actual feature importance 
        would require SHAP or similar analysis
        
        Args:
            model_name: 'mimic' or 'sepsiexp'
            
        Returns:
            List of feature importance dictionaries
        """
        features = self.MIMIC_FEATURES if model_name == 'mimic' else self.SEPSIEXP_FEATURES
        
        # Placeholder - return empty list
        # In production, use SHAP or similar
        return [{'feature': f, 'importance': 0.0} for f in features]
    
    def model_info(self, model_name: str) -> Dict:
        """Get information about a loaded model"""
        if model_name not in self.models:
            return {'status': 'not_loaded', 'model': model_name}
        
        model_info = self.models[model_name]
        config = model_info['config']
        
        return {
            'status': 'loaded',
            'model': model_name,
            'input_dim': config['input_dim'],
            'hidden_dim': config['hidden_dim'],
            'num_layers': config['num_layers'],
            'pooling': config['pooling'],
            'architecture': 'LSTM Bidirectional',
            'features': len(self.MIMIC_FEATURES if model_name == 'mimic' else self.SEPSIEXP_FEATURES)
        }


# if __name__ == '__main__':


#     sepsi_cfg = {
#             'input_dim': 29,  # Number of features for SepsisExp (actual trained model size)
#             'hidden_dim': 4,  # Hidden dimension from checkpoint (weights show 16 = 4*4)
#             'num_layers': 4,
#             'pooling': 'max',
#             'dropout': 0.2,
#         }
    

#     classifier = LSTMClassifier(**sepsi_cfg)
#     # classifier.load_state_dict(torch.load('models/mimic_24-6_sequence_smote.pth', map_location='cpu', weights_only=False))
#     classifier.eval()
#     x = torch.randn(1, 1, 29)  # Example input
#     classifier(x)