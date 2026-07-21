"""ML Scorer Module for PRISM API."""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from src.api.model_loader import ModelLoader
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch models not available")
    TORCH_AVAILABLE = False


class MLScorer:
    """Machine Learning scoring engine for PRISM."""

    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.torch_loader = None

        if TORCH_AVAILABLE:
            try:
                self.torch_loader = ModelLoader(str(self.models_dir))
                self.torch_loader.load_all_models()
                logger.info("PyTorch models loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load PyTorch models: {e}")
                self.torch_loader = None

    def predict(self, data: Dict, db_name: str) -> Dict:
        if self.torch_loader:
            return self.torch_loader.predict(data, db_name)
        return self._heuristic_predict(data, db_name)

    def predict_batch(self, data_list: List[Dict], db_name: str) -> List[Dict]:
        return [self.predict(data, db_name) for data in data_list]

    def _heuristic_predict(self, data: Dict, db_name: str) -> Dict:
        return {
            'score': 0.0, 'class': 'unknown',
            'probability': 0.0, 'prediction': 0,
            'method': 'heuristic', 'error': 'Model not available'
        }