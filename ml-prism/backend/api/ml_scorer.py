"""
ML Scorer Module for PRISM API
Integrates with trained models to generate predictions and explanations
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import pickle
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PyTorch models
try:
    from api.torch_models import ModelLoader
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch models not available, using heuristic predictions")
    TORCH_AVAILABLE = False


class MLScorer:
    """Machine Learning scoring engine for PRISM"""
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize ML scorer
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_names = {}
        self.feature_importance = {}
        
        # Initialize PyTorch model loader if available
        if TORCH_AVAILABLE:
            try:
                self.torch_loader = ModelLoader(str(self.models_dir))
                self.torch_loader.load_all_models()
                logger.info("PyTorch models loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load PyTorch models: {e}")
                self.torch_loader = None
        else:
            self.torch_loader = None
        
        # Load traditional models if available
        self._load_models()
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            # Try to load MIMIC model
            mimic_model_path = self.models_dir / 'mimic_model.pkl'
            if mimic_model_path.exists():
                with open(mimic_model_path, 'rb') as f:
                    self.models['mimic'] = pickle.load(f)
                logger.info("Loaded MIMIC sklearn model")
            
            # Try to load SepsisExp model
            sepsiexp_model_path = self.models_dir / 'sepsiexp_model.pkl'
            if sepsiexp_model_path.exists():
                with open(sepsiexp_model_path, 'rb') as f:
                    self.models['sepsiexp'] = pickle.load(f)
                logger.info("Loaded SepsisExp sklearn model")
            
            # Load feature names
            for db_name in ['mimic', 'sepsiexp']:
                feature_file = self.models_dir / f'{db_name}_features.json'
                if feature_file.exists():
                    with open(feature_file, 'r') as f:
                        self.feature_names[db_name] = json.load(f)
            
            # Load feature importance
            for db_name in ['mimic', 'sepsiexp']:
                importance_file = self.models_dir / f'{db_name}_importance.json'
                if importance_file.exists():
                    with open(importance_file, 'r') as f:
                        self.feature_importance[db_name] = json.load(f)
                        
        except Exception as e:
            logger.warning(f"Could not load traditional models: {str(e)}")
    
    def _prepare_features(self, db_name: str, record: Dict) -> np.ndarray:
        """
        Prepare feature vector from record
        
        Args:
            db_name: Database name
            record: Patient record
            
        Returns:
            Feature vector as numpy array
        """
        # Get expected feature names
        if db_name in self.feature_names:
            feature_list = self.feature_names[db_name]
        else:
            # Use record keys (excluding metadata)
            feature_list = [k for k in record.keys() 
                          if k not in ['sample_id', 'timestamp', 'submission_id']]
        
        # Extract features in correct order
        features = []
        for feature_name in feature_list:
            value = record.get(feature_name)
            
            # Handle missing values
            if value is None:
                # Use median or default value
                value = 0.0
            
            features.append(float(value))
        
        return np.array(features).reshape(1, -1)
    
    def _classify_score(self, score: float) -> str:
        """
        Convert probability score to risk class.

        Args:
            score: Probability (0.0 to 1.0)

        Returns:
            Risk class: 'low_risk' (<0.50), 'moderate_risk' ([0.50, 0.75)), 'high_risk' (≥0.75)
        """
        if score >= 0.75:
            return 'high_risk'
        elif score >= 0.50:
            return 'moderate_risk'
        return 'low_risk'
    
    def _get_explanations(self, db_name: str, record: Dict, score: float) -> Dict:
        """
        Generate explanation for prediction using feature importance
        
        Args:
            db_name: Database name
            record: Patient record
            score: Prediction score
            
        Returns:
            Explanation dictionary
        """
        explanations = {
            'method': 'feature_importance',
            'top_features': []
        }
        
        # Get feature importance if available
        if db_name not in self.feature_importance:
            return explanations
        
        importance = self.feature_importance[db_name]
        
        # Get top contributing features
        feature_impacts = []
        for feature_name, importance_value in importance.items():
            if feature_name in record and record[feature_name] is not None:
                # Determine impact direction based on value
                value = record[feature_name]
                impact = 'positive' if importance_value > 0 else 'negative'
                
                feature_impacts.append({
                    'name': feature_name,
                    'value': value,
                    'importance': abs(importance_value),
                    'impact': impact
                })
        
        # Sort by importance and take top 5
        feature_impacts.sort(key=lambda x: x['importance'], reverse=True)
        
        for fi in feature_impacts[:5]:
            explanations['top_features'].append({
                'name': fi['name'],
                'value': fi['value'],
                'impact': fi['impact']
            })
        
        return explanations
    
    def predict(self, db_name: str, record: Dict) -> Dict:
        """
        Generate prediction for a patient record
        
        Args:
            db_name: Database name (mimic or sepsiexp)
            record: Patient data record
            
        Returns:
            Dictionary with score, class, and explanations
        """
        warnings = []
        
        # Try PyTorch model first if available
        if TORCH_AVAILABLE and self.torch_loader is not None:
            try:
                prediction = self.torch_loader.predict(record, db_name)
                if prediction is not None:
                    logger.info(f"PyTorch prediction for {db_name}: score={prediction['score']}")
                    return prediction
                else:
                    warnings.append('PyTorch model not loaded for this dataset')
                    logger.warning(f"PyTorch model not available for {db_name}")
            except Exception as e:
                logger.error(f"PyTorch prediction error: {str(e)}", exc_info=True)
                warnings.append(f'PyTorch prediction failed: {str(e)}')
        
        # Check if legacy scikit-learn model is loaded
        if db_name not in self.models:
            logger.warning(f"No trained model available for {db_name}, using mock prediction")
            # Generate mock prediction based on heuristics
            return self._mock_predict(db_name, record)
        
        try:
            # Prepare features
            features = self._prepare_features(db_name, record)
            
            # Get prediction
            model = self.models[db_name]
            
            # Check if model has predict_proba
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0]
                score = float(proba[1])  # Probability of positive class
            else:
                # For models without predict_proba, use decision function or raw prediction
                prediction = model.predict(features)[0]
                score = float(prediction)
            
            # Ensure score is in [0, 1] range
            score = max(0.0, min(1.0, score))

            # Classify risk (0.0-1.0 scale)
            risk_class = self._classify_score(score)

            # Generate explanations
            explanations = self._get_explanations(db_name, record, score)

            return {
                'score': round(score, 4),
                'class': risk_class,
                'explanations': explanations,
                'warnings': warnings
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            # Fall back to mock prediction
            return self._mock_predict(db_name, record)
    
    def _mock_predict(self, db_name: str, record: Dict) -> Dict:
        """
        Generate mock prediction using simple heuristics
        This is used when no trained model is available
        
        Args:
            db_name: Database name
            record: Patient record
            
        Returns:
            Dictionary with score, class, and explanations
        """
        warnings = ['Using heuristic prediction. No trained model available.']
        
        # Simple heuristic based on key clinical indicators
        if db_name == 'mimic':
            risk_factors = []
            score_components = []
            
            # Age factor
            age = record.get('age', 50)
            if age > 65:
                risk_factors.append({'name': 'age', 'value': age, 'impact': 'positive'})
                score_components.append(0.15)
            
            # Vital signs
            heartrate = record.get('heartrate')
            if heartrate and (heartrate > 100 or heartrate < 60):
                risk_factors.append({'name': 'heartrate', 'value': heartrate, 'impact': 'positive'})
                score_components.append(0.10)
            
            meanbp = record.get('meanbp')
            if meanbp and meanbp < 65:
                risk_factors.append({'name': 'meanbp', 'value': meanbp, 'impact': 'positive'})
                score_components.append(0.15)
            
            # Lab values
            lactate = record.get('lactate')
            if lactate and lactate > 2.0:
                risk_factors.append({'name': 'lactate', 'value': lactate, 'impact': 'positive'})
                score_components.append(0.20)
            
            creatinine = record.get('creatinine')
            if creatinine and creatinine > 1.5:
                risk_factors.append({'name': 'creatinine', 'value': creatinine, 'impact': 'positive'})
                score_components.append(0.10)
            
            wbc = record.get('wbc')
            if wbc and (wbc > 12 or wbc < 4):
                risk_factors.append({'name': 'wbc', 'value': wbc, 'impact': 'positive'})
                score_components.append(0.10)
            
            # Calculate score
            score = 0.2 + sum(score_components)  # Base risk + components
            score = max(0.0, min(1.0, score))
            
        else:  # sepsiexp
            risk_factors = []
            score_components = []
            
            # Age
            age = record.get('age', 50)
            if age > 65:
                risk_factors.append({'name': 'age', 'value': age, 'impact': 'positive'})
                score_components.append(0.15)
            
            # Lactate (critical for sepsis)
            lactate = record.get('lactate')
            if lactate:
                if lactate > 4.0:
                    risk_factors.append({'name': 'lactate', 'value': lactate, 'impact': 'positive'})
                    score_components.append(0.30)
                elif lactate > 2.0:
                    risk_factors.append({'name': 'lactate', 'value': lactate, 'impact': 'positive'})
                    score_components.append(0.15)
            
            # Mean BP
            meanbp = record.get('meanbp')
            if meanbp and meanbp < 65:
                risk_factors.append({'name': 'meanbp', 'value': meanbp, 'impact': 'positive'})
                score_components.append(0.15)
            
            # PCT (procalcitonin - sepsis marker)
            pct = record.get('pct')
            if pct and pct > 2.0:
                risk_factors.append({'name': 'pct', 'value': pct, 'impact': 'positive'})
                score_components.append(0.20)
            
            # CRP (inflammation marker)
            crp = record.get('crp')
            if crp and crp > 100:
                risk_factors.append({'name': 'crp', 'value': crp, 'impact': 'positive'})
                score_components.append(0.10)
            
            score = 0.2 + sum(score_components)
            score = max(0.0, min(1.0, score))
        
        # Classify risk using 0.0-1.0 scale
        risk_class = self._classify_score(score)
        
        return {
            'score': round(score, 4),
            'class': risk_class,
            'explanations': {
                'top_features': risk_factors,
                'method': 'heuristic'
            },
            'warnings': warnings
        }
