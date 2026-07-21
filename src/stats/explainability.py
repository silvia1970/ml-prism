import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def run_shap(model, X: np.ndarray, feature_names: Optional[List[str]] = None,
              max_samples: int = 100, output_dir: Optional[str] = None) -> Dict:
    """
    Run SHAP analysis for model explainability.

    Args:
        model: Trained model (sklearn-compatible for TreeExplainer).
        X: Input features array.
        feature_names: List of feature names.
        max_samples: Maximum samples for SHAP computation.
        output_dir: Directory to save SHAP plots.

    Returns:
        Dict with SHAP values and summary.
    """
    if X.shape[0] > max_samples:
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    # Use TreeExplainer for tree-based models, KernelExplainer otherwise
    if hasattr(model, 'feature_importances_'):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, X_sample[:min(50, len(X_sample))])

    shap_values = explainer.shap_values(X_sample)

    result = {
        "expected_value": float(explainer.expected_value) if isinstance(explainer.expected_value, (float, np.floating)) else explainer.expected_value,
        "n_samples": X_sample.shape[0],
    }

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=150, bbox_inches='tight')
        plt.close()

    return result


def run_lime(model, X: np.ndarray, instance_idx: int = 0,
              feature_names: Optional[List[str]] = None,
              class_names: Optional[List[str]] = None,
              num_features: int = 10) -> Dict:
    """
    Run LIME analysis for a single instance.

    Args:
        model: Trained model with predict_proba.
        X: Input features array.
        instance_idx: Index of instance to explain.
        feature_names: List of feature names.
        class_names: List of class names.
        num_features: Number of top features to return.

    Returns:
        Dict with LIME explanation features and weights.
    """
    X_df = pd.DataFrame(X, columns=feature_names)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_df.values, feature_names=feature_names, class_names=class_names,
        discretize_continuous=True
    )

    exp = explainer.explain_instance(X_df.iloc[instance_idx].values, model.predict_proba, num_features=num_features)

    features = []
    for name, weight in exp.as_list():
        features.append({"feature": name, "weight": weight})

    return {
        "instance_idx": instance_idx,
        "features": features,
        "predicted_class": exp.predictions[0],
    }