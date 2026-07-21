import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, recall_score, precision_score,
    matthews_corrcoef, brier_score_loss, confusion_matrix, classification_report
)
import logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true: List[int], y_pred: List[int],
                     probs: List[float] = None) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        probs: Predicted probabilities (optional).

    Returns:
        Dict of metric names and values.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }

    if probs is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, probs)
        except ValueError:
            metrics['auc'] = 0.0
        metrics['brier_score'] = brier_score_loss(y_true, probs)

    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def evaluate_model(model, data_loader, device: torch.device,
                    criterion=None) -> Dict:
    """
    Evaluate a model on a data loader.

    Args:
        model: Trained model.
        data_loader: Data loader.
        device: Torch device.
        criterion: Loss function.

    Returns:
        Dict with metrics and predictions.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)

            if criterion is not None:
                loss = criterion(logits, batch_y.float())
                total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['avg_loss'] = total_loss / len(data_loader) if data_loader else 0.0

    logger.info(f"Evaluation - Accuracy: {metrics['accuracy']:.4f}, "
                f"F1: {metrics['f1']:.4f}, AUC: {metrics.get('auc', 0):.4f}")

    return {
        'metrics': metrics,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
    }


def evaluate_predictions(y_true, y_pred, labels=None, probs=None):
    """Print detailed classification report."""
    print("\n=== Classification Report ===")
    target_names = labels if labels else ['Class 0', 'Class 1']
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    metrics = compute_metrics(y_true, y_pred, probs)
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"{k}: {v:.4f}")

    return metrics