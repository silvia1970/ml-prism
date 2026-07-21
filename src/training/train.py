import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Optional, Tuple
import logging
import random
import numpy as np

from src.models.lstm import LSTMClassifier
from src.training.data_prep import create_dataloader

logger = logging.getLogger(__name__)


def set_deterministic(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def balance_patients(data_by_patient: Dict, method: str = "undersample",
                      random_state: int = 42) -> Dict:
    """
    Balance patient data by upsampling or downsampling.

    Args:
        data_by_patient: Dict of patient data.
        method: 'undersample', 'oversample', or 'smote'.
        random_state: Random seed.

    Returns:
        Balanced dict of patient data.
    """
    if method != "undersample":
        return data_by_patient

    ids = list(data_by_patient.keys())
    labels = [data_by_patient[pid]["y"].item() for pid in ids]

    pos_ids = [ids[i] for i in range(len(ids)) if labels[i] == 1]
    neg_ids = [ids[i] for i in range(len(ids)) if labels[i] == 0]

    rng = np.random.RandomState(random_state)
    min_count = min(len(pos_ids), len(neg_ids))
    if min_count == 0:
        return data_by_patient

    neg_ids = list(rng.choice(neg_ids, min_count, replace=False))

    return {pid: data_by_patient[pid] for pid in pos_ids + neg_ids}


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_epoch(model, data_loader, criterion, device):
    """Evaluate on a data loader."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y.float())
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    return total_loss / len(data_loader), all_preds, all_labels


def train_and_evaluate_kfold(data_by_patient: Dict, device: torch.device,
                              n_splits: int = 5, batch_size: int = 64,
                              hidden_dim: int = 4, num_layers: int = 4,
                              dropout: float = 0.3, pooling: str = "mean",
                              lr: float = 5e-3, max_epochs: int = 50,
                              val_ratio: float = 0.2, patience: int = 5,
                              loss_fn: Optional[nn.Module] = None,
                              method: str = "undersample",
                              seed: int = 42,
                              weight_decay: float = 0) -> Dict:
    """
    K-fold cross-validation with early stopping.

    Args:
        data_by_patient: Dict from prepare_*_data functions.
        device: Torch device.
        n_splits: Number of CV folds.
        batch_size: Batch size.
        hidden_dim: LSTM hidden dimension.
        num_layers: Number of LSTM layers.
        dropout: Dropout rate.
        pooling: Pooling strategy.
        lr: Learning rate.
        max_epochs: Maximum training epochs.
        val_ratio: Validation split ratio.
        patience: Early stopping patience.
        loss_fn: Custom loss function.
        method: Class balancing method.
        seed: Random seed.
        weight_decay: L2 regularization.

    Returns:
        Dict with fold results and aggregated metrics.
    """
    set_deterministic(seed)

    patient_ids = list(data_by_patient.keys())
    patient_labels = [data_by_patient[pid]["y"].item() for pid in patient_ids]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, patient_labels)):
        logger.info(f"\n=== Fold {fold+1}/{n_splits} ===")

        train_ids = [patient_ids[i] for i in train_idx]
        test_ids = [patient_ids[i] for i in test_idx]

        X_train_full = torch.stack([data_by_patient[pid]["x"] for pid in train_ids])
        y_train_full = torch.stack([data_by_patient[pid]["y"] for pid in train_ids])
        X_test = torch.stack([data_by_patient[pid]["x"] for pid in test_ids])
        y_test = torch.stack([data_by_patient[pid]["y"] for pid in test_ids])

        full_dataset = TensorDataset(X_train_full, y_train_full)
        val_size = int(val_ratio * len(full_dataset))
        train_size = len(full_dataset) - val_size

        split_gen = torch.Generator().manual_seed(seed + fold)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=split_gen)

        train_data_by_patient = {i: {"x": x, "y": y} for i, (x, y) in enumerate(train_dataset)}
        train_data_by_patient = balance_patients(train_data_by_patient, method=method, random_state=seed + fold)

        train_loader = create_dataloader(train_data_by_patient, batch_size=batch_size, seed=seed + fold)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

        model = LSTMClassifier(
            input_dim=X_train_full.shape[2],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            dropout=dropout,
            pooling=pooling
        ).to(device)

        criterion = loss_fn or nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0]).to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(max_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, _, _ = evaluate_epoch(model, val_loader, criterion, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            model.load_state_dict(best_state)

        test_loss, test_preds, test_labels = evaluate_epoch(model, test_loader, criterion, device)

        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
        acc = accuracy_score(test_labels, test_preds)
        try:
            auc = roc_auc_score(test_labels, [p for p in test_preds])
        except ValueError:
            auc = 0.0
        f1 = f1_score(test_labels, test_preds, zero_division=0)
        rec = recall_score(test_labels, test_preds, zero_division=0)
        prec = precision_score(test_labels, test_preds, zero_division=0)

        fold_results.append({
            'fold': fold + 1,
            'train_loss': round(train_loss, 4),
            'val_loss': round(best_val_loss, 4),
            'test_loss': round(test_loss, 4),
            'accuracy': round(acc, 4),
            'auc': round(auc, 4),
            'f1': round(f1, 4),
            'recall': round(rec, 4),
            'precision': round(prec, 4),
        })
        logger.info(f"Fold {fold+1} - Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")

    avg_metrics = {
        'accuracy': sum(r['accuracy'] for r in fold_results) / len(fold_results),
        'auc': sum(r['auc'] for r in fold_results) / len(fold_results),
        'f1': sum(r['f1'] for r in fold_results) / len(fold_results),
        'recall': sum(r['recall'] for r in fold_results) / len(fold_results),
        'precision': sum(r['precision'] for r in fold_results) / len(fold_results),
    }

    return {'folds': fold_results, 'averages': avg_metrics}


def train_final_model(data_by_patient: Dict, device: torch.device,
                       model_config: Dict, batch_size: int = 64,
                       max_epochs: int = 50, lr: float = 5e-3,
                       seed: int = 42) -> LSTMClassifier:
    """
    Train final model on all data.

    Args:
        data_by_patient: Full patient data dict.
        device: Torch device.
        model_config: Model configuration dict.
        batch_size: Batch size.
        max_epochs: Maximum epochs.
        lr: Learning rate.
        seed: Random seed.

    Returns:
        Trained LSTMClassifier model.
    """
    set_deterministic(seed)

    X = torch.stack([v["x"] for v in data_by_patient.values()])
    y = torch.stack([v["y"] for v in data_by_patient.values()])

    model = LSTMClassifier(
        input_dim=X.shape[2],
        hidden_dim=model_config.get('hidden_dim', 4),
        num_layers=model_config.get('num_layers', 4),
        bidirectional=False,
        dropout=model_config.get('dropout', 0.3),
        pooling=model_config.get('pooling', 'mean')
    ).to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        loss = train_epoch(model, loader, criterion, optimizer, device)
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{max_epochs}, Loss: {loss:.4f}")

    return model