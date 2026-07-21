import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from src.utils.preprocessing import (
    normalize_single_patient_mimic,
    normalize_single_patient_sepsisexp,
    is_standardized,
    pad_or_truncate_sequence,
)
from src.utils.field_mappings import MIMIC_FEATURES, SEPSIEXP_FEATURES

logger = logging.getLogger(__name__)


def load_and_prepare_data(folder_path: str) -> tuple:
    """
    Load and prepare SepsisExp data from TSV files.

    Args:
        folder_path: Path to dataset directory with train/val/test subdirs.

    Returns:
        Tuple of (all_df, train_df, val_df, test_df) with timestep labels.
    """
    df_train_X = pd.read_csv(os.path.join(folder_path, 'train', 'X_train.tsv'), sep='\t')
    df_val_X = pd.read_csv(os.path.join(folder_path, 'val', 'X_val.tsv'), sep='\t')
    df_test_X = pd.read_csv(os.path.join(folder_path, 'test', 'X_test.tsv'), sep='\t')

    df_train_y = pd.read_csv(os.path.join(folder_path, 'train', 'y_train.tsv'), sep='\t', index_col=0)
    df_val_y = pd.read_csv(os.path.join(folder_path, 'val', 'y_val.tsv'), sep='\t', index_col=0)
    df_test_y = pd.read_csv(os.path.join(folder_path, 'test', 'y_test.tsv'), sep='\t', index_col=0)

    # Realign indices
    df_val_X = df_val_X.copy()
    df_val_X['Index'] = df_val_X['Index'] + df_train_X['Index'].iloc[-1] + 1

    df_test_X = df_test_X.copy()
    df_test_X['Index'] = df_test_X['Index'] + df_val_X['Index'].iloc[-1] + 1

    all_df = pd.concat([df_train_X, df_val_X, df_test_X])
    all_labels = pd.concat([df_train_y, df_val_y, df_test_y]).rename(columns={'0': 'sepsis'})
    all_labels['Index'] = np.arange(0, len(all_df['Index'].unique()), 1, dtype=int)
    all_labels = all_labels[['Index', 'sepsis']]

    def add_timestep(df, labels):
        group_list = []
        for i, group in df.groupby(by='Index'):
            group = group.copy()
            group['timestep'] = np.arange(-49, 0, 1, dtype=int)
            group_list.append(group)
        df_timestep = pd.concat(group_list, axis=0)
        return df_timestep.merge(labels, on='Index', how='left')

    all_df_timestep = add_timestep(all_df, all_labels)
    train_df_timestep = add_timestep(df_train_X, df_train_y.rename(columns={'0': 'sepsis'}).reset_index().rename(columns={'index': 'Index'}))
    val_df_timestep = add_timestep(df_val_X, df_val_y.rename(columns={'0': 'sepsis'}).reset_index().rename(columns={'index': 'Index'}))
    test_df_timestep = add_timestep(df_test_X, df_test_y.rename(columns={'0': 'sepsis'}).reset_index().rename(columns={'index': 'Index'}))

    return all_df_timestep, train_df_timestep, val_df_timestep, test_df_timestep


def prepare_sepsisexp_data(hour_df: pd.DataFrame, target_len: int = 24,
                            padding_value: float = -100, stride: int = 6) -> Dict:
    """
    Prepare SepsisExp data for model training/inference with sliding windows.

    Args:
        hour_df: DataFrame with patient time-series data.
        target_len: Window size (default 24).
        padding_value: Value for padding short sequences.
        stride: Sliding window stride.

    Returns:
        Dict mapping patient index to data dict with 'x', 'y', indices.
    """
    data_by_patient = {}

    y = hour_df["sepsis"].iloc[0] if "sepsis" in hour_df.columns else 0
    y_tensor = torch.tensor(y, dtype=torch.long)

    sepsis_onset_index = -1
    if y == 1 and "severity" in hour_df.columns:
        sepsis_onset_index = int(np.argmax(hour_df['severity']))
    else:
        sepsis_onset_index = 0

    total_len = len(hour_df)

    if total_len < target_len:
        n_slices = 1
    else:
        n_slices = (total_len - target_len) // stride + 1

    drop_cols = ["id", "sepsis", "timestep", "severity"]
    feature_cols = [c for c in hour_df.columns if c not in drop_cols]
    features = hour_df[feature_cols].values

    for i in range(n_slices):
        start_idx = i * stride
        end_idx = start_idx + target_len
        frame = features[start_idx:end_idx]

        frame_tensor = pad_or_truncate_sequence(
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


def prepare_mimic_data(mimic_df: pd.DataFrame, target_len: int = 24,
                        stride: int = 6, padding_value: float = -100) -> Dict:
    """
    Prepare MIMIC data for model training/inference with sliding windows.

    Args:
        mimic_df: DataFrame with MIMIC patient time-series data.
        target_len: Window size (default 24).
        stride: Sliding window stride.
        padding_value: Value for padding short sequences.

    Returns:
        Dict mapping patient index to data dict with 'x', 'y', indices.
    """
    features_selected = MIMIC_FEATURES
    available_features = [f for f in features_selected if f in mimic_df.columns]

    if not available_features:
        raise ValueError(f"No MIMIC features found. Expected: {features_selected[:5]}...")

    if not is_standardized(mimic_df[available_features]):
        normalize_df = normalize_single_patient_mimic(mimic_df, available_features)
        for col in ['Index', 'timestep', 'sepsis', 'icustay_id']:
            if col in mimic_df.columns:
                normalize_df[col] = mimic_df[col].values
        mimic_df = normalize_df.copy()

    data_by_patient = {}

    y = mimic_df["sepsis"].iloc[0] if "sepsis" in mimic_df.columns else 0
    y_tensor = torch.tensor(y, dtype=torch.long)
    sepsis_onset_index = mimic_df.shape[0]

    total_len = len(mimic_df)

    if total_len < target_len:
        n_slices = 1
    else:
        n_slices = (total_len - target_len) // stride + 1

    drop_cols = ["Index", "sepsis", "timestep", "icustay_id", "label", "chart_time"]
    feature_cols = [c for c in mimic_df.columns if c not in drop_cols and c in available_features]
    features = mimic_df[feature_cols].values

    for i in range(n_slices):
        start_idx = i * stride
        end_idx = start_idx + target_len
        frame = features[start_idx:end_idx]

        frame_tensor = pad_or_truncate_sequence(
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


def create_dataloader(data_by_patient: Dict, batch_size: int = 64,
                       shuffle: bool = True, seed: int = 42) -> DataLoader:
    """
    Create a DataLoader from prepared patient data.

    Args:
        data_by_patient: Dict from prepare_*_data functions.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        seed: Random seed.

    Returns:
        PyTorch DataLoader.
    """
    gen = torch.Generator().manual_seed(seed)
    X = torch.stack([v["x"] for v in data_by_patient.values()])
    y = torch.stack([v["y"] for v in data_by_patient.values()])

    return DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen
    )