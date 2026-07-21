import os
import glob
import json
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def from_tsv_to_csv(path_to_dataset: str = 'datasets/SepsisExp') -> None:
    """
    Convert all .tsv files in the specified directory to .csv files.

    Args:
        path_to_dataset: Path to directory containing .tsv files.
    """
    for partition in os.listdir(path_to_dataset):
        if partition.endswith(".tsv"):
            tsv_path = os.path.join(path_to_dataset, partition)
            csv_path = os.path.join(
                path_to_dataset,
                os.path.splitext(partition)[0] + ".csv"
            )
            with open(tsv_path, 'r') as tsv_file:
                with open(csv_path, 'w') as csv_file:
                    for line in tsv_file:
                        csv_file.write(re.sub("\t", ",", line))
    logger.info(f"Converted TSV files in {path_to_dataset} to CSV")


def analyze_csvs(folder_path: str, verbose: bool = True) -> tuple:
    """
    Analyze CSV files in a folder: merge, describe, detect outliers.

    Args:
        folder_path: Path to directory containing CSV files.
        verbose: If True, print details and show plots.

    Returns:
        Tuple of (data, merged_df, outliers_list)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if verbose:
        sns.set_style(style="whitegrid")

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if verbose:
        print(f"Found {len(csv_files)} CSV files")

    dfs = [pd.read_csv(f) for f in csv_files]
    merged_df = pd.concat(dfs, ignore_index=True)

    drop_cols = [c for c in ['id', 'sepsis', 'severity', 'timestep', 'age'] if c in merged_df.columns]
    data = merged_df.drop(columns=drop_cols)

    if verbose:
        print(f"Shape: {data.shape}")
        print(f"\nMissing values:\n{data.isna().sum()[data.isna().sum() > 0]}")
        print(f"\nDuplicated rows: {data.duplicated().sum()}")
        print(f"\nDescriptive stats:\n{data.describe()}")

    outliers_list = []
    for col in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((data[col] < lower) | (data[col] > upper)).sum()
        outliers_list.append(outliers)
        if verbose:
            print(f"Outliers in '{col}': {outliers}")

    return data, merged_df, outliers_list


def is_standardized(df: pd.DataFrame, tol_mean: float = 1.0, tol_std: float = 2.0) -> bool:
    """Check if data is already standardized (mean ~0, std ~1)."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return False
    means = numeric_df.mean()
    stds = numeric_df.std()
    return (means.abs() < tol_mean).all() and ((stds - 1).abs() < tol_std).all()


def pad_or_truncate_sequence(seq: torch.Tensor, target_length: int,
                              padding_value: float = -100) -> torch.Tensor:
    """Pad or truncate a sequence tensor to target length."""
    if len(seq) < target_length:
        pad_len = target_length - len(seq)
        padding = torch.full((pad_len, *seq.shape[1:]), padding_value)
        seq = torch.cat([padding, seq], dim=0)
    if len(seq) > target_length:
        seq = seq[-target_length:]
    return seq


def normalize_single_patient_mimic(single_patient_df: pd.DataFrame,
                                    features_selected: List[str],
                                    path_to_json: Optional[str] = None) -> pd.DataFrame:
    """
    Normalize MIMIC patient data using stored statistics.

    Args:
        single_patient_df: DataFrame with patient time-series data.
        features_selected: List of feature column names to normalize.
        path_to_json: Path to MIMIC normalization stats JSON file.

    Returns:
        Normalized DataFrame.
    """
    if path_to_json is None:
        path_to_json = Path(__file__).parent.parent.parent / \
                       "datasets/MIMIC/carry_forward/mean/backend/temporal_signature_info_split_0 (1).json"

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
                                        path_to_json: Optional[str] = None) -> pd.DataFrame:
    """
    Normalize SepsisExp patient data using stored statistics.

    Args:
        single_patient_df: DataFrame with patient time-series data.
        path_to_json: Path to SepsisExp normalization stats JSON file.

    Returns:
        Normalized DataFrame (preserves metadata columns).
    """
    if path_to_json is None:
        path_to_json = Path(__file__).parent.parent.parent / \
                       "datasets/SepsisExp/original_sepsisexp/sepsisexp_normalization_global.json"

    with open(path_to_json) as f:
        sepsisexp_dict = json.load(f)

    normalize_dict = {}
    meta_cols = ['id', 'sepsis', 'severity', 'timestep']

    for col in single_patient_df.columns:
        if col not in meta_cols and col in sepsisexp_dict:
            mean_val = sepsisexp_dict[col]["mean"]
            std_val = sepsisexp_dict[col]["std"]
            if std_val > 0:
                normalize_dict[col] = (single_patient_df[col] - mean_val) / std_val
            else:
                normalize_dict[col] = single_patient_df[col] - mean_val

    normalize_df = pd.DataFrame(normalize_dict)

    for col in meta_cols:
        if col in single_patient_df.columns:
            normalize_df[col] = single_patient_df[col].values

    return normalize_df