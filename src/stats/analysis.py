import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def run_eda(df: pd.DataFrame, output_dir: Optional[str] = None,
             verbose: bool = True) -> Dict:
    """
    Run Exploratory Data Analysis on a DataFrame.

    Args:
        df: Input DataFrame.
        output_dir: Directory to save plots.
        verbose: Whether to show plots.

    Returns:
        Dict with EDA summary statistics.
    """
    summary = {
        "shape": df.shape,
        "missing_values": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nunique": df.nunique().to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
    }

    logger.info(f"EDA - Shape: {df.shape}, Missing: {sum(df.isna().sum())}, Duplicates: {summary['duplicated_rows']}")

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

    numeric_df = df.select_dtypes(include=[np.number])

    if not numeric_df.empty:
        summary["descriptive_stats"] = {
            col: {
                "mean": float(numeric_df[col].mean()),
                "std": float(numeric_df[col].std()),
                "min": float(numeric_df[col].min()),
                "max": float(numeric_df[col].max()),
                "median": float(numeric_df[col].median()),
            }
            for col in numeric_df.columns
        }

        # Outlier detection
        outlier_counts = {}
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum()
            outlier_counts[col] = int(outliers)

        summary["outlier_counts"] = outlier_counts

    return summary


def compute_correlations(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Compute correlation matrix for numerical columns.

    Args:
        df: Input DataFrame.
        method: Correlation method ('pearson', 'spearman', 'kendall').

    Returns:
        Correlation matrix DataFrame.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr(method=method)


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot correlation heatmap."""
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(18, 14))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, annot_kws={"size": 6},
        cmap='coolwarm', fmt=".2f", linewidths=0.3, linecolor='gray'
    )
    plt.title("Correlation Heatmap (Upper Triangle)", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_feature_distributions(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot distributions of numerical features."""
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df.hist(figsize=(12, 10), bins=20, edgecolor='black')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()