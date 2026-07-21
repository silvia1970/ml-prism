import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, adjusted_rand_score,
    adjusted_mutual_info_score, confusion_matrix
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def cluster_accuracy(y_true, y_pred):
    """
    Compute clustering accuracy with optimal label matching.

    Args:
        y_true: True labels.
        y_pred: Predicted cluster labels.

    Returns:
        Tuple of (accuracy, confusion_matrix, matched_confusion_matrix).
    """
    classes = np.unique(y_true)
    n_classes = len(classes)
    pred_labels = np.unique(y_pred)

    n = max(n_classes, len(pred_labels))
    cm = confusion_matrix(y_true, y_pred, labels=list(classes) + list(pred_labels[n_classes:]))[:n_classes, :n]

    cost_matrix = -cm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_cm = cm[row_ind, :][:, np.argsort(col_ind)]

    accuracy = np.trace(matched_cm) / cm.sum()
    return accuracy, cm, matched_cm


def find_optimal_clusters(X, max_k: int = 10) -> int:
    """Find optimal number of clusters using elbow method."""
    inertias = []
    K_range = range(2, max_k + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    kn = KneeLocator(list(K_range), inertias, curve="convex", direction="decreasing")
    return kn.elbow if kn.elbow else min(K_range)


def run_clustering(X: np.ndarray, y: Optional[np.ndarray] = None,
                    n_clusters: Optional[int] = None, method: str = "kmeans",
                    save_dir: Optional[str] = None, verbose: bool = True) -> Dict:
    """
    Run clustering analysis with evaluation metrics.

    Args:
        X: Feature matrix.
        y: True labels (optional, for supervised metrics).
        n_clusters: Number of clusters (auto-detected if None).
        method: Clustering method ('kmeans', 'dbscan', 'agglomerative', 'gmm').
        save_dir: Directory to save plots and results.
        verbose: Whether to show plots.

    Returns:
        Dict with clustering results and metrics.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if n_clusters is None:
        n_clusters = find_optimal_clusters(X)

    result = {"n_clusters": n_clusters, "method": method}

    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(X)
        result["silhouette"] = silhouette_score(X, labels)
        result["davies_bouldin"] = davies_bouldin_score(X, labels)

    elif method == "dbscan":
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X)
        n_clusters_found = len(set(labels) - {-1})
        result["n_clusters"] = n_clusters_found
        if n_clusters_found > 1:
            result["silhouette"] = silhouette_score(X, labels)

    elif method == "agglomerative":
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(X)
        result["silhouette"] = silhouette_score(X, labels)

    elif method == "gmm":
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gmm.fit_predict(X)
        result["silhouette"] = silhouette_score(X, labels)
        result["bic"] = gmm.bic(X)

    if y is not None:
        result["ari"] = adjusted_rand_score(y, labels)
        result["ami"] = adjusted_mutual_info_score(y, labels)
        acc, cm, matched_cm = cluster_accuracy(y, labels)
        result["accuracy"] = acc

    logger.info(f"Clustering ({method}, k={n_clusters}): "
                f"Silhouette={result.get('silhouette', 'N/A')}, "
                f"ARI={result.get('ari', 'N/A')}")

    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'{method.title()} Clustering (k={n_clusters})')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)

    if y is not None:
        plt.subplot(1, 2, 2)
        for label in np.unique(y):
            mask = y == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Class {label}', s=50)
        plt.title('True Labels')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.grid(True)

    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{method}_k{n_clusters}.png"), dpi=150)
    if verbose:
        plt.show()
    else:
        plt.close()

    if save_dir:
        with open(os.path.join(save_dir, f"{method}_k{n_clusters}.json"), "w") as f:
            json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                       for k, v in result.items()}, f, indent=4)

    return result