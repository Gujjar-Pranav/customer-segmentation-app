# src/pca_viz.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def compute_pca(
    X_scaled: np.ndarray,
    feature_names: List[str],
    labels: np.ndarray,
    cluster_name_map: Dict[int, str],
    kmeans_centers: np.ndarray | None = None,
    n_components: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Returns:
      pca_points_df: PCA coordinates per row
      explained_df: explained variance per component
      loadings_df: feature loadings per component
      centroids_df: PCA coords of centroids (if kmeans_centers provided)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    pca_points = pd.DataFrame(X_pca, columns=[f"PCA{i+1}" for i in range(n_components)])
    pca_points["Cluster"] = labels.astype(int)
    pca_points["Cluster_Name"] = pd.Series(labels).map(cluster_name_map).astype(str)

    explained = pd.DataFrame(
        {
            "PCA_Component": [f"PCA{i+1}" for i in range(n_components)],
            "Explained_Variance_%": (pca.explained_variance_ratio_ * 100).round(2),
        }
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PCA{i+1}" for i in range(n_components)],
    ).round(3)
    loadings = loadings.sort_values("PCA1", ascending=False)

    centroids_df = None
    if kmeans_centers is not None:
        centroids_pca = pca.transform(kmeans_centers)
        centroids_df = pd.DataFrame(centroids_pca, columns=[f"PCA{i+1}" for i in range(n_components)])
        centroids_df["Cluster"] = list(range(len(centroids_df)))
        centroids_df["Cluster_Name"] = centroids_df["Cluster"].map(cluster_name_map)

    return pca_points, explained, loadings, centroids_df


def save_pca_scatter_html(pca_df: pd.DataFrame, out_html: Path) -> None:
    fig = px.scatter_3d(
        pca_df,
        x="PCA1",
        y="PCA2",
        z="PCA3",
        color="Cluster_Name",
        opacity=0.7,
        title="3D PCA — Cluster Visualization",
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    logger.info("Saved PCA HTML: %s", out_html.resolve())


def save_pca_with_centroids_html(pca_df: pd.DataFrame, centroids_df: pd.DataFrame, out_html: Path) -> None:
    fig = px.scatter_3d(
        pca_df,
        x="PCA1",
        y="PCA2",
        z="PCA3",
        color="Cluster_Name",
        opacity=0.55,
        title="3D PCA + KMeans Centroids",
    )

    fig.add_trace(
        go.Scatter3d(
            x=centroids_df["PCA1"],
            y=centroids_df["PCA2"],
            z=centroids_df["PCA3"],
            mode="markers+text",
            text=centroids_df["Cluster_Name"],
            marker=dict(size=7, color="black"),
            name="Centroids",
        )
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    logger.info("Saved PCA+Centroids HTML: %s", out_html.resolve())
