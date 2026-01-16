# src/evaluation.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KMeansQuality:
    silhouette_mean: float
    silhouette_min: float
    negative_silhouette_pct: float
    cluster_share: Dict[int, float]


def evaluate_partition(X_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    return {
        "Silhouette_Score": float(silhouette_score(X_scaled, labels)),
        "Davies_Bouldin_Index": float(davies_bouldin_score(X_scaled, labels)),
        "Calinski_Harabasz_Index": float(calinski_harabasz_score(X_scaled, labels)),
    }


def evaluate_kmeans_quality(X_scaled: np.ndarray, labels: np.ndarray) -> KMeansQuality:
    sil_mean = float(silhouette_score(X_scaled, labels))
    sil_samp = silhouette_samples(X_scaled, labels)
    neg_pct = float((sil_samp < 0).mean() * 100.0)

    props = (
        pd.Series(labels)
        .value_counts(normalize=True)
        .sort_index()
        .to_dict()
    )

    return KMeansQuality(
        silhouette_mean=sil_mean,
        silhouette_min=float(sil_samp.min()),
        negative_silhouette_pct=neg_pct,
        cluster_share={int(k): float(v) for k, v in props.items()},
    )


def compare_models(X_scaled: np.ndarray, k: int) -> pd.DataFrame:
    """
    Compare KMeans result with a few alternative clusterers using the same X_scaled.
    (KMeans scores should be added by caller if needed; this function focuses on alternatives.)
    """
    results: List[Dict[str, object]] = []

    # Hierarchical
    for linkage in ["ward", "average", "complete"]:
        try:
            agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = agg.fit_predict(X_scaled)
            scores = evaluate_partition(X_scaled, labels)
            results.append({"Model": "Hierarchical", "Config": linkage, **scores})
        except Exception as e:
            logger.warning("Hierarchical failed (%s): %s", linkage, e)

    # GMM
    for cov in ["full", "tied", "diag"]:
        try:
            gmm = GaussianMixture(n_components=k, covariance_type=cov, random_state=42)
            labels = gmm.fit_predict(X_scaled)
            scores = evaluate_partition(X_scaled, labels)
            results.append({"Model": "GMM", "Config": cov, **scores})
        except Exception as e:
            logger.warning("GMM failed (%s): %s", cov, e)

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("Silhouette_Score", ascending=False).reset_index(drop=True)
    return df
