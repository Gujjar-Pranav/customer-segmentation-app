# src/modeling.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

ScalerName = Literal["StandardScaler", "RobustScaler"]


@dataclass(frozen=True)
class ClusterArtifacts:
    scaler_name: ScalerName
    k: int
    scaler: object
    model: KMeans
    X_scaled: np.ndarray
    labels: np.ndarray


def get_scaler(name: ScalerName):
    if name == "StandardScaler":
        return StandardScaler()
    if name == "RobustScaler":
        return RobustScaler()
    raise ValueError(f"Unknown scaler: {name}")


def fit_kmeans(
    X: pd.DataFrame,
    scaler_name: ScalerName,
    k: int,
    n_init: int = 10,
    random_state: int = 42,
) -> ClusterArtifacts:
    scaler = get_scaler(scaler_name)
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X_scaled)

    props = pd.Series(labels).value_counts(normalize=True).sort_index()
    logger.info(
        "KMeans fit complete: k=%d scaler=%s cluster_share=%s",
        k,
        scaler_name,
        props.round(3).to_dict(),
    )

    return ClusterArtifacts(
        scaler_name=scaler_name,
        k=k,
        scaler=scaler,
        model=km,
        X_scaled=X_scaled,
        labels=labels,
    )


def attach_clusters(df: pd.DataFrame, labels: np.ndarray, col: str = "Cluster") -> pd.DataFrame:
    df = df.copy()
    df[col] = labels.astype(int)
    return df
