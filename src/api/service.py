# src/api/service.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import pandas as pd

from src.artifacts import load_model_artifacts
from src.inference_features import build_features_for_inference


@lru_cache(maxsize=4)
def get_bundle(artifacts_dir_str: str) -> Dict[str, Any]:
    """
    Cache artifacts in memory.
    Keyed by artifacts_dir path string so you can support multiple models if needed.
    """
    artifacts_dir = Path(artifacts_dir_str)
    return load_model_artifacts(artifacts_dir)


def model_info(artifacts_dir: Path) -> Dict[str, Any]:
    bundle = get_bundle(str(artifacts_dir))
    return {
        "final_features": bundle.get("final_features"),
        "meta": bundle.get("meta", {}),
    }


def predict(
    payload: Dict[str, Any],
    artifacts_dir: Path,
    mode: Literal["raw", "engineered"] = "raw",
) -> Tuple[int, str | None]:
    bundle = get_bundle(str(artifacts_dir))

    scaler = bundle["scaler"]
    model = bundle["model"]
    final_features = bundle["final_features"]

    if mode == "raw":
        raw_df = pd.DataFrame([payload])
        X = build_features_for_inference(raw_df)
    else:
        X = pd.DataFrame([payload])

    missing = [c for c in final_features if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = X[final_features].copy()
    X_scaled = scaler.transform(X)

    cluster = int(model.predict(X_scaled)[0])

    cluster_name_map = bundle.get("meta", {}).get("cluster_names", {}) or {}
    cluster_name = cluster_name_map.get(str(cluster)) or cluster_name_map.get(cluster)

    return cluster, cluster_name


def predict_batch(
    df_in: pd.DataFrame,
    artifacts_dir: Path,
    mode: Literal["raw", "engineered"] = "raw",
) -> pd.DataFrame:
    bundle = get_bundle(str(artifacts_dir))

    scaler = bundle["scaler"]
    model = bundle["model"]
    final_features = bundle["final_features"]

    if mode == "raw":
        X = build_features_for_inference(df_in)
    else:
        X = df_in.copy()

    missing = [c for c in final_features if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = X[final_features].copy()
    X_scaled = scaler.transform(X)

    clusters = model.predict(X_scaled).astype(int)

    cluster_name_map = bundle.get("meta", {}).get("cluster_names", {}) or {}
    names = [cluster_name_map.get(str(int(c))) or cluster_name_map.get(int(c)) for c in clusters]

    df_out = df_in.copy()
    df_out["predicted_cluster"] = clusters
    df_out["predicted_cluster_name"] = names
    return df_out
