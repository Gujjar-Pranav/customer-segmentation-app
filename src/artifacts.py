# src/artifacts.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib

logger = logging.getLogger(__name__)


def save_model_artifacts(
    run_dir: Path,
    scaler: Any,
    model: Any,
    final_features: List[str],
    meta: Dict[str, Any],
) -> Tuple[Path, Path]:
    """
    Saves:
      - scaler.joblib
      - kmeans.joblib
      - artifacts_meta.joblib (features + metadata)
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = run_dir / "scaler.joblib"
    model_path = run_dir / "kmeans.joblib"
    meta_path = run_dir / "artifacts_meta.joblib"

    joblib.dump(scaler, scaler_path)
    joblib.dump(model, model_path)
    joblib.dump({"final_features": final_features, "meta": meta}, meta_path)

    logger.info("Saved scaler: %s", scaler_path.resolve())
    logger.info("Saved model: %s", model_path.resolve())
    logger.info("Saved metadata: %s", meta_path.resolve())

    return scaler_path, model_path


def load_model_artifacts(run_dir: Path) -> Dict[str, Any]:
    scaler = joblib.load(run_dir / "scaler.joblib")
    model = joblib.load(run_dir / "kmeans.joblib")
    meta = joblib.load(run_dir / "artifacts_meta.joblib")
    return {"scaler": scaler, "model": model, **meta}
