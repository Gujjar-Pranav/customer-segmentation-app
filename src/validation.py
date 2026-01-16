# src/validation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: List[str]


def _get(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(f"Missing config key: '{path}' (failed at '{p}')")
        cur = cur[p]
    return cur


def validate_config(cfg: Dict[str, Any]) -> ValidationResult:
    errors: List[str] = []

    # required top keys
    required_paths = [
        "project.name",
        "project.seed",
        "paths.input_path",
        "paths.output_dir",
        "data.datetime_col",
        "data.id_col",
        "preprocessing.iqr_cap_cols",
        "clustering.k",
        "clustering.scaler",
        "clustering.kmeans.n_init",
        "clustering.kmeans.random_state",
        "run.save_tables",
        "run.save_plots",
        "run.show_plots",
        "plots.cluster_counts_png",
        "plots.heatmap_png",
        "plots.pca_html",
        "plots.pca_centroids_html",
        "personas.cluster_names",
        "personas.cluster_personas",
    ]

    for p in required_paths:
        try:
            _get(cfg, p)
        except Exception as e:
            errors.append(str(e))

    ok = len(errors) == 0
    return ValidationResult(ok=ok, errors=errors)


def require_columns(df: pd.DataFrame, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{where}] Missing columns: {missing}")
