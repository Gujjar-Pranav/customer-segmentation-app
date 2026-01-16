# src/categoricals.py
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_MARITAL_MAP: Dict[str, str] = {
    "Single": "Single",
    "Alone": "Single",
    "YOLO": "Single",
    "Absurd": "Single",
    "Married": "Couple",
    "Together": "Couple",
    "Divorced": "Previously Married",
    "Widow": "Previously Married",
}

DEFAULT_EDUCATION_MAP: Dict[str, str] = {
    "Basic": "Basic",
    "2n Cycle": "Basic",
    "Graduation": "Graduate",
    "Master": "Postgraduate",
    "PhD": "Postgraduate",
}


def add_marital_group(
    df: pd.DataFrame,
    source_col: str = "Marital_Status",
    out_col: str = "Marital_Group",
    mapping: Dict[str, str] = DEFAULT_MARITAL_MAP,
) -> pd.DataFrame:
    if source_col not in df.columns:
        raise KeyError(f"Missing column: {source_col}")
    df = df.copy()
    df[out_col] = df[source_col].map(mapping).fillna("Single")
    return df


def add_education_group(
    df: pd.DataFrame,
    source_col: str = "Education",
    out_col: str = "Education_Group",
    mapping: Dict[str, str] = DEFAULT_EDUCATION_MAP,
) -> pd.DataFrame:
    if source_col not in df.columns:
        raise KeyError(f"Missing column: {source_col}")
    df = df.copy()
    df[out_col] = df[source_col].map(mapping).fillna("Basic")
    return df


def cluster_distribution_table(
    df: pd.DataFrame,
    cluster_name_col: str,
    category_col: str,
    ordered_cols: List[str],
) -> pd.DataFrame:
    """
    Returns normalized distribution per cluster (rows sum to 1).
    """
    tab = pd.crosstab(df[cluster_name_col], df[category_col], normalize="index").round(3)
    # keep consistent column order (missing columns are allowed)
    tab = tab.reindex(columns=ordered_cols, fill_value=0)
    return tab
