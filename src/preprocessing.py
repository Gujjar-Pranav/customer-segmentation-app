# src/preprocessing.py
from __future__ import annotations

import logging
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def parse_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Datetime column missing: {col}")
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def drop_corrupted_ids(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col not in df.columns:
        return df
    before = len(df)
    df = df[df[id_col] != 0].copy()
    logger.info("Dropped corrupted IDs (%s==0): %d rows", id_col, before - len(df))
    return df


def drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    if constant_cols:
        df = df.drop(columns=constant_cols).copy()
        logger.info("Dropped constant columns: %s", constant_cols)
    return df, constant_cols


def impute_income(df: pd.DataFrame, income_col: str = "Income") -> pd.DataFrame:
    if income_col not in df.columns:
        return df
    df = df.copy()

    # treat 0 as missing
    df.loc[df[income_col] == 0, income_col] = np.nan

    median_val = df[income_col].median()
    df[income_col] = df[income_col].fillna(median_val)

    logger.info("Income imputed: median=%s", round(float(median_val), 2))
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Dropped duplicates: %d", before - len(df))
    return df


def iqr_cap(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            logger.warning("IQR cap skipped missing column: %s", col)
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)
    logger.info("Applied IQR capping to %d columns", len(list(cols)))
    return df
