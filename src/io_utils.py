# src/io_utils.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd


logger = logging.getLogger(__name__)


def load_excel(path: Union[str, Path], sheet_name: Optional[Union[str, int]] = 0) -> pd.DataFrame:
    """
    Load an Excel file into a single DataFrame.

    Default behavior: read the first sheet (sheet_name=0) to avoid pandas returning dict-of-DFs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    df = pd.read_excel(path, sheet_name=sheet_name)

    # Safety: if something still returns a dict, take the first sheet.
    if isinstance(df, dict):
        first_key = next(iter(df.keys()))
        df = df[first_key]
        logger.warning("Excel returned multiple sheets; using first sheet: %s", first_key)

    logger.info("Loaded data: shape=%s from %s", df.shape, path.name)
    return df


def save_excel(df: pd.DataFrame, path: Union[str, Path], index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=index)
    logger.info("Saved: %s (shape=%s)", path.resolve(), df.shape)
