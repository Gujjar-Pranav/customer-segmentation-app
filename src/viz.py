# src/viz.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_or_show(fig, path: Path, save: bool, show: bool) -> None:
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot: %s", path.resolve())
    if show:
        plt.show()
    plt.close(fig)


def plot_cluster_counts(df: pd.DataFrame, out_path: Path, save: bool, show: bool) -> None:
    counts = df["Cluster_Name"].value_counts().sort_index()

    fig = plt.figure(figsize=(9, 4))
    ax = plt.gca()
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Customers per Cluster")
    ax.set_ylabel("Customers")
    ax.set_xlabel("")
    plt.xticks(rotation=10)

    save_or_show(fig, out_path, save, show)


def plot_normalized_heatmap(
    df: pd.DataFrame,
    feature_cols: List[str],
    out_path: Path,
    save: bool,
    show: bool,
) -> None:
    profile = df.groupby("Cluster_Name")[feature_cols].mean()
    profile_norm = (profile - profile.mean()) / profile.std()

    fig = plt.figure(figsize=(14, 6))
    ax = plt.gca()
    im = ax.imshow(profile_norm.values, aspect="auto")
    ax.set_title("Cluster Feature Profile (Normalized)")
    ax.set_yticks(range(len(profile_norm.index)))
    ax.set_yticklabels(profile_norm.index)
    ax.set_xticks(range(len(profile_norm.columns)))
    ax.set_xticklabels(profile_norm.columns, rotation=45, ha="right")

    # annotate values
    for i in range(profile_norm.shape[0]):
        for j in range(profile_norm.shape[1]):
            ax.text(j, i, f"{profile_norm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()

    save_or_show(fig, out_path, save, show)
