# src/latest_manager.py
from __future__ import annotations

import shutil
from pathlib import Path


def update_latest(run_dir: Path, latest_dir: Path) -> None:
    """
    Copies the full run_dir into outputs/latest (overwrites previous latest).
    Works on Windows + Mac + Linux safely (no symlink dependency).
    """
    if latest_dir.exists():
        shutil.rmtree(latest_dir)

    shutil.copytree(run_dir, latest_dir)
