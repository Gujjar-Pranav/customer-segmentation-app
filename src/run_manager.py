# src/run_manager.py
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


def create_run_dir(cfg: Dict[str, Any]) -> Tuple[Path, str]:
    base_out = Path(cfg["paths"]["output_dir"])
    use_ts = bool(cfg["paths"].get("use_timestamped_run_dir", True))
    prefix = str(cfg["paths"].get("run_dir_prefix", "runs/run_"))

    if use_ts:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_out / f"{prefix}{ts}"
    else:
        run_dir = base_out

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_dir.name


def copy_config_to_run_dir(config_path: str | Path, run_dir: Path) -> None:
    config_path = Path(config_path)
    dst = run_dir / "config_used.yaml"
    shutil.copyfile(config_path, dst)
