# src/package_artifacts.py
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Package model artifacts into a ZIP for deployment")
    p.add_argument("--run-dir", type=str, default="outputs/latest", help="Folder containing trained artifacts")
    p.add_argument("--out", type=str, default="outputs/deploy_artifacts.zip", help="ZIP output path")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_zip = Path(args.out)

    required_files = [
        "scaler.joblib",
        "kmeans.joblib",
        "artifacts_meta.joblib",
        "run.log",
        "config_used.yaml",
    ]

    missing = [f for f in required_files if not (run_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {run_dir}: {missing}")

    # create temp folder
    temp_dir = run_dir.parent / "_deploy_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # copy required files
    for f in required_files:
        shutil.copyfile(run_dir / f, temp_dir / f)

    # also include requirements.txt from project root if exists
    req = Path("requirements.txt")
    if req.exists():
        shutil.copyfile(req, temp_dir / "requirements.txt")

    # create zip
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(out_zip).replace(".zip", ""), "zip", temp_dir)

    # cleanup temp
    shutil.rmtree(temp_dir)

    print(f"\n✅ Deployment ZIP created at: {out_zip.resolve()}")


if __name__ == "__main__":
    main()
