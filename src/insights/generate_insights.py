from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]  # project root
LATEST_DIR = ROOT / "outputs" / "latest"
OUT_DIR = LATEST_DIR / "insights"

# Inputs produced by your notebook/pipeline
PERSONAS_XLSX = LATEST_DIR / "cluster_personas.xlsx"
CLUSTER_SUMMARY_XLSX = LATEST_DIR / "cluster_summary.xlsx"
CLV_SUMMARY_XLSX = LATEST_DIR / "clv_summary.xlsx"
CHANNEL_STRATEGY_XLSX = LATEST_DIR / "channel_strategy.xlsx"


def _read_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_excel(path)


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Anything you already export to outputs/latest
    xlsx_files = sorted(LATEST_DIR.glob("*.xlsx"))
    png_files = sorted(LATEST_DIR.glob("*.png"))

    insights_tables: dict[str, list[dict]] = {}
    for f in xlsx_files:
        df = _read_excel(f)
        insights_tables[f.stem] = _df_to_records(df)

    # For images, we just publish URLs later (frontend will use them)
    # Here we only list filenames so API can serve them.
    insights_images = [p.name for p in png_files]

    payload = {
        "generated_from": str(LATEST_DIR),
        "tables": insights_tables,      # includes rfm, promo_roi, edu group, marital group, etc (whatever exists)
        "images": insights_images,      # cluster_feature_heatmap.png, cluster_counts.png, etc
    }

    out_path = OUT_DIR / "insights.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"✅ Wrote {out_path}")


if __name__ == "__main__":
    main()