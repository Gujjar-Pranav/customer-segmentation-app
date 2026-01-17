from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

BASE = Path("outputs/latest")
OUT = BASE / "insights.json"


def read_xlsx(name: str):
    p = BASE / name
    if not p.exists():
        return None
    df = pd.read_excel(p)
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def main():
    payload = {
        "status": "ready",
        "generated_at": str(date.today()),
        "source_dir": str(BASE),

        # Tables (from your exported notebook files)
        "personas": read_xlsx("cluster_personas.xlsx"),
        "revenue_contribution": read_xlsx("revenue_contribution.xlsx"),
        "rfm_summary": read_xlsx("rfm_summary.xlsx"),
        "promo_roi": read_xlsx("promo_roi.xlsx"),
        "channel_strategy": read_xlsx("channel_strategy.xlsx"),
        "discount_risk": read_xlsx("discount_risk.xlsx"),
        "clv_summary": read_xlsx("clv_summary.xlsx"),
        "cluster_summary": read_xlsx("cluster_summary.xlsx"),
        "education_group_distribution": read_xlsx("cluster_education_group_distribution.xlsx"),
        "marital_group_distribution": read_xlsx("cluster_marital_group_distribution.xlsx"),

        # Visual assets your frontend can load
        "images": [
            "cluster_counts.png",
            "cluster_feature_heatmap.png",
        ],

        # Interactive html outputs (frontend can iframe later)
        "html": [
            "pca_3d.html",
            "pca_3d_centroids.html",
        ],
    }

    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"✅ wrote {OUT}")


if __name__ == "__main__":
    main()
