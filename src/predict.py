# src/predict.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.artifacts import load_model_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict customer cluster using saved artifacts")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="outputs/latest",
        help="Folder containing scaler.joblib, kmeans.joblib, artifacts_meta.joblib",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="engineered",
        choices=["engineered", "raw"],
        help="engineered = provide 13 final features; raw = provide raw fields and auto-engineer features",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-json", type=str, help="Customer input as JSON string (one record)")
    group.add_argument("--input-csv", type=str, help="Path to CSV file containing customers")

    parser.add_argument(
        "--out",
        type=str,
        default="predictions.xlsx",
        help="Output file path (.xlsx or .csv)",
    )
    return parser.parse_args()


def predict_one_engineered(artifacts_dir: Path, engineered_features: Dict[str, Any]) -> Dict[str, Any]:
    bundle = load_model_artifacts(artifacts_dir)

    scaler = bundle["scaler"]
    model = bundle["model"]
    final_features = bundle["final_features"]

    # Make input DF (one row)
    X = pd.DataFrame([engineered_features])

    # Ensure all required features exist
    missing = [c for c in final_features if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required engineered features: {missing}")

    X = X[final_features].copy()
    X_scaled = scaler.transform(X)

    cluster = int(model.predict(X_scaled)[0])

    # Map cluster -> name (stored in artifacts meta)
    cluster_name_map = bundle.get("meta", {}).get("cluster_names", {}) or {}
    cluster_name = cluster_name_map.get(str(cluster)) or cluster_name_map.get(cluster)

    return {
        "predicted_cluster": cluster,
        "predicted_cluster_name": cluster_name,
    }

def predict_df_engineered(artifacts_dir: Path, X: pd.DataFrame) -> pd.DataFrame:
    bundle = load_model_artifacts(artifacts_dir)
    scaler = bundle["scaler"]
    model = bundle["model"]
    final_features = bundle["final_features"]

    missing = [c for c in final_features if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required engineered features in dataframe: {missing}")

    X_use = X[final_features].copy()
    X_scaled = scaler.transform(X_use)

    clusters = model.predict(X_scaled).astype(int)

    cluster_name_map = bundle.get("meta", {}).get("cluster_names", {}) or {}
    names = [cluster_name_map.get(str(int(c))) or cluster_name_map.get(int(c)) for c in clusters]

    out = X.copy()
    out["predicted_cluster"] = clusters
    out["predicted_cluster_name"] = names
    return out


def save_predictions(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_excel(out_path, index=False)


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    out_path = Path(args.out)

    if args.input_json:
        customer_dict = json.loads(args.input_json)

        if args.mode == "engineered":
            result = predict_one_engineered(artifacts_dir, customer_dict)
        else:
            from src.inference_features import build_features_for_inference
            raw_df = pd.DataFrame([customer_dict])
            X_feat = build_features_for_inference(raw_df)
            result = predict_one_engineered(artifacts_dir, X_feat.iloc[0].to_dict())

        print("\n✅ Prediction Result")
        print(json.dumps(result, indent=2))
        return

    # ---- CSV batch mode ----
    df_in = pd.read_csv(args.input_csv)

    if args.mode == "raw":
        from src.inference_features import build_features_for_inference
        X_feat = build_features_for_inference(df_in)
        df_out = predict_df_engineered(artifacts_dir, X_feat)
        # keep original raw columns too (optional)
        df_out = pd.concat([df_in.reset_index(drop=True), df_out[["predicted_cluster", "predicted_cluster_name"]]], axis=1)
    else:
        df_out = predict_df_engineered(artifacts_dir, df_in)

    save_predictions(df_out, out_path)
    print(f"\n✅ Saved batch predictions to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
