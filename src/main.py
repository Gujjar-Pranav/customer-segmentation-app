# src/main.py
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict
from src.io_utils import load_excel
from src.preprocessing import (parse_datetime,drop_corrupted_ids,drop_constant_columns,
                               impute_income, drop_duplicates, iqr_cap,)
from src.features import add_features, build_feature_matrix
from src.modeling import fit_kmeans, attach_clusters
from src.artifacts import save_model_artifacts
from src.evaluation import evaluate_kmeans_quality, evaluate_partition, compare_models
from src.reporting import (attach_cluster_names,build_persona_table,revenue_contribution,
                           rfm_summary,promo_roi,channel_strategy,
                           discount_addiction,clv_proxy,cluster_summary_table,)
from src.categoricals import (add_marital_group,add_education_group,cluster_distribution_table,)
from src.viz import plot_cluster_counts, plot_normalized_heatmap
from src.pca_viz import compute_pca, save_pca_scatter_html, save_pca_with_centroids_html
from src.validation import validate_config
import numpy as np
import pandas as pd
import yaml


# ---------- Logging ----------
def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    handlers = [logging.StreamHandler()]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )



# ---------- Config ----------
def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dir(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------- Main pipeline ----------
def run_pipeline(cfg: Dict[str, Any]) -> None:
    logger = logging.getLogger("pipeline")

    seed = int(cfg["project"]["seed"])
    np.random.seed(seed)

    input_path = cfg["paths"]["input_path"]
    output_dir = ensure_output_dir(cfg["paths"]["output_dir"])

    logger.info("Starting project: %s", cfg["project"]["name"])
    logger.info("Input: %s", input_path)
    logger.info("Outputs: %s", output_dir.resolve())

    # ---- Load ----
    df = load_excel(input_path)

    # ---- Preprocess ----
    df = parse_datetime(df, cfg["data"]["datetime_col"])

    if cfg["preprocessing"]["drop_id_if_zero"]:
        df = drop_corrupted_ids(df, cfg["data"]["id_col"])

    if cfg["preprocessing"]["treat_income_zero_as_missing"]:
        df = impute_income(df, "Income")

    df, constant_cols = drop_constant_columns(df)

    if cfg["preprocessing"]["drop_duplicates"]:
        df = drop_duplicates(df)

    df = iqr_cap(df, cfg["preprocessing"]["iqr_cap_cols"])

    logger.info("After preprocessing: shape=%s", df.shape)

    # Quick save checkpoint (optional but helpful)
    if cfg["run"]["save_tables"]:
        from src.io_utils import save_excel
        save_excel(df, output_dir / "data_cleaned.xlsx", index=False)

    logger.info("Step 3 complete ✅ Next: feature engineering + clustering.")

    # ---- Feature engineering ----
    df = add_features(df, dt_col=cfg["data"]["datetime_col"])
    X, final_features = build_feature_matrix(df)
    logger.info("Feature matrix ready: X shape=%s, n_features=%d", X.shape, len(final_features))

    if cfg["run"]["save_tables"]:
        from src.io_utils import save_excel
        save_excel(df, output_dir / "data_featured.xlsx", index=False)

    # ---- Clustering (KMeans) ----
    k = int(cfg["clustering"]["k"])
    scaler_name = cfg["clustering"]["scaler"]
    n_init = int(cfg["clustering"]["kmeans"]["n_init"])
    random_state = int(cfg["clustering"]["kmeans"]["random_state"])

    artifacts = fit_kmeans(
        X=X,
        scaler_name=scaler_name,
        k=k,
        n_init=n_init,
        random_state=random_state,
    )

    # Attach clusters to df
    df = attach_clusters(df, artifacts.labels, col="Cluster")
    logger.info(
        "Clusters attached. Cluster counts: %s",
        df["Cluster"].value_counts().sort_index().to_dict(),
    )

    # ✅ Save trained artifacts (scaler + kmeans + metadata)

    save_model_artifacts(
        run_dir=Path(cfg["paths"]["output_dir"]),
        scaler=artifacts.scaler,
        model=artifacts.model,
        final_features=final_features,
        meta={
            "k": k,
            "scaler_name": scaler_name,
            "random_state": random_state,
            "n_init": n_init,
            # ✅ store cluster names for inference time
            "cluster_names": cfg["personas"]["cluster_names"],
        },
    )

    # Save clustered data
    if cfg["run"]["save_tables"]:
        from src.io_utils import save_excel
        save_excel(df, output_dir / "data_clustered.xlsx", index=False)

    # ---- Evaluation ----
    km_scores = evaluate_partition(artifacts.X_scaled, artifacts.labels)
    km_quality = evaluate_kmeans_quality(artifacts.X_scaled, artifacts.labels)

    logger.info(
        "KMeans metrics: silhouette=%.3f dbi=%.3f ch=%.1f | neg_sil=%.2f%% sil_min=%.3f",
        km_scores["Silhouette_Score"],
        km_scores["Davies_Bouldin_Index"],
        km_scores["Calinski_Harabasz_Index"],
        km_quality.negative_silhouette_pct,
        km_quality.silhouette_min,
    )

    # Optional comparison table
    comp = compare_models(artifacts.X_scaled, k=k)
    if cfg["run"]["save_tables"]:
        from src.io_utils import save_excel
        if not comp.empty:
            save_excel(comp, output_dir / "model_comparison.xlsx", index=False)

    # ---- Reporting ----
    # YAML keys come as strings; convert to int keys
    cluster_name_map = {int(k): v for k, v in cfg["personas"]["cluster_names"].items()}
    df = attach_cluster_names(df, cluster_name_map)

    persona_cfg = {int(k): v for k, v in cfg["personas"]["cluster_personas"].items()}
    persona_df = build_persona_table(persona_cfg)

    rev_df = revenue_contribution(df, id_col=cfg["data"]["id_col"])
    rfm_df = rfm_summary(df)
    promo_df = promo_roi(df)
    channel_df = channel_strategy(df)

    df, discount_risk_df = discount_addiction(df)
    df, clv_df = clv_proxy(df)

    summary_df = cluster_summary_table(df, id_col=cfg["data"]["id_col"])

    if cfg["run"]["save_tables"]:
        from src.io_utils import save_excel
        save_excel(persona_df, output_dir / "cluster_personas.xlsx", index=False)
        save_excel(rev_df, output_dir / "revenue_contribution.xlsx", index=False)
        save_excel(rfm_df, output_dir / "rfm_summary.xlsx", index=False)
        save_excel(promo_df, output_dir / "promo_roi.xlsx", index=False)
        save_excel(channel_df, output_dir / "channel_strategy.xlsx", index=False)
        save_excel(discount_risk_df, output_dir / "discount_risk.xlsx", index=False)
        save_excel(clv_df, output_dir / "clv_summary.xlsx", index=False)
        save_excel(summary_df, output_dir / "cluster_summary.xlsx", index=False)

        # overwrite clustered file with named clusters + extra fields
        save_excel(df, output_dir / "data_clustered_named.xlsx", index=False)

    logger.info("Reporting exports complete ✅")

    # ---- Categorical consolidation + distributions ----
    df = add_marital_group(df)
    df = add_education_group(df)

    marital_dist = cluster_distribution_table(
        df,
        cluster_name_col="Cluster_Name",
        category_col="Marital_Group",
        ordered_cols=["Single", "Couple", "Previously Married"],
    )

    edu_dist = cluster_distribution_table(
        df,
        cluster_name_col="Cluster_Name",
        category_col="Education_Group",
        ordered_cols=["Basic", "Graduate", "Postgraduate"],
    )

    if cfg["run"]["save_tables"]:
        from src.io_utils import save_excel
        save_excel(marital_dist, output_dir / "cluster_marital_group_distribution.xlsx", index=True)
        save_excel(edu_dist, output_dir / "cluster_education_group_distribution.xlsx", index=True)

        # Save final “full” dataframe with these new columns too
        save_excel(df, output_dir / "data_clustered_named_final.xlsx", index=False)

    logger.info("Categorical distributions exported ✅")

    # ---- Visualizations (saved; no popups unless show_plots=true) ----
    save_plots = bool(cfg["run"]["save_plots"])
    show_plots = bool(cfg["run"]["show_plots"])

    if save_plots or show_plots:
        plot_cluster_counts(
            df,
            out_path=output_dir / cfg["plots"]["cluster_counts_png"],
            save=save_plots,
            show=show_plots,
        )

        # Use the same final_features you already used for clustering (13 cols)
        plot_normalized_heatmap(
            df,
            feature_cols=final_features,
            out_path=output_dir / cfg["plots"]["heatmap_png"],
            save=save_plots,
            show=show_plots,
        )

        # PCA HTML (uses scaled features from KMeans artifacts)
        # cluster_name_map already exists earlier in your pipeline
        import pandas as pd
        from sklearn.decomposition import PCA
        import plotly.express as px

        pca = PCA(n_components=3, random_state=42)
        X_pca = pca.fit_transform(artifacts.X_scaled)

        pca_df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2", "PCA3"])
        pca_df["Cluster_Name"] = df["Cluster_Name"].values

        fig = px.scatter_3d(
            pca_df,
            x="PCA1", y="PCA2", z="PCA3",
            color="Cluster_Name",
            opacity=0.7,
            title="3D PCA — Cluster Visualization",
        )
        fig.write_html(str(output_dir / cfg["plots"]["pca_html"]))
        logger.info("Saved PCA HTML: %s", (output_dir / cfg["plots"]["pca_html"]).resolve())

        # PCA exports + HTML
        pca_points, explained_df, loadings_df, centroids_df = compute_pca(
            X_scaled=artifacts.X_scaled,
            feature_names=final_features,
            labels=artifacts.labels,
            cluster_name_map=cluster_name_map,
            kmeans_centers=artifacts.model.cluster_centers_,
        )

        if cfg["run"]["save_tables"]:
            from src.io_utils import save_excel
            save_excel(explained_df, output_dir / "pca_explained_variance.xlsx", index=False)
            save_excel(loadings_df.reset_index().rename(columns={"index": "Feature"}),
                       output_dir / "pca_loadings.xlsx", index=False)

        save_pca_scatter_html(pca_points, output_dir / cfg["plots"]["pca_html"])

        if centroids_df is not None:
            save_pca_with_centroids_html(
                pca_points,
                centroids_df,
                output_dir / cfg["plots"]["pca_centroids_html"],
            )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer Segmentation App")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config first (without logging)
    cfg = load_config(args.config)

    # Create run directory
    from src.run_manager import create_run_dir, copy_config_to_run_dir
    run_dir, run_name = create_run_dir(cfg)

    # Setup logging (console + file)
    log_file = str(run_dir / "run.log")
    setup_logging(args.log_level, log_file=log_file)

    # Now validate config (with logging)
    vr = validate_config(cfg)
    if not vr.ok:
        raise ValueError("Config validation failed:\n" + "\n".join(vr.errors))

    # Save the config used for this run
    copy_config_to_run_dir(args.config, run_dir)

    logging.getLogger("pipeline").info("Run folder: %s", run_dir.resolve())

    # Run pipeline with run_dir override
    cfg["paths"]["output_dir"] = str(run_dir)

    run_pipeline(cfg)

    # Update outputs/latest
    from src.latest_manager import update_latest
    latest_dir = Path("outputs/latest")
    update_latest(Path(cfg["paths"]["output_dir"]), latest_dir)

    logging.getLogger("pipeline").info("✅ Latest outputs updated at: %s", latest_dir.resolve())


if __name__ == "__main__":
    main()

