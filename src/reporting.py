# src/reporting.py
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def attach_cluster_names(df: pd.DataFrame, cluster_name_map: Dict[int, str]) -> pd.DataFrame:
    df = df.copy()
    df["Cluster_Name"] = df["Cluster"].map(cluster_name_map)
    missing = df["Cluster_Name"].isna().sum()
    if missing:
        logger.warning("Cluster_Name missing for %d rows (map incomplete?)", missing)
        df["Cluster_Name"] = df["Cluster_Name"].fillna("Unknown")
    return df


def build_persona_table(cluster_personas: Dict[int, Dict[str, str]]) -> pd.DataFrame:
    rows = []
    for k, v in cluster_personas.items():
        rows.append(
            {
                "Cluster": int(k),
                "Persona": v.get("Persona", ""),
                "Key_Traits": v.get("Key_Traits", ""),
                "Business_Action": v.get("Business_Action", ""),
            }
        )
    out = pd.DataFrame(rows).sort_values("Cluster").reset_index(drop=True)
    return out


def cluster_profile_means(df: pd.DataFrame, cluster_col: str, features: List[str]) -> pd.DataFrame:
    return df.groupby(cluster_col)[features].mean().round(2)


def revenue_contribution(df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    out = (
        df.groupby("Cluster_Name")
        .agg(Customers=(id_col, "count"), Total_Revenue=("Total_Spend", "sum"))
        .reset_index()
    )
    out["Customer_%"] = (out["Customers"] / out["Customers"].sum() * 100).round(2)
    out["Revenue_%"] = (out["Total_Revenue"] / out["Total_Revenue"].sum() * 100).round(2)
    return out.sort_values("Revenue_%", ascending=False).reset_index(drop=True)


def rfm_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Cluster_Name")[["Recency_RFM", "Frequency_RFM", "Monetary_RFM"]].mean().round(2)


def promo_roi(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("Cluster_Name")
        .agg(
            Promo_Response_Rate=("Promo_Responsive", "mean"),
            Avg_Spend=("Total_Spend", "mean"),
            Avg_Deal_Dependency=("Deal_Dependency", "mean"),
        )
        .round(3)
        .reset_index()
        .sort_values("Promo_Response_Rate", ascending=False)
        .reset_index(drop=True)
    )
    return out


def channel_strategy(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Cluster_Name")[["Web_Purchase_Ratio", "Store_Purchase_Ratio"]]
        .mean()
        .round(3)
        .reset_index()
    )


def discount_addiction(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    median_spend = df["Avg_Spend_Per_Purchase"].median()
    df["Discount_Addicted"] = (
        (df["Deal_Dependency"] > 0.5) & (df["Avg_Spend_Per_Purchase"] < median_spend)
    ).astype(int)
    risk = df.groupby("Cluster_Name")["Discount_Addicted"].mean().round(3).reset_index()
    return df, risk


def clv_proxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CLV_Proxy"] = df["Avg_Spend_Per_Purchase"] * df["Frequency_RFM"] * df["Customer_Tenure"]
    out = df.groupby("Cluster_Name")["CLV_Proxy"].mean().round(0).sort_values(ascending=False).reset_index()
    return df, out


def cluster_summary_table(df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    out = (
        df.groupby(["Cluster", "Cluster_Name"])
        .agg(
            Customers=(id_col, "count"),
            Avg_Income=("Income", "mean"),
            Avg_Total_Spend=("Total_Spend", "mean"),
            Avg_Frequency=("Frequency_RFM", "mean"),
            Avg_Recency=("Recency_RFM", "mean"),
            Avg_Deal_Dependency=("Deal_Dependency", "mean"),
            Avg_Product_Variety=("Product_Variety", "mean"),
            Promo_Response_Rate=("Promo_Responsive", "mean"),
            Avg_Web_Ratio=("Web_Purchase_Ratio", "mean"),
            Avg_Store_Ratio=("Store_Purchase_Ratio", "mean"),
        )
        .reset_index()
    )
    out["Customer_%"] = (out["Customers"] / len(df) * 100).round(2)
    out = out.sort_values("Customers", ascending=False).reset_index(drop=True)
    return out
