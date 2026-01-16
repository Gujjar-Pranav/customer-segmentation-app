# src/features.py
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_features(df: pd.DataFrame, dt_col: str = "Dt_Customer") -> pd.DataFrame:
    df = df.copy()

    # Age
    if "Year_Birth" not in df.columns:
        raise KeyError("Missing column: Year_Birth")
    current_year = datetime.now().year
    df["Age"] = current_year - df["Year_Birth"]
    df["Age"] = df["Age"].clip(lower=10, upper=100)

    # Children
    for col in ["Kidhome", "Teenhome"]:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    # Spend + purchases
    spend_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    for c in spend_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    df["Total_Spend"] = df[spend_cols].sum(axis=1)

    purchase_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    for c in purchase_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    df["Total_Purchases"] = df[purchase_cols].sum(axis=1)

    # Tenure
    if dt_col not in df.columns:
        raise KeyError(f"Missing datetime column: {dt_col}")
    df["Customer_Tenure"] = (datetime.now() - df[dt_col]).dt.days

    # Campaign acceptance
    campaign_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]
    for c in campaign_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    df["Total_Campaign_Accepted"] = df[campaign_cols].sum(axis=1)

    # RFM naming
    if "Recency" not in df.columns:
        raise KeyError("Missing column: Recency")
    df["Recency_RFM"] = df["Recency"]
    df["Frequency_RFM"] = df["Total_Purchases"]
    df["Monetary_RFM"] = df["Total_Spend"]

    # Ratios + behavior
    df["Web_Purchase_Ratio"] = df["NumWebPurchases"] / (df["Total_Purchases"] + 1)
    df["Store_Purchase_Ratio"] = df["NumStorePurchases"] / (df["Total_Purchases"] + 1)
    df["Catalog_Purchase_Ratio"] = df["NumCatalogPurchases"] / (df["Total_Purchases"] + 1)

    df["Avg_Spend_Per_Purchase"] = df["Total_Spend"] / (df["Total_Purchases"] + 1)

    if "NumDealsPurchases" not in df.columns:
        raise KeyError("Missing column: NumDealsPurchases")
    df["Deal_Dependency"] = df["NumDealsPurchases"] / (df["Total_Purchases"] + 1)

    df["Has_Children"] = (df["Total_Children"] > 0).astype(int)
    df["Promo_Responsive"] = (df["Total_Campaign_Accepted"] > 0).astype(int)

    product_cols = spend_cols
    df["Product_Variety"] = (df[product_cols] > 0).sum(axis=1)

    logger.info("Feature engineering complete. New shape=%s", df.shape)
    return df


def get_final_features() -> List[str]:
    # This matches your notebook's clustering feature set
    return [
        "Income",
        "Age",
        "Total_Children",
        "Recency_RFM",
        "Frequency_RFM",
        "Monetary_RFM",
        "Avg_Spend_Per_Purchase",
        "Customer_Tenure",
        "Web_Purchase_Ratio",
        "Store_Purchase_Ratio",
        "Promo_Responsive",
        "Deal_Dependency",
        "Product_Variety",
    ]


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feats = get_final_features()
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise KeyError(f"Missing engineered features: {missing}")
    X = df[feats].copy()

    # Optional: replace inf values from divide (should be rare with +1 protection)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, feats
