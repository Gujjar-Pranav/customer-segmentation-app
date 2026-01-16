# src/inference_features.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd


REQUIRED_RAW_COLS: List[str] = [
    "Income",
    "Year_Birth",
    "Kidhome",
    "Teenhome",
    "Recency",
    "Dt_Customer",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumDealsPurchases",
]

SPEND_COLS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

PURCHASE_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]

FINAL_FEATURES_13 = [
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


def build_features_for_inference(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the same 13 clustering features from raw input.
    Assumes raw_df contains the REQUIRED_RAW_COLS and optional campaign fields for promo.
    """
    df = raw_df.copy()

    # Ensure types
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")

    # Feature engineering (match training logic)
    df["Age"] = datetime.now().year - df["Year_Birth"]
    df["Age"] = df["Age"].clip(lower=10, upper=100)

    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    df["Total_Spend"] = df[SPEND_COLS].sum(axis=1)
    df["Total_Purchases"] = df[PURCHASE_COLS].sum(axis=1)

    df["Customer_Tenure"] = (datetime.now() - df["Dt_Customer"]).dt.days

    # Promo responsive: if campaign fields exist, use them; else default 0
    campaign_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]
    existing_campaign = [c for c in campaign_cols if c in df.columns]
    if existing_campaign:
        df["Total_Campaign_Accepted"] = df[existing_campaign].sum(axis=1)
        df["Promo_Responsive"] = (df["Total_Campaign_Accepted"] > 0).astype(int)
    else:
        df["Promo_Responsive"] = 0

    # RFM naming
    df["Recency_RFM"] = df["Recency"]
    df["Frequency_RFM"] = df["Total_Purchases"]
    df["Monetary_RFM"] = df["Total_Spend"]

    # Ratios
    df["Web_Purchase_Ratio"] = df["NumWebPurchases"] / (df["Total_Purchases"] + 1)
    df["Store_Purchase_Ratio"] = df["NumStorePurchases"] / (df["Total_Purchases"] + 1)

    df["Avg_Spend_Per_Purchase"] = df["Total_Spend"] / (df["Total_Purchases"] + 1)

    df["Deal_Dependency"] = df["NumDealsPurchases"] / (df["Total_Purchases"] + 1)

    df["Product_Variety"] = (df[SPEND_COLS] > 0).sum(axis=1)

    return df[FINAL_FEATURES_13].copy()
