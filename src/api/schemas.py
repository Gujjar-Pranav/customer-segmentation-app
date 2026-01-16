# src/api/schemas.py
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------- Engineered (13 features) ----------
class PredictEngineeredRequest(BaseModel):
    Income: float = Field(..., ge=0)
    Age: int = Field(..., ge=0, le=120)
    Total_Children: int = Field(..., ge=0)
    Recency_RFM: int = Field(..., ge=0)
    Frequency_RFM: int = Field(..., ge=0)
    Monetary_RFM: float = Field(..., ge=0)
    Avg_Spend_Per_Purchase: float = Field(..., ge=0)
    Customer_Tenure: int = Field(..., ge=0)
    Web_Purchase_Ratio: float = Field(..., ge=0, le=1)
    Store_Purchase_Ratio: float = Field(..., ge=0, le=1)
    Promo_Responsive: int = Field(..., ge=0, le=1)
    Deal_Dependency: float = Field(..., ge=0)
    Product_Variety: int = Field(..., ge=0)


# ---------- Raw (marketing dataset style) ----------
# These are the typical columns needed to compute engineered features.
# Dt_Customer accepts "YYYY-MM-DD".
class PredictRawRequest(BaseModel):
    Income: float = Field(..., ge=0)
    Year_Birth: int = Field(..., ge=1900, le=2026)

    Kidhome: int = Field(..., ge=0)
    Teenhome: int = Field(..., ge=0)

    Dt_Customer: date

    Recency: int = Field(..., ge=0)

    NumWebPurchases: int = Field(..., ge=0)
    NumCatalogPurchases: int = Field(..., ge=0)
    NumStorePurchases: int = Field(..., ge=0)
    NumWebVisitsMonth: int = Field(..., ge=0)
    NumDealsPurchases: int = Field(..., ge=0)

    MntWines: float = Field(..., ge=0)
    MntFruits: float = Field(..., ge=0)
    MntMeatProducts: float = Field(..., ge=0)
    MntFishProducts: float = Field(..., ge=0)
    MntSweetProducts: float = Field(..., ge=0)
    MntGoldProds: float = Field(..., ge=0)

    AcceptedCmp1: int = Field(..., ge=0, le=1)
    AcceptedCmp2: int = Field(..., ge=0, le=1)
    AcceptedCmp3: int = Field(..., ge=0, le=1)
    AcceptedCmp4: int = Field(..., ge=0, le=1)
    AcceptedCmp5: int = Field(..., ge=0, le=1)
    Response: int = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    predicted_cluster: int
    predicted_cluster_name: Optional[str] = None
    mode: Literal["raw", "engineered"]
