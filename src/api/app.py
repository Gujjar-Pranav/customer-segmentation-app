# src/api/app.py
from __future__ import annotations
import io
import os
import json
import pandas as pd

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query,UploadFile, File,Depends

from src.api.schemas import PredictRawRequest, PredictEngineeredRequest, PredictResponse

from src.api.service import predict, predict_batch, get_bundle, model_info
from fastapi.responses import StreamingResponse,JSONResponse,FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from src.api.auth import require_api_key


ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "outputs/latest")


#ARTIFACTS_DIR = Path("outputs/latest")

app = FastAPI(
    title="Customer Segmentation API",
    version="1.0.0",
    description="FastAPI backend for customer segmentation using saved KMeans model artifacts",
)

# Paths
LATEST_DIR = Path("outputs/latest")
INSIGHTS_JSON = Path("outputs/insights.json")

# Serve PNGs from outputs/latest at /assets/*
app.mount("/assets", StaticFiles(directory=str(LATEST_DIR)), name="assets")


# --- CORS ---
# For production: replace "*" with your frontend domain (ex: "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def warmup_model_cache():
    # Load artifacts once at startup
    get_bundle(str(ARTIFACTS_DIR))

@app.get("/")
def root():
    return {"message": "Customer Segmentation API is running ✅"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict_endpoint(
    req: PredictEngineeredRequest,
    mode: Literal["raw", "engineered"] = Query(default="engineered"),
):
    cluster, name = predict(req.model_dump(), ARTIFACTS_DIR, mode=mode)
    return PredictResponse(predicted_cluster=cluster, predicted_cluster_name=name, mode=mode)

@app.post("/predict-engineered", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict_engineered(req: PredictEngineeredRequest):
    cluster, name = predict(req.model_dump(), ARTIFACTS_DIR, mode="engineered")
    return PredictResponse(predicted_cluster=cluster, predicted_cluster_name=name, mode="engineered")


@app.post("/predict-raw", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict_raw(req: PredictRawRequest):
    payload = req.model_dump()
    payload["Dt_Customer"] = payload["Dt_Customer"].isoformat()
    cluster, name = predict(payload, ARTIFACTS_DIR, mode="raw")
    return PredictResponse(predicted_cluster=cluster, predicted_cluster_name=name, mode="raw")

@app.post("/predict-batch", dependencies=[Depends(require_api_key)])
async def predict_batch_endpoint(
    file: UploadFile = File(...),
    mode: Literal["raw", "engineered"] = Query(default="raw"),
):
    try:
        contents = await file.read()

        df_in = pd.read_csv(io.BytesIO(contents))
        from src.api.service import predict_batch

        df_out = predict_batch(df_in=df_in, artifacts_dir=ARTIFACTS_DIR, mode=mode)

        # return CSV file response
        buffer = io.StringIO()
        df_out.to_csv(buffer, index=False)
        buffer.seek(0)

        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/insights")
def get_insights():
    if not INSIGHTS_JSON.exists():
        return {"error": "insights.json not found. Run insights generator first."}

    data = json.loads(INSIGHTS_JSON.read_text(encoding="utf-8"))

    # Convert image filenames into URLs the frontend can load
    data["image_urls"] = [f"/assets/{name}" for name in data.get("images", [])]
    return data

@app.get("/model-info")
def model_info_endpoint():
    return model_info(ARTIFACTS_DIR)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Bad Request",
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {
        "service": "customer-segmentation-app",
        "artifacts_dir": str(ARTIFACTS_DIR),
    }


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok", "message": "Customer Segmentation API is running"}
