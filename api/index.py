"""
Vercel serverless entrypoint — exposes the FastAPI app.
Models load lazily on first request and are cached in the module scope.
"""

import sys
import os
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Ecomeal AI",
    description="Food waste prediction and recommendation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy model cache ────────────────────────────────────────────────────────────
_models = {}

def _get_models():
    if _models:
        return _models
    from src.models.wastage_predictor import WastagePredictor
    from src.models.anomaly_detector import InventoryAnomalyDetector

    wastage = WastagePredictor()
    wastage.load()
    anomaly = InventoryAnomalyDetector()
    anomaly.load()

    _models["wastage"] = wastage
    _models["anomaly"] = anomaly
    return _models


@app.get("/")
def root():
    return {
        "name": "Ecomeal AI",
        "description": "Food waste prediction and recommendation API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "models": ["wastage_predictor", "anomaly_detector"]}


@app.post("/api/v1/inventory/analyze")
def analyze(payload: dict):
    """Analyze inventory items for waste risk."""
    import pandas as pd
    from src.data.preprocessor import clean_inventory, encode_categoricals

    models = _get_models()
    records = payload.get("items", [])
    if not records:
        return JSONResponse({"error": "No items provided"}, status_code=422)

    df = pd.DataFrame(records)
    df = clean_inventory(df)
    preds = models["wastage"].predict(df)
    anom = models["anomaly"].detect(df)

    results = []
    for i, row in preds.iterrows():
        results.append({
            "index": i,
            "waste_probability": row["waste_probability"],
            "risk_level": row["risk_level_pred"],
            "is_anomaly": bool(anom.iloc[i]["is_anomaly"]),
            "anomaly_reason": anom.iloc[i].get("anomaly_reason"),
        })
    return {"results": results, "count": len(results)}


@app.post("/api/v1/inventory/explain/{idx}")
def explain(idx: int, payload: dict):
    """Return plain-English explanation for a single inventory item."""
    import pandas as pd
    from src.data.preprocessor import clean_inventory

    models = _get_models()
    records = payload.get("items", [])
    if idx >= len(records):
        return JSONResponse({"error": "Index out of range"}, status_code=422)

    df = clean_inventory(pd.DataFrame(records))
    row_df = df.iloc[[idx]]
    explanations = models["wastage"].explain(row_df, max_items=1)
    return {"index": idx, "explanation": explanations[0] if explanations else {}}
