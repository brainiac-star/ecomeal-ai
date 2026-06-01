"""
Vercel serverless entrypoint — FastAPI app serving wastage predictions.
Only loads XGBoost + LightGBM models (no scikit-learn dependency).
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Ecomeal AI API",
    description="Food waste prediction API — wastage risk, feature explanations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy model cache ────────────────────────────────────────────────────────────
_lgb = None
_metrics = {}

FEATURE_COLS = [
    "quantity", "daily_consumption", "days_to_expiry", "total_shelf_life_days",
    "days_since_purchase", "price_per_unit", "wastage_history_pct",
    "stock_days_available", "stock_expiry_ratio", "shelf_life_consumed_pct",
    "potential_waste_value", "overstock_flag", "below_reorder_point",
    "storage_type_enc", "category_code",
]

MODEL_DIR = Path(__file__).parent.parent / "models"


def _load():
    global _lgb, _metrics
    if _lgb is not None:
        return
    _lgb = joblib.load(MODEL_DIR / "lgb_wastage.pkl")
    try:
        _metrics = joblib.load(MODEL_DIR / "wastage_metrics.pkl")
    except Exception:
        _metrics = {}


def _prepare(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    # Encode storage_type
    storage_map = {"frozen": 0, "refrigerated": 1, "ambient": 2}
    if "storage_type" in df.columns:
        df["storage_type_enc"] = df["storage_type"].map(storage_map).fillna(1)
    # Encode category
    if "category" in df.columns:
        cats = sorted(df["category"].dropna().unique().tolist())
        cat_map = {c: i for i, c in enumerate(cats)}
        df["category_code"] = df["category"].map(cat_map).fillna(0)
    # Derived features
    df["daily_consumption"] = df.get("daily_consumption", pd.Series(1.0)).clip(lower=0.01)
    df["stock_days_available"] = df.get("quantity", 0) / df["daily_consumption"]
    dte = df.get("days_to_expiry", pd.Series(30))
    df["stock_expiry_ratio"] = df["stock_days_available"] / dte.clip(lower=1)
    tsl = df.get("total_shelf_life_days", pd.Series(30))
    dsp = df.get("days_since_purchase", pd.Series(0))
    df["shelf_life_consumed_pct"] = (dsp / tsl.clip(lower=1)).clip(0, 1)
    df["potential_waste_value"] = (
        df.get("quantity", 0) *
        df.get("wastage_history_pct", 0.1) *
        df.get("price_per_unit", 10)
    )
    df["overstock_flag"] = (df["stock_days_available"] > dte * 1.2).astype(int)
    df["below_reorder_point"] = (
        df.get("quantity", 0) < df.get("reorder_point", 0)
    ).astype(int)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_COLS].fillna(0)


def _predict(X: pd.DataFrame):
    return _lgb.predict_proba(X)[:, 1]


def _risk(p: float) -> str:
    if p >= 0.70: return "critical"
    if p >= 0.45: return "high"
    if p >= 0.20: return "medium"
    return "low"


@app.get("/")
def root():
    return {
        "name": "Ecomeal AI",
        "description": "Food waste prediction API for restaurants",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "analyze": "POST /api/v1/inventory/analyze",
            "explain": "POST /api/v1/inventory/explain/{idx}",
        },
    }


@app.get("/health")
def health():
    try:
        _load()
        return {"status": "ok", "model_metrics": _metrics}
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.post("/api/v1/inventory/analyze")
def analyze(payload: dict):
    """
    Analyze inventory items for waste risk.

    Body: { "items": [ { "ingredient_name": "Tomatoes", "quantity": 10,
                          "daily_consumption": 2, "days_to_expiry": 3, ... } ] }
    """
    _load()
    records = payload.get("items", [])
    if not records:
        return JSONResponse({"error": "No items provided"}, status_code=422)

    X = _prepare(records)
    probs = _predict(X)

    results = []
    for i, (prob, rec) in enumerate(zip(probs, records)):
        results.append({
            "index": i,
            "ingredient_name": rec.get("ingredient_name", f"item_{i}"),
            "waste_probability": round(float(prob), 4),
            "risk_level": _risk(prob),
            "will_waste": bool(prob >= 0.5),
        })

    results.sort(key=lambda r: r["waste_probability"], reverse=True)
    return {
        "count": len(results),
        "critical": sum(1 for r in results if r["risk_level"] == "critical"),
        "high": sum(1 for r in results if r["risk_level"] == "high"),
        "results": results,
    }


@app.post("/api/v1/inventory/explain/{idx}")
def explain(idx: int, payload: dict):
    """Return top feature contributions for a single item."""
    _load()
    records = payload.get("items", [])
    if idx >= len(records):
        return JSONResponse({"error": "Index out of range"}, status_code=422)

    X = _prepare([records[idx]])
    contribs = _lgb.predict(X, pred_contrib=True)
    sv = contribs[0, :-1]
    max_abs = np.abs(sv).max()
    if max_abs > 0:
        sv = sv / max_abs

    top = sorted(zip(FEATURE_COLS, sv), key=lambda x: abs(x[1]), reverse=True)[:5]
    features = [
        {
            "feature": f,
            "impact": round(float(v), 4),
            "direction": "increases_risk" if v > 0 else "decreases_risk",
        }
        for f, v in top
    ]
    prob = float(_predict(X)[0])
    return {
        "index": idx,
        "ingredient_name": records[idx].get("ingredient_name", f"item_{idx}"),
        "waste_probability": round(prob, 4),
        "risk_level": _risk(prob),
        "top_features": features,
    }
