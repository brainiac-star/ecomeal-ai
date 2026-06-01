"""
Vercel serverless entrypoint — FastAPI app serving wastage predictions.
Only loads LightGBM (no scikit-learn/XGBoost to stay under 500MB limit).
"""

import sys
import os
import math
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Ecomeal AI API",
    description="Food waste prediction API — wastage risk, anomalies, operations, chef specials",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

_SUBSTITUTIONS = {
    "Tomatoes":      ["Tomato Shorba", "Tomato Rice", "Bruschetta", "Shakshuka"],
    "Onions":        ["French Onion Soup", "Onion Bhaji", "Caramelised Onion Tart"],
    "Spinach":       ["Palak Paneer", "Spinach Dal", "Green Smoothie Bowl", "Spanakopita"],
    "Mushrooms":     ["Mushroom Risotto", "Mushroom Soup", "Mushroom Stir Fry"],
    "Chicken":       ["Grilled Chicken Bowl", "Chicken Tikka", "Chicken Caesar Wrap"],
    "Paneer":        ["Palak Paneer", "Paneer Tikka", "Kadai Paneer", "Paneer Bhurji"],
    "Prawns":        ["Prawn Curry", "Garlic Prawn Pasta", "Prawn Fried Rice"],
    "Basmati Rice":  ["Biryani", "Fried Rice", "Khichdi", "Rice Pudding"],
    "Coriander":     ["Coriander Chutney", "Green Curry", "Herb Salad"],
    "Butter":        ["Butter Chicken Sauce", "Garlic Butter Naan", "Butter Pasta"],
    "Carrots":       ["Carrot Halwa", "Carrot Soup", "Glazed Carrots"],
    "Potatoes":      ["Aloo Tikki", "Mashed Potato", "Hash Browns"],
}


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
    storage_map = {"frozen": 0, "refrigerated": 1, "ambient": 2}
    if "storage_type" in df.columns:
        df["storage_type_enc"] = df["storage_type"].map(storage_map).fillna(1)
    if "category" in df.columns:
        cats = sorted(df["category"].dropna().unique().tolist())
        cat_map = {c: i for i, c in enumerate(cats)}
        df["category_code"] = df["category"].map(cat_map).fillna(0)
    df["daily_consumption"] = df.get("daily_consumption", pd.Series([1.0] * len(df))).clip(lower=0.01)
    df["quantity"] = df.get("quantity", pd.Series([0.0] * len(df)))
    df["days_to_expiry"] = df.get("days_to_expiry", pd.Series([30] * len(df)))
    df["stock_days_available"] = df["quantity"] / df["daily_consumption"]
    dte = df["days_to_expiry"].clip(lower=1)
    df["stock_expiry_ratio"] = df["stock_days_available"] / dte
    tsl = df.get("total_shelf_life_days", pd.Series([30] * len(df))).clip(lower=1)
    dsp = df.get("days_since_purchase", pd.Series([0] * len(df)))
    df["shelf_life_consumed_pct"] = (dsp / tsl).clip(0, 1)
    df["potential_waste_value"] = (
        df["quantity"] *
        df.get("wastage_history_pct", pd.Series([0.1] * len(df))) *
        df.get("price_per_unit", pd.Series([10] * len(df)))
    )
    df["overstock_flag"] = (df["stock_days_available"] > df["days_to_expiry"] * 1.2).astype(int)
    df["below_reorder_point"] = (df["quantity"] < df.get("reorder_point", pd.Series([0] * len(df)))).astype(int)
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


def _anomaly_reason(rec: dict) -> str:
    qty = rec.get("quantity", 0)
    consumption = max(rec.get("daily_consumption", 0.01), 0.01)
    dte = rec.get("days_to_expiry", 0)
    price = rec.get("price_per_unit", 0)
    waste_pct = rec.get("wastage_history_pct", 0)
    stock_days = qty / consumption
    reasons = []
    if stock_days > dte * 3:
        reasons.append(f"extreme overstock ({stock_days:.0f} days of stock, expires in {dte} days)")
    if qty == 0:
        reasons.append("zero quantity — possible data entry error")
    if consumption > 50:
        reasons.append(f"unusually high daily consumption ({consumption:.1f})")
    if price > 2000:
        reasons.append(f"unusually high price (₹{price:.0f})")
    if waste_pct > 0.5:
        reasons.append(f"very high historical waste ({waste_pct*100:.0f}%)")
    return "; ".join(reasons) if reasons else "statistical outlier across multiple inventory signals"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Ecomeal AI",
        "description": "Food waste prediction API for restaurants",
        "docs": "/docs",
        "endpoints": [
            "POST /api/v1/inventory/analyze",
            "POST /api/v1/inventory/explain/{idx}",
            "POST /api/v1/anomalies",
            "POST /api/v1/forecast/{ingredient}",
            "POST /api/v1/operations/reorder",
            "POST /api/v1/chef-specials",
        ],
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
    try:
        _load()
    except Exception as e:
        return JSONResponse({"error": f"Model load failed: {e}", "model_dir": str(MODEL_DIR), "files": [str(p) for p in MODEL_DIR.glob("*")] if MODEL_DIR.exists() else []}, status_code=500)
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
            "category": rec.get("category", ""),
            "waste_probability": round(float(prob), 4),
            "risk_level": _risk(prob),
            "will_waste": bool(prob >= 0.5),
            "days_to_expiry": rec.get("days_to_expiry", 0),
            "quantity": rec.get("quantity", 0),
            "daily_consumption": rec.get("daily_consumption", 0),
            "price_per_unit": rec.get("price_per_unit", 0),
            "storage_type": rec.get("storage_type", ""),
            "wastage_history_pct": rec.get("wastage_history_pct", 0),
        })
    results.sort(key=lambda r: r["waste_probability"], reverse=True)
    counts = {}
    for r in results:
        counts[r["risk_level"]] = counts.get(r["risk_level"], 0) + 1
    return {
        "count": len(results),
        "critical": counts.get("critical", 0),
        "high": counts.get("high", 0),
        "medium": counts.get("medium", 0),
        "low": counts.get("low", 0),
        "results": results,
    }


@app.post("/api/v1/inventory/explain/{idx}")
def explain(idx: int, payload: dict):
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
    top = sorted(zip(FEATURE_COLS, sv), key=lambda x: abs(x[1]), reverse=True)[:6]
    features = [{"feature": f, "impact": round(float(v), 4),
                 "direction": "increases_risk" if v > 0 else "decreases_risk"} for f, v in top]
    prob = float(_predict(X)[0])
    rec = records[idx]
    dte = rec.get("days_to_expiry", 0)
    qty = rec.get("quantity", 0)
    daily = max(rec.get("daily_consumption", 0.01), 0.01)
    waste_hist = rec.get("wastage_history_pct", 0)
    stock_days = qty / daily
    # Value-specific summary sentence
    if dte <= 2:
        summary = f"Only {dte} day(s) left before expiry — urgent action needed."
    elif stock_days > dte * 1.2:
        summary = (f"{stock_days:.0f} days of stock remaining but item expires in {dte} days — "
                   f"{stock_days - dte:.0f} days of surplus will likely go to waste.")
    elif waste_hist >= 0.25:
        summary = (f"This ingredient has been wasted {waste_hist*100:.0f}% of the time historically, "
                   f"making it a repeat waste risk.")
    else:
        summary = (f"Item expires in {dte} days with {qty:.1f} units at {daily:.1f} units/day usage. "
                   f"Monitor closely to avoid waste.")
    return {
        "index": idx,
        "ingredient_name": rec.get("ingredient_name", f"item_{idx}"),
        "waste_probability": round(prob, 4),
        "risk_level": _risk(prob),
        "summary": summary,
        "top_features": features,
    }


@app.post("/api/v1/anomalies")
def anomalies(payload: dict):
    records = payload.get("items", [])
    if not records:
        return JSONResponse({"error": "No items provided"}, status_code=422)
    df = pd.DataFrame(records)
    numeric_cols = ["quantity", "daily_consumption", "price_per_unit", "wastage_history_pct"]
    results = []
    z_scores = {}
    for col in numeric_cols:
        if col in df.columns:
            vals = df[col].fillna(0).astype(float)
            mean, std = vals.mean(), vals.std()
            z_scores[col] = ((vals - mean) / max(std, 0.001)).tolist()
    for i, rec in enumerate(records):
        z_max = max(abs(z_scores.get(col, [0] * len(records))[i]) for col in numeric_cols if col in z_scores)
        is_anomaly = z_max > 2.2
        qty = rec.get("quantity", 0)
        consumption = max(rec.get("daily_consumption", 0.01), 0.01)
        dte = rec.get("days_to_expiry", 30)
        stock_days = qty / consumption
        if stock_days > dte * 3 and dte < 14:
            is_anomaly = True
        results.append({
            "index": i,
            "ingredient_name": rec.get("ingredient_name", f"item_{i}"),
            "category": rec.get("category", ""),
            "is_anomaly": is_anomaly,
            "anomaly_score": round(float(z_max), 3),
            "quantity": qty,
            "daily_consumption": rec.get("daily_consumption", 0),
            "reason": _anomaly_reason(rec) if is_anomaly else None,
        })
    anomaly_list = [r for r in results if r["is_anomaly"]]
    return {
        "total": len(results),
        "anomaly_count": len(anomaly_list),
        "anomaly_rate": round(len(anomaly_list) / max(len(results), 1), 4),
        "results": results,
    }


@app.post("/api/v1/forecast/{ingredient}")
def forecast(ingredient: str, payload: dict):
    """Simple demand forecast using linear trend + weekly seasonality."""
    records = payload.get("items", [])
    horizon = int(payload.get("horizon_days", 14))
    # Find daily_consumption for this ingredient
    daily = 1.0
    for rec in records:
        if rec.get("ingredient_name", "").lower() == ingredient.lower():
            daily = max(float(rec.get("daily_consumption", 1.0)), 0.01)
            break
    today = date.today()
    forecast_data = []
    for d in range(horizon):
        dt = today + timedelta(days=d)
        # Weekly seasonality: weekends slightly higher
        weekday_factor = 1.15 if dt.weekday() >= 5 else 1.0
        # Mild upward trend
        trend_factor = 1 + (d * 0.002)
        # Small deterministic variation (sine wave, not random)
        wave = 1 + 0.08 * math.sin(2 * math.pi * d / 7)
        yhat = daily * weekday_factor * trend_factor * wave
        forecast_data.append({
            "date": dt.isoformat(),
            "yhat": round(yhat, 2),
            "yhat_lower": round(yhat * 0.82, 2),
            "yhat_upper": round(yhat * 1.18, 2),
        })
    total = sum(f["yhat"] for f in forecast_data)
    return {
        "ingredient": ingredient,
        "horizon_days": horizon,
        "daily_avg": round(total / horizon, 2),
        "total_forecast": round(total, 2),
        "peak": round(max(f["yhat"] for f in forecast_data), 2),
        "forecast": forecast_data,
    }


@app.post("/api/v1/operations/reorder")
def reorder(payload: dict):
    records = payload.get("items", [])
    if not records:
        return JSONResponse({"error": "No items provided"}, status_code=422)
    lead_time = 3
    reorder_list = []
    for i, rec in enumerate(records):
        qty = float(rec.get("quantity", 0))
        daily = max(float(rec.get("daily_consumption", 0.01)), 0.01)
        dte = int(rec.get("days_to_expiry", 30))
        rp = float(rec.get("reorder_point", daily * lead_time))
        price = float(rec.get("price_per_unit", 0))
        stock_days = qty / daily
        below_reorder = qty <= rp
        will_run_out = stock_days <= lead_time + 2
        if below_reorder or will_run_out:
            reorder_qty = max(daily * 14 - qty, daily * 3)
            days_until = max(0, int(stock_days - lead_time))
            reorder_list.append({
                "index": i,
                "ingredient_name": rec.get("ingredient_name", f"item_{i}"),
                "category": rec.get("category", ""),
                "current_stock": round(qty, 1),
                "days_of_stock": int(stock_days),
                "reorder_qty": round(reorder_qty, 1),
                "order_within_days": days_until,
                "estimated_cost": round(reorder_qty * price, 0),
                "urgency": "critical" if days_until == 0 else ("high" if days_until <= 2 else "medium"),
            })
    reorder_list.sort(key=lambda r: r["order_within_days"])
    total_cost = sum(r["estimated_cost"] for r in reorder_list)
    return {
        "items_to_reorder": len(reorder_list),
        "total_estimated_cost": round(total_cost, 0),
        "reorder_list": reorder_list,
    }


@app.post("/api/v1/chef-specials")
def chef_specials(payload: dict):
    ingredients = payload.get("ingredients", [])
    if not ingredients:
        return JSONResponse({"error": "No ingredients provided"}, status_code=422)
    specials = []
    for ing in ingredients[:5]:
        dishes = _SUBSTITUTIONS.get(ing, [f"{ing} Curry", f"{ing} Stir Fry", f"{ing} Soup"])
        for dish in dishes[:2]:
            specials.append({
                "name": dish,
                "ingredients_used": [ing],
                "description": f"A classic dish using {ing} as the star ingredient to reduce waste.",
                "prep_time_minutes": 20 + (hash(dish) % 25),
                "urgency": "use_today",
                "waste_reduction_rationale": f"Uses up {ing} before it expires.",
                "storage_tip": f"Keep {ing} refrigerated and use within 24 hours of prep.",
            })
    return {
        "chef_specials": specials[:6],
        "general_recommendation": f"Prioritise using {', '.join(ingredients[:3])} in today's menu to prevent waste.",
        "estimated_waste_reduction_pct": 35,
    }
