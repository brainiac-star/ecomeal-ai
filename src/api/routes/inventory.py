"""Inventory analysis and prediction routes."""

import time
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List

from src.data.validator import BatchInventoryRequest
from src.data.preprocessor import clean_inventory
from src.explainability.explainer import generate_portfolio_summary, explain_item_risk
from src.monitoring.metrics import PREDICTIONS_COUNT, HIGH_RISK_ITEM_COUNT, WASTE_VALUE_AT_RISK
from src.utils.logger import logger
from src.api.state import get_models

router = APIRouter(prefix="/inventory", tags=["Inventory"])


@router.post("/analyze")
async def analyze_inventory(request: BatchInventoryRequest):
    """
    Analyze a batch of inventory items:
    - Clean & validate data
    - Predict waste risk per item
    - Detect anomalies
    - Return ranked risk report
    """
    start = time.perf_counter()
    models = get_models()

    try:
        raw_df = pd.DataFrame([item.model_dump() for item in request.items])
        df, quality_report = clean_inventory(raw_df)

        # Compute missing derived features if absent
        if "days_to_expiry" not in df.columns:
            from datetime import date
            df["expiry_date"] = pd.to_datetime(df.get("expiry_date", pd.Timestamp.today()))
            df["days_to_expiry"] = (df["expiry_date"] - pd.Timestamp(date.today())).dt.days.clip(lower=0)

        # Wastage predictions
        predictions_df = models["wastage"].predict(df)
        df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)

        # SHAP explanations for top risky items
        top_risk = df.nlargest(min(10, len(df)), "waste_probability")
        shap_explanations = models["wastage"].explain(top_risk, max_items=min(5, len(top_risk)))

        # Anomaly detection
        anomaly_df = models["anomaly"].detect(df)
        anomalies = anomaly_df[anomaly_df["is_anomaly"]][
            ["item_id", "ingredient_name", "anomaly_score", "anomaly_reason"]
        ].to_dict("records") if "is_anomaly" in anomaly_df.columns else []

        # Portfolio summary
        summary = generate_portfolio_summary(df)

        # Update Prometheus metrics
        for _, row in df.iterrows():
            PREDICTIONS_COUNT.labels(risk_level=row.get("risk_level_pred", "unknown")).inc()
        HIGH_RISK_ITEM_COUNT.set(summary.get("high_risk_count", 0) + summary.get("critical_count", 0))
        WASTE_VALUE_AT_RISK.set(summary.get("waste_value_at_risk_inr", 0))

        # Build item-level response
        result_records = df[[
            "item_id", "ingredient_name", "category", "quantity", "unit",
            "days_to_expiry", "waste_probability", "risk_level_pred",
            "potential_waste_value", "overstock_flag",
        ]].rename(columns={"risk_level_pred": "risk_level"})
        result_records = result_records.sort_values("waste_probability", ascending=False)

        elapsed = round(time.perf_counter() - start, 3)
        logger.info(
            f"Analyzed {len(df)} items in {elapsed}s, "
            f"{summary['critical_count']} critical, {summary['high_risk_count']} high risk"
        )

        return {
            "status": "ok",
            "restaurant": request.restaurant_name,
            "processing_time_s": elapsed,
            "data_quality": quality_report,
            "portfolio_summary": summary,
            "items": result_records.to_dict("records"),
            "anomalies": anomalies,
            "shap_explanations": shap_explanations,
        }

    except Exception as e:
        logger.error(f"Inventory analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/explain/{item_index}")
async def explain_item(
    request: BatchInventoryRequest,
    item_index: int = 0,
):
    """Return detailed explanation for a specific item's risk level."""
    models = get_models()
    if item_index >= len(request.items):
        raise HTTPException(status_code=400, detail="item_index out of range")

    raw_df = pd.DataFrame([request.items[item_index].model_dump()])
    df, _ = clean_inventory(raw_df)
    predictions = models["wastage"].predict(df)
    shap_exp = models["wastage"].explain(df, max_items=1)

    row = pd.concat([df.iloc[0], predictions.iloc[0]])
    explanation = explain_item_risk(row, shap_exp[0]["top_features"] if shap_exp else None)

    return {"item_index": item_index, "explanation": explanation}


@router.get("/risk-levels")
async def get_risk_levels():
    """Return the definition of each risk level."""
    return {
        "risk_levels": {
            "critical": {"threshold": 0.70, "action": "Use immediately or discard", "color": "#FF0000"},
            "high": {"threshold": 0.45, "action": "Use within 2 days", "color": "#FF6600"},
            "medium": {"threshold": 0.20, "action": "Plan usage this week", "color": "#FFAA00"},
            "low": {"threshold": 0.0, "action": "Normal operations", "color": "#00AA00"},
        }
    }
