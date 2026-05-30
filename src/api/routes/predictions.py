"""Demand forecasting and overstock detection routes."""

import pandas as pd
from fastapi import APIRouter, HTTPException
from typing import List, Optional

from src.data.validator import DemandForecastRequest
from src.utils.logger import logger
from src.api.state import get_models

router = APIRouter(prefix="/forecast", tags=["Demand Forecasting"])


@router.post("/demand")
async def forecast_demand(request: DemandForecastRequest):
    """
    Forecast future demand for a specific ingredient.
    Returns daily predictions with confidence intervals.
    """
    models = get_models()
    forecaster = models["forecaster"]

    if not forecaster.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Demand forecaster not fitted yet. Run /train endpoint first.",
        )

    try:
        fc = forecaster.forecast(request.ingredient_name, request.horizon_days)
        records = fc.to_dict("records")
        # Make dates JSON-serializable
        for r in records:
            if hasattr(r.get("ds"), "isoformat"):
                r["ds"] = r["ds"].isoformat()

        return {
            "ingredient": request.ingredient_name,
            "horizon_days": request.horizon_days,
            "forecast": records,
            "has_trained_model": request.ingredient_name in forecaster.models,
        }
    except Exception as e:
        logger.error(f"Demand forecast failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/overstock")
async def detect_overstock(items: List[dict]):
    """
    Given current inventory, identify overstock and shortage risks
    using demand forecast comparison.
    """
    models = get_models()
    if not items:
        raise HTTPException(status_code=400, detail="Empty inventory list")

    try:
        df = pd.DataFrame(items)
        result_df = models["forecaster"].detect_overstock(df, horizon_days=14)
        return {
            "overstock_items": result_df[result_df["overstock_risk"]].to_dict("records"),
            "shortage_items": result_df[result_df["shortage_risk"]].to_dict("records"),
            "all_items": result_df.to_dict("records"),
        }
    except Exception as e:
        logger.error(f"Overstock detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ingredients")
async def list_forecasted_ingredients():
    """Return list of ingredients with trained demand models."""
    models = get_models()
    return {
        "ingredients": list(models["forecaster"].models.keys()),
        "count": len(models["forecaster"].models),
    }
