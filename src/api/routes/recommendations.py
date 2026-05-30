"""Chef Specials and inventory recommendation routes."""

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.data.validator import ChefSpecialsRequest
from src.monitoring.metrics import CHEF_SPECIALS_COUNT
from src.utils.logger import logger
from src.api.state import get_models

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


@router.post("/chef-specials")
async def generate_chef_specials(request: ChefSpecialsRequest):
    """
    Generate AI-powered Chef Specials for expiring ingredients.
    Uses the configured LLM provider with RAG-augmented context.
    """
    models = get_models()
    chef_engine = models["chef"]
    rag_engine = models["rag"]

    try:
        # Retrieve knowledge context for the ingredients
        knowledge = rag_engine.get_ingredient_knowledge(request.ingredients, top_k_per_ingredient=2)
        context_str = " | ".join(k["text"] for k in knowledge[:5]) if knowledge else None

        result = chef_engine.generate(
            ingredients=request.ingredients,
            cuisine_preference=request.cuisine_preference,
            dietary_restrictions=request.dietary_restrictions,
            n_suggestions=request.n_suggestions,
            context=context_str,
        )

        source = "fallback" if result.get("_source") == "fallback" else "llm"
        CHEF_SPECIALS_COUNT.labels(source=source).inc()

        return {
            **result,
            "knowledge_context": [k["text"] for k in knowledge[:3]],
        }
    except Exception as e:
        logger.error(f"Chef specials generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inventory-advice")
async def get_inventory_advice(payload: dict):
    """
    Generate strategic inventory advice based on risk analysis results.
    """
    models = get_models()
    high_risk = payload.get("high_risk_items", [])
    overstock = payload.get("overstock_items", [])

    if not high_risk and not overstock:
        return {"advice": "No high-risk or overstock items detected. Inventory looks healthy!"}

    try:
        advice = models["chef"].generate_inventory_advice(high_risk, overstock)
        return {"advice": advice}
    except Exception as e:
        logger.error(f"Inventory advice failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag-search")
async def rag_search(payload: dict):
    """Semantic search over ingredient knowledge base."""
    query = payload.get("query", "")
    top_k = min(int(payload.get("top_k", 5)), 10)
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    models = get_models()
    results = models["rag"].retrieve(query, top_k=top_k)
    return {"query": query, "results": results}
