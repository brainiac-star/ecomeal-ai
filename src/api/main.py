"""
Ecomeal AI — FastAPI application entry point.
Loads all ML models on startup, wires routes, and exposes health/metrics endpoints.
"""

import time
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.utils.logger import logger
from src.utils.config import get_settings
from src.api.state import set_models
from src.api.routes import inventory, predictions, recommendations
from src.monitoring.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, MODEL_LOAD_SUCCESS, metrics_endpoint
)

settings = get_settings()

# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.rate_limit_per_minute}/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    logger.info("🚀 Starting Ecomeal AI API...")

    from src.models.wastage_predictor import WastagePredictor
    from src.models.demand_forecaster import DemandForecaster
    from src.models.anomaly_detector import InventoryAnomalyDetector
    from src.recommendations.chef_specials import ChefSpecialsEngine
    from src.recommendations.rag_engine import RAGRecommendationEngine
    from src.data.generator import generate_inventory_dataset, generate_demand_history
    from src.data.preprocessor import clean_inventory, encode_categoricals

    wastage = WastagePredictor()
    forecaster = DemandForecaster()
    anomaly = InventoryAnomalyDetector()
    chef = ChefSpecialsEngine()
    rag = RAGRecommendationEngine()

    # Try loading saved models first
    model_loaded = wastage.load() and anomaly.load()
    forecaster_loaded = forecaster.load()
    rag.load() or rag.build_index()

    # If no saved models, train on synthetic data
    if not model_loaded:
        logger.info("No saved models found — training on synthetic data...")
        df_raw = generate_inventory_dataset(n_records=settings.dataset_size, seed=settings.random_seed)
        df, _ = clean_inventory(df_raw)
        df = encode_categoricals(df)

        wastage.train(df)
        wastage.save()

        anomaly.fit(df)
        anomaly.save()

        MODEL_LOAD_SUCCESS.labels(model_name="wastage_predictor").set(1)
        MODEL_LOAD_SUCCESS.labels(model_name="anomaly_detector").set(1)
    else:
        MODEL_LOAD_SUCCESS.labels(model_name="wastage_predictor").set(1)
        MODEL_LOAD_SUCCESS.labels(model_name="anomaly_detector").set(1)

    if not forecaster_loaded:
        logger.info("Training demand forecasters on synthetic data...")
        df_raw = generate_inventory_dataset(n_records=settings.dataset_size, seed=settings.random_seed)
        df, _ = clean_inventory(df_raw)
        ingredients = df["ingredient_name"].unique().tolist()[:30]
        demand_df = generate_demand_history(ingredients=ingredients, n_days=180, seed=settings.random_seed)
        forecaster.fit(demand_df)
        forecaster.save()
        MODEL_LOAD_SUCCESS.labels(model_name="demand_forecaster").set(1)
    else:
        MODEL_LOAD_SUCCESS.labels(model_name="demand_forecaster").set(1)

    set_models({
        "wastage": wastage,
        "forecaster": forecaster,
        "anomaly": anomaly,
        "chef": chef,
        "rag": rag,
    })

    logger.info("✅ All models ready. API is live.")
    yield

    logger.info("Shutting down Ecomeal AI API...")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ecomeal AI",
    description=(
        "AI-powered food waste prediction and recommendation system for restaurants. "
        "Predicts waste risk, forecasts demand, detects anomalies, and generates Chef Specials."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=500)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(elapsed)
    return response


# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(inventory.router, prefix="/api/v1")
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(recommendations.router, prefix="/api/v1")


# ── Core endpoints ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    from src.api.state import get_models
    models = get_models()
    return {
        "status": "healthy",
        "models_loaded": {name: True for name in models},
        "version": "1.0.0",
        "env": settings.app_env,
    }


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    return metrics_endpoint()


@app.post("/api/v1/train", tags=["System"])
async def retrain_models():
    """Trigger model retraining on fresh synthetic data."""
    from src.api.state import get_models, set_models
    from src.models.wastage_predictor import WastagePredictor
    from src.models.anomaly_detector import InventoryAnomalyDetector
    from src.data.generator import generate_inventory_dataset
    from src.data.preprocessor import clean_inventory, encode_categoricals

    logger.info("Manual retraining triggered")
    df_raw = generate_inventory_dataset(n_records=settings.dataset_size, seed=settings.random_seed)
    df, quality = clean_inventory(df_raw)
    df = encode_categoricals(df)

    models = get_models()
    wastage_metrics = models["wastage"].train(df)
    models["wastage"].save()
    models["anomaly"].fit(df)
    models["anomaly"].save()

    return {
        "status": "retrained",
        "records_used": len(df),
        "data_quality": quality,
        "wastage_model_metrics": wastage_metrics,
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Ecomeal AI API — Food Waste Intelligence Platform",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0",
    }


def start():
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.app_env == "development",
    )


if __name__ == "__main__":
    start()
