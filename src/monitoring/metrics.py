"""
Prometheus metrics for the Ecomeal API.
Tracks request counts, latency, prediction quality, and waste metrics.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# ── Request metrics ────────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "ecomeal_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "ecomeal_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ── Prediction metrics ─────────────────────────────────────────────────────────
PREDICTIONS_COUNT = Counter(
    "ecomeal_predictions_total",
    "Total waste predictions made",
    ["risk_level"],
)

CHEF_SPECIALS_COUNT = Counter(
    "ecomeal_chef_specials_total",
    "Total chef specials generated",
    ["source"],  # llm | fallback
)

ANOMALIES_DETECTED = Counter(
    "ecomeal_anomalies_detected_total",
    "Total inventory anomalies detected",
)

# ── Business metrics ───────────────────────────────────────────────────────────
WASTE_VALUE_AT_RISK = Gauge(
    "ecomeal_waste_value_at_risk_inr",
    "Estimated monetary value at risk of wastage (INR)",
)

HIGH_RISK_ITEM_COUNT = Gauge(
    "ecomeal_high_risk_items",
    "Number of high/critical risk inventory items",
)

CRITICAL_EXPIRY_COUNT = Gauge(
    "ecomeal_items_expiring_today",
    "Number of items expiring within 24 hours",
)

# ── Model metrics ──────────────────────────────────────────────────────────────
MODEL_LOAD_SUCCESS = Gauge(
    "ecomeal_model_loaded",
    "Whether ML models are loaded successfully",
    ["model_name"],
)

FORECAST_HORIZON = Gauge(
    "ecomeal_forecast_horizon_days",
    "Current demand forecast horizon in days",
)


def metrics_endpoint() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
