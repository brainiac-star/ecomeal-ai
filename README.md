# 🥗 Ecomeal AI — Food Waste Intelligence Platform

AI-powered food waste prediction and recommendation system for restaurants. Predicts waste risk, forecasts ingredient demand, detects inventory anomalies, and generates AI-powered Chef Specials for expiring ingredients.

**🚀 Live Demo:** [https://ecomeal-ai-xlpkjjudmoxh5zwwegnuhp.streamlit.app/](https://ecomeal-ai-xlpkjjudmoxh5zwwegnuhp.streamlit.app/)

---

## Architecture Overview

```
ecomeal-ai/
├── src/
│   ├── data/
│   │   ├── generator.py        # Synthetic 1200+ record dataset with restaurant profiles
│   │   ├── preprocessor.py     # Robust cleaning + feature engineering
│   │   └── validator.py        # Pydantic input validation (XSS-safe)
│   ├── models/
│   │   ├── wastage_predictor.py  # XGBoost + LightGBM ensemble
│   │   ├── demand_forecaster.py  # Facebook Prophet per-ingredient
│   │   └── anomaly_detector.py   # Isolation Forest
│   ├── recommendations/
│   │   ├── chef_specials.py    # Multi-provider LLM + rule-based fallback
│   │   └── rag_engine.py       # FAISS + SentenceTransformer RAG
│   ├── explainability/
│   │   └── explainer.py        # SHAP + plain-English explanations
│   ├── api/
│   │   ├── main.py             # FastAPI app + lifespan model loading
│   │   ├── state.py            # Global model registry
│   │   └── routes/             # inventory / predictions / recommendations
│   ├── dashboard/
│   │   └── app.py              # Streamlit interactive dashboard (8 tabs)
│   ├── monitoring/
│   │   └── metrics.py          # Prometheus counters / gauges / histograms
│   ├── utils/
│   │   ├── config.py           # Pydantic settings
│   │   └── logger.py           # Loguru structured logging
│   └── pipeline.py             # End-to-end training + inference pipeline
├── tests/                      # 29 pytest tests
├── scripts/train.py            # CLI training entrypoint
├── docker-compose.yml          # API + Dashboard + Redis
├── Dockerfile
└── requirements.txt
```

---

## Dataset Approach

**Synthetic simulation** (`src/data/generator.py`) generates 1200+ realistic inventory records with per-restaurant management profiles and category-specific ordering behaviour:

| Field | Details |
|---|---|
| `ingredient_name` | 60+ ingredients across 8 categories |
| `category` | vegetables, dairy, meat_seafood, grains_pulses, spices, oils, beverages, bakery |
| `quantity` | Modelled as ordered batch minus consumed — realistic stock depletion curve |
| `expiry_date` | Derived from shelf life (range 1–730 days per category) |
| `daily_consumption` | Per-ingredient base rate with lognormal noise |
| `storage_type` | frozen / refrigerated / ambient (category-appropriate) |
| `wastage_history_pct` | Correlated with restaurant management quality and storage type |
| `supplier`, `restaurant` | Restaurant-preferred supplier assignment (not purely random) |

**Restaurant profiles** drive realistic variation — well-managed restaurants order more frequently, overstock less, and have lower waste rates. Poorly-managed ones over-order and waste more.

**Risk distribution target:** ~8% critical · ~15% high · ~30% medium · ~47% low

**Intentional dirty data** (~5% of records): null quantities, zero daily consumption, invalid date strings, negative prices — the preprocessor handles all without crashing.

---

## Feature Engineering

`src/data/preprocessor.py` derives:

| Feature | Formula | Why |
|---|---|---|
| `stock_days_available` | `quantity / daily_consumption` | Days of stock at current rate |
| `stock_expiry_ratio` | `stock_days / days_to_expiry` | >1 = overstock risk |
| `shelf_life_consumed_pct` | `days_since_purchase / total_shelf_life` | How "old" the item is |
| `potential_waste_value` | `quantity × waste_pct × price` | Financial risk |
| `overstock_flag` | `stock_days > expiry * 1.2` | Boolean overstock signal |
| `below_reorder_point` | `quantity < reorder_point` | Shortage signal |
| `storage_type_enc` | Ordinal: frozen=0, refrigerated=1, ambient=2 | For ML |

---

## Model Selection Reasoning

### Wastage Predictor (XGBoost + LightGBM Ensemble)

**Why ensemble?** XGBoost handles feature interactions well; LightGBM is faster on large data. Weighted average (55% XGB, 45% LGB) reduces variance.

**Why not neural nets?** The tabular dataset (~1200 rows) doesn't justify the complexity. Tree ensembles consistently outperform MLPs on structured tabular data at this scale.

**Class imbalance handling:** `scale_pos_weight` (XGB) and `class_weight='balanced'` (LGB).

**Target:** Binary classification — `will_waste`. The label is derived from a rule-based waste risk score (expiry proximity + stock-vs-demand + historical waste rate) with 15% random label noise injected to simulate real-world uncertainty and prevent perfect overfitting.

### Demand Forecaster (Facebook Prophet)

**Why Prophet?** Handles weekly seasonality, trend, and missing data gracefully. 14-day horizon is well within its reliable range. Linear trend fallback when data is sparse (<7 points).

### Anomaly Detector (Isolation Forest)

**Why Isolation Forest?** Unsupervised, no labelled anomaly data needed. Contamination rate set at 5%. Better than one-class SVM for inventory anomalies — no hyperplane tuning required.

---

## AI Integration

### Chef Specials (Multi-provider LLM)

Provider priority: **Groq** (free, llama-3.1-8b-instant) → **Gemini Flash** (free) → **Hosted LLM** → **Ollama** (local) → rule-based fallback

- Structured JSON output with schema enforcement
- RAG context injected from 25-entry knowledge base
- Retry logic via `tenacity` (3 attempts, exponential backoff)

### RAG (Retrieval-Augmented Generation)

- `all-MiniLM-L6-v2` SentenceTransformer embeddings
- FAISS `IndexFlatL2` for fast similarity search
- Top-k knowledge retrieved per ingredient and passed as LLM context

---

## Explainability

Three layers:

1. **SHAP (TreeExplainer on LightGBM)** — per-item feature impact values
2. **Natural language sentences** — each SHAP factor translated to plain English (e.g. *"Very few days remain before this item expires, pushing waste risk up."*)
3. **Rule-based summary** — domain logic sentences always visible regardless of SHAP availability

---

## Dashboard Tabs

| Tab | What it shows |
|---|---|
| 🔍 Risk Analysis | Waste risk distribution, top at-risk items, category breakdown |
| 📈 Demand Forecast | Prophet forecast with confidence interval, overstock/shortage detection |
| ⚠️ Anomalies | Isolation Forest flags, anomaly scatter plot |
| 👨‍🍳 Chef Specials | LLM-generated dishes for expiring ingredients |
| 🔬 Explainability | Feature importance, plain-English SHAP explanations per item |
| 🛒 Operations | Reorder suggestions, waste cost tracker, ingredient substitution |
| 📊 Insights | Multi-restaurant comparison, 30-day waste risk trend by category |
| 🗄️ Data | Full inventory table with CSV export |

---

## Failure Handling

| Failure Mode | Handling |
|---|---|
| Null / NaN values | Filled with domain defaults |
| Negative quantities/prices | Replaced with defaults |
| Invalid date strings | Replaced with `today` |
| Division by zero | `daily_consumption` floored at 0.01 |
| LLM API timeout/error | 3-retry with backoff, then rule-based fallback |
| Models not found on disk | Automatic retraining on synthetic data at startup |
| Empty dataset input | Returns empty DataFrame, never raises |
| Malformed API request | Pydantic 422 with field-level errors |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/brainiac-star/ecomeal-ai
cd ecomeal-ai
pip install -r requirements.txt

# 2. Configure environment (optional — app works without any API key)
cp .env.example .env
# Add GROQ_API_KEY for free AI-generated Chef Specials (console.groq.com)

# 3. Train models
python scripts/train.py --records 1200

# 4a. Start Dashboard
streamlit run src/dashboard/app.py

# 4b. Start API (separate terminal)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Run tests
pytest tests/ -v

# Docker (all-in-one)
docker-compose up --build
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System health + model status |
| GET | `/metrics` | Prometheus metrics |
| POST | `/api/v1/train` | Retrain all models |
| POST | `/api/v1/inventory/analyze` | Batch waste risk analysis |
| POST | `/api/v1/inventory/explain/{idx}` | SHAP explanation for item |
| POST | `/api/v1/forecast/demand` | Demand forecast for ingredient |
| POST | `/api/v1/forecast/overstock` | Overstock/shortage detection |
| POST | `/api/v1/recommendations/chef-specials` | AI Chef Specials |
| POST | `/api/v1/recommendations/inventory-advice` | Strategic recommendations |
| POST | `/api/v1/recommendations/rag-search` | Semantic knowledge search |

Full interactive docs at `/docs` (Swagger UI).

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Recommended** — free at console.groq.com, uses llama-3.1-8b-instant |
| `GEMINI_API_KEY` | — | Google Gemini Flash — free at aistudio.google.com |
| `LLM_API_KEY` | — | Optional hosted LLM provider key |
| `OLLAMA_MODEL` | `llama3` | Ollama local model — no key needed |
| `APP_ENV` | `development` | Environment |
| `DATASET_SIZE` | `1200` | Synthetic dataset size |
| `LOG_LEVEL` | `INFO` | Logging level |
| `RATE_LIMIT_PER_MINUTE` | `60` | API rate limit |
| `MODEL_DIR` | `data/models` | Saved model path |

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost, LightGBM, scikit-learn, SHAP |
| Forecasting | Facebook Prophet, statsmodels |
| LLM | Groq (llama-3.1-8b-instant), Gemini Flash, Ollama |
| RAG | FAISS, SentenceTransformers |
| API | FastAPI, Pydantic, uvicorn |
| Dashboard | Streamlit, Plotly |
| Monitoring | Prometheus, Loguru |
| Infrastructure | Docker, Redis |
| Testing | pytest, pytest-asyncio |
