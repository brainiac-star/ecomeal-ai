# 🥗 Ecomeal AI — Food Waste Intelligence Platform

AI-powered food waste prediction and recommendation system for restaurants. Predicts waste risk, forecasts ingredient demand, detects inventory anomalies, and generates Claude AI–powered Chef Specials for expiring ingredients.

---

## Architecture Overview

```
ecomeal-ai/
├── src/
│   ├── data/
│   │   ├── generator.py        # Synthetic 1200+ record dataset with dirty injection
│   │   ├── preprocessor.py     # Robust cleaning + feature engineering
│   │   └── validator.py        # Pydantic input validation (XSS-safe)
│   ├── models/
│   │   ├── wastage_predictor.py  # XGBoost + LightGBM ensemble
│   │   ├── demand_forecaster.py  # Facebook Prophet per-ingredient
│   │   └── anomaly_detector.py   # Isolation Forest
│   ├── recommendations/
│   │   ├── chef_specials.py    # Claude API + fallback
│   │   └── rag_engine.py       # FAISS + SentenceTransformer RAG
│   ├── explainability/
│   │   └── explainer.py        # SHAP + rule-based explanations
│   ├── api/
│   │   ├── main.py             # FastAPI app + lifespan model loading
│   │   ├── state.py            # Global model registry
│   │   └── routes/             # inventory / predictions / recommendations
│   ├── dashboard/
│   │   └── app.py              # Streamlit interactive dashboard
│   ├── monitoring/
│   │   └── metrics.py          # Prometheus counters / gauges / histograms
│   ├── utils/
│   │   ├── config.py           # Pydantic settings
│   │   └── logger.py           # Loguru structured logging
│   └── pipeline.py             # End-to-end training + inference pipeline
├── tests/                      # pytest suite
├── scripts/train.py            # CLI training entrypoint
├── docker-compose.yml          # API + Dashboard + Redis
├── Dockerfile
└── requirements.txt
```

---

## Dataset Approach

**Synthetic simulation** (`src/data/generator.py`) generates 1200+ realistic inventory records covering:

| Field | Details |
|---|---|
| `ingredient_name` | 60+ ingredients across 8 categories |
| `category` | vegetables, dairy, meat_seafood, grains_pulses, spices, oils, beverages, bakery |
| `quantity` | Randomized relative to consumption + shelf life (lognormal noise) |
| `expiry_date` | Derived from shelf life (range 1–730 days per category) |
| `daily_consumption` | Per-ingredient base rate with ±20% lognormal noise |
| `storage_type` | frozen / refrigerated / ambient (category-appropriate) |
| `wastage_history_pct` | Category-calibrated historical waste rate |
| `supplier`, `restaurant` | Randomly drawn from realistic Indian F&B names |

**Intentional dirty data** (~5% of records):
- Null quantities
- Zero daily consumption
- Invalid date strings (`"INVALID"`)
- Negative prices

The preprocessor handles all of these without crashing.

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

**Why not neural nets?** The tabular dataset (~1200 rows) doesn't justify the complexity. Tree ensembles outperform MLPs on structured tabular data at this scale. Prophet paper (Taylor & Letham, 2018) confirms this for non-temporal data.

**Class imbalance handling:** `scale_pos_weight` (XGB) and `class_weight='balanced'` (LGB) address the imbalance between low-risk and high-risk items.

**Target:** Binary classification — `will_waste` (1 if `waste_risk_score ≥ 0.45`). The score itself combines expiry proximity, stock-vs-demand ratio, and historical waste rate using a domain rule.

### Demand Forecaster (Facebook Prophet)

**Why Prophet?** Handles weekly seasonality (restaurant demand spikes on weekends), trend, and missing data gracefully. Requires no manual stationarity testing. 14-day horizon is well within its reliable range.

**Fallback:** Linear trend model (numpy polyfit) when Prophet is unavailable or data is sparse (<7 points).

### Anomaly Detector (Isolation Forest)

**Why Isolation Forest?** Unsupervised, scales well, and is interpretable in terms of anomaly score. Contamination rate set dynamically (1–5%) based on dataset size. Better for inventory anomalies than one-class SVM (no hyperplane tuning needed).

---

## AI Integration

### Claude API (Chef Specials)
- Uses `claude-sonnet-4-6` with **prompt caching** on the system prompt to reduce latency and cost
- Structured JSON output with schema validation
- Retry logic via `tenacity` (3 attempts, exponential backoff)
- **Fallback**: Template-based suggestions when API key is absent

### RAG (Retrieval-Augmented Generation)
- 25-entry knowledge base of ingredient storage tips, dish ideas, waste strategies
- `all-MiniLM-L6-v2` SentenceTransformer embeddings
- FAISS `IndexFlatL2` for fast L2 similarity search
- Top-k knowledge retrieved and passed as context to Claude

---

## Explainability

Three layers of explanation:

1. **SHAP (TreeExplainer)** — Per-item feature impact values showing which features drove the prediction
2. **Rule-based** — Human-readable reasons based on domain logic (e.g., "Overstock: 12.3 days of supply but only 3 days until expiry")
3. **Portfolio summary** — Fleet-level risk distribution, top-at-risk items, waste value at risk

---

## Failure Handling

| Failure Mode | Handling |
|---|---|
| Null / NaN values | Filled with domain defaults (quantity→0, consumption→0.01) |
| Negative quantities/prices | Replaced with defaults |
| Invalid date strings | Replaced with `today` |
| Division by zero | `daily_consumption` floored at 0.01 |
| Claude API timeout/error | 3-retry with backoff, then rule-based fallback |
| Models not found on disk | Automatic retraining on synthetic data at startup |
| Empty dataset input | Returns empty DataFrame, never raises |
| Malformed API request | Pydantic validation returns 422 with field-level errors |
| Missing ML features | Filled with 0 before inference |

---

## Scalability Considerations

- **Preprocessing**: vectorized pandas operations, no row-level Python loops
- **ML inference**: XGBoost/LGB batch prediction — 1000 items processes in <200ms
- **FAISS**: In-memory index, O(n) build, O(log n) search
- **API**: FastAPI async with GZip middleware; rate-limited (60 req/min default)
- **Caching**: `@st.cache_resource` for models, `@st.cache_data` for dataset
- **Docker**: Multi-service compose with Redis (ready for response caching)
- **Future**: Replace FAISS with Pinecone/Weaviate for distributed search at scale

---

## Tradeoffs Made

| Decision | Chosen | Alternative | Reason |
|---|---|---|---|
| Forecasting | Prophet | LSTM/ARIMA | Prophet needs less data, handles missing dates, interpretable |
| ML | XGB+LGB | Random Forest, CatBoost | Lower latency, better on this tabular schema |
| Embeddings | MiniLM-L6 | OpenAI text-embedding | No API cost, runs locally, fast |
| Storage | FAISS flat | Chroma/Pinecone | Zero infra for demo; swap at scale |
| API framework | FastAPI | Flask/Django | Async, auto-docs, Pydantic native |
| Monitoring | Prometheus | DataDog/New Relic | Free, self-hosted, standard |

---

## Assumptions

1. **Demand is relatively stable** within a 14-day horizon. Sudden events (festivals, shutdowns) are out-of-scope.
2. **`daily_consumption`** is an accurate running average (not point-in-time). Real systems would compute this from POS transaction history.
3. **Waste risk threshold of 0.45** for "high risk" is domain-calibrated. Real deployment would tune this via restaurant operator feedback.
4. **Synthetic data reflects Indian restaurant context** (ingredient names, prices in INR, supplier names). Internationalization is a future task.
5. **`wastage_history_pct`** is per-ingredient, not per-restaurant. A multi-tenant system would track this at restaurant-ingredient grain.

---

## Future Improvements

- [ ] **Real POS integration** — replace synthetic data with actual order history
- [ ] **Multi-tenant model** — per-restaurant demand models with transfer learning
- [ ] **Online learning** — incremental model updates as new waste events are recorded
- [ ] **Computer vision** — camera-based portion tracking to auto-update `daily_consumption`
- [ ] **Supplier integration** — auto-generate purchase orders based on shortage predictions
- [ ] **LLM routing** — use smaller Claude Haiku for simple queries, Sonnet for complex specials
- [ ] **Notification system** — push alerts when items enter critical zone
- [ ] **A/B testing framework** — evaluate impact of Chef Specials on actual waste reduction

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-org/ecomeal-ai
cd ecomeal-ai
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Train models
python scripts/train.py --records 1200

# 4a. Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 4b. Start Dashboard (separate terminal)
streamlit run src/dashboard/app.py

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
| `GROQ_API_KEY` | — | **Recommended** — free at console.groq.com, uses llama3-8b |
| `GEMINI_API_KEY` | — | Google Gemini Flash — free at aistudio.google.com |
| `ANTHROPIC_API_KEY` | — | Anthropic Claude (paid) |
| `OLLAMA_MODEL` | `llama3` | Ollama local model — no key needed, just run Ollama |
| `APP_ENV` | `development` | Environment (development/production) |
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
| AI / LLM | Anthropic Claude (claude-sonnet-4-6) |
| RAG | FAISS, SentenceTransformers |
| API | FastAPI, Pydantic, uvicorn |
| Dashboard | Streamlit, Plotly |
| Monitoring | Prometheus, Loguru |
| Infrastructure | Docker, Redis |
| Testing | pytest, pytest-asyncio |
