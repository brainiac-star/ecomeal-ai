"""
Ecomeal AI — Streamlit Dashboard
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import get_settings
settings = get_settings()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ecomeal AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0f1117; }

/* Header */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #2d3561;
}
.hero h1 { color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0; }
.hero p  { color: #a0aec0; margin: 0.4rem 0 0; font-size: 1rem; }

/* KPI Cards */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 140px;
    background: #1a1d2e;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    border-left: 4px solid;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.kpi-card.red   { border-color: #fc5c7d; }
.kpi-card.orange{ border-color: #f7971e; }
.kpi-card.green { border-color: #56ab2f; }
.kpi-card.blue  { border-color: #4facfe; }
.kpi-card.purple{ border-color: #a855f7; }
.kpi-label { color: #718096; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { color: #ffffff; font-size: 1.8rem; font-weight: 700; line-height: 1.2; margin-top: 0.3rem; }
.kpi-sub   { color: #718096; font-size: 0.75rem; margin-top: 0.2rem; }

/* Risk badges */
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-critical { background: #fde8e8; color: #c81e1e; }
.badge-high     { background: #fef3c7; color: #d97706; }
.badge-medium   { background: #fff3cd; color: #856404; }
.badge-low      { background: #d1fae5; color: #065f46; }

/* Section headers */
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
    padding: 0.5rem 0; border-bottom: 2px solid #2d3748;
    margin-bottom: 1rem;
}

/* Chef Special cards */
.dish-card {
    background: #1a1d2e;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    border: 1px solid #2d3748;
}
.dish-name { color: #fff; font-size: 1.1rem; font-weight: 700; }
.dish-desc { color: #a0aec0; font-size: 0.9rem; margin-top: 0.4rem; }
.dish-meta { color: #718096; font-size: 0.8rem; margin-top: 0.6rem; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #1a1d2e; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: #a0aec0; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background: #1a1d2e; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #718096; font-weight: 600; border-radius: 8px; }
.stTabs [aria-selected="true"] { background: #2d3748 !important; color: #fff !important; }

/* Dataframe */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* NL explanation card */
.nl-card {
    background: #1e2a3a;
    border-left: 4px solid #4facfe;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin: 0.8rem 0;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.5;
}
.nl-card .nl-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4facfe;
    margin-bottom: 0.3rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Info/warning/success boxes */
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── SHAP factor deduplication (mirrors wastage_predictor._deduplicate_factors) ─
def _deduplicate_factors(top_features):
    _SEMANTIC_GROUPS = [
        {"stock_days_available", "stock_expiry_ratio", "overstock_flag"},
        {"days_to_expiry", "shelf_life_consumed_pct", "days_since_purchase"},
        {"quantity", "potential_waste_value"},
        {"daily_consumption"},
    ]
    seen_groups = set()
    risk_drivers, safe_drivers = [], []
    for f in top_features:
        feat = f["feature"]
        direction = f["direction"]
        group_id = next((i for i, g in enumerate(_SEMANTIC_GROUPS) if feat in g), None)
        if group_id is not None:
            conflict = (group_id, "decreases_risk" if direction == "increases_risk" else "increases_risk")
            if conflict in seen_groups:
                continue
            seen_groups.add((group_id, direction))
        (risk_drivers if direction == "increases_risk" else safe_drivers).append(f)
    return risk_drivers, safe_drivers


# ── Plain-English explanation helpers ─────────────────────────────────────────
def _nl_explain_row(row: "pd.Series") -> str:
    dte = int(row.get("days_to_expiry", 0))
    qty = float(row.get("quantity", 0))
    daily = float(row.get("daily_consumption", 0.01))
    shelf_pct = float(row.get("shelf_life_consumed_pct", 0))
    waste_hist = float(row.get("wastage_history_pct", 0))
    stock_days = qty / max(daily, 0.01)
    overstock = stock_days > dte * 1.2

    if dte == 0:
        return "This item expires today — it must be used or discarded immediately."
    if dte <= 2:
        return f"Only {dte} day(s) left before this expires and there is more stock than can be consumed in time — urgent use is needed."
    if overstock and dte <= 10:
        return (
            f"There are {stock_days:.0f} days worth of stock but only {dte} days until expiry, "
            f"meaning a significant portion will likely go to waste."
        )
    if shelf_pct >= 0.8:
        return f"{shelf_pct*100:.0f}% of this item's shelf life has already elapsed — it needs to be prioritised in the next few days."
    if waste_hist >= 0.25:
        return (
            f"This ingredient has been wasted {waste_hist*100:.0f}% of the time historically, "
            f"making it a repeat problem worth addressing in menu planning."
        )
    if overstock:
        return (
            f"Current stock ({qty:.1f} units) will last {stock_days:.0f} days at today's usage rate, "
            f"but the item expires in {dte} days — consider increasing usage or reducing the next order."
        )
    return (
        f"This item expires in {dte} days with {qty:.1f} units remaining. "
        f"At the current usage rate it should be consumed in time, but close monitoring is advised."
    )


def _nl_factor_sentences(row: "pd.Series"):
    dte = int(row.get("days_to_expiry", 0))
    qty = float(row.get("quantity", 0))
    daily = float(row.get("daily_consumption", 0.01))
    shelf_pct = float(row.get("shelf_life_consumed_pct", 0))
    stock_ratio = float(row.get("stock_expiry_ratio", 1))
    waste_hist = float(row.get("wastage_history_pct", 0))
    overstock = bool(row.get("overstock_flag", False))
    stock_days = qty / max(daily, 0.01)
    storage = str(row.get("storage_type", "")).lower()

    risk, safe = [], []

    if dte <= 3:
        risk.append(f"Only {dte} day(s) remain before expiry.")
    elif dte <= 7:
        risk.append(f"Expiry is in {dte} days — within the critical planning window.")

    if shelf_pct >= 0.75:
        risk.append(f"{shelf_pct*100:.0f}% of the shelf life has been consumed.")

    if overstock or stock_days > dte * 1.2:
        risk.append(
            f"Stock will last {stock_days:.0f} days but the item expires in {dte} — "
            f"that's {stock_days - dte:.0f} extra days of surplus."
        )
    elif stock_ratio > 1.5:
        risk.append(f"The stock-to-expiry ratio is {stock_ratio:.1f}x (ideal is below 1.0).")

    if waste_hist >= 0.20:
        risk.append(f"Historical waste rate for this ingredient is {waste_hist*100:.0f}%.")

    if storage == "ambient" and dte <= 5:
        risk.append("Ambient storage accelerates spoilage for short-dated items.")

    # Safe factors
    if dte > 14:
        safe.append(f"Still {dte} days until expiry — enough time to plan usage.")
    if daily >= qty / max(dte, 1):
        safe.append("Daily consumption is high enough to use up stock before expiry.")
    if shelf_pct < 0.4:
        safe.append(f"Only {shelf_pct*100:.0f}% of shelf life used — item is still fresh.")
    if waste_hist < 0.05:
        safe.append(f"Very low historical waste rate ({waste_hist*100:.0f}%) for this ingredient.")
    if storage == "frozen":
        safe.append("Frozen storage significantly extends usable life.")

    return risk, safe


# ── Model loader ───────────────────────────────────────────────────────────────
_MODEL_VERSION = "v4"  # bump this to force retrain on next deploy

@st.cache_resource(show_spinner=False)
def load_models():
    import joblib
    from pathlib import Path
    from src.models.wastage_predictor import WastagePredictor
    from src.models.demand_forecaster import DemandForecaster
    from src.models.anomaly_detector import InventoryAnomalyDetector
    from src.recommendations.rag_engine import RAGRecommendationEngine
    from src.data.generator import generate_inventory_dataset, generate_demand_history
    from src.data.preprocessor import clean_inventory, encode_categoricals

    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    version_file = model_dir / "model_version.txt"

    saved_version = version_file.read_text().strip() if version_file.exists() else ""
    needs_train = saved_version != _MODEL_VERSION

    wastage    = WastagePredictor()
    anomaly    = InventoryAnomalyDetector()
    forecaster = DemandForecaster()
    rag        = RAGRecommendationEngine()

    if needs_train:
        # Use smaller dataset for faster cold-start (~8s vs ~25s)
        df_raw = generate_inventory_dataset(n_records=600, seed=settings.random_seed)
        df, _  = clean_inventory(df_raw)
        df     = encode_categoricals(df)

        wastage.train(df);    wastage.save()
        anomaly.fit(df);      anomaly.save()

        ingredients = df["ingredient_name"].unique().tolist()[:20]
        demand_df   = generate_demand_history(ingredients=ingredients, n_days=90)
        forecaster.fit(demand_df); forecaster.save()

        version_file.write_text(_MODEL_VERSION)
    else:
        wastage.load()
        anomaly.load()
        forecaster.load()

    rag.load() or rag.build_index()
    return wastage, forecaster, anomaly, rag


@st.cache_data(show_spinner=False)
def get_inventory_data(n_records: int = 1200):
    from src.data.generator import generate_inventory_dataset
    from src.data.preprocessor import clean_inventory, encode_categoricals
    df_raw       = generate_inventory_dataset(n_records=n_records, seed=settings.random_seed)
    df, quality  = clean_inventory(df_raw)
    df           = encode_categoricals(df)
    return df, quality


def get_groq_key():
    key = ""
    try:   key = st.secrets.get("GROQ_API_KEY", "")
    except: pass
    return key or os.environ.get("GROQ_API_KEY", "")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Hero header
    st.markdown("""
    <div class="hero">
        <h1>🥗 Ecomeal AI</h1>
        <p>AI-powered food waste intelligence — predict, prevent, and act before ingredients expire.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        n_records = st.slider("Inventory Size", 200, 2000, 1200, 100,
                              help="Number of inventory records to analyse")
        restaurant_filter = st.selectbox("Restaurant", [
            "All Restaurants", "The Spice Garden", "Urban Kitchen",
            "Green Leaf Bistro", "The Curry House", "Bay Leaf Restaurant", "Savanna Grill",
        ])
        st.markdown("---")
        st.markdown("### 📌 About")
        st.markdown("""
        **Ecomeal AI** helps restaurants:
        - 🔴 Spot expiring stock early
        - 📈 Forecast ingredient demand
        - 👨‍🍳 Generate Chef Specials from at-risk items
        - 🚨 Detect data anomalies
        """)

    # Load models & data
    with st.spinner("🤖 Loading AI models..."):
        wastage, forecaster, anomaly, rag = load_models()

    with st.spinner("📦 Loading inventory data..."):
        df, quality = get_inventory_data(n_records)

    if restaurant_filter != "All Restaurants" and "restaurant" in df.columns:
        df = df[df["restaurant"] == restaurant_filter]

    with st.spinner("🔍 Running predictions..."):
        try:
            preds  = wastage.predict(df)
            df_pred = pd.concat([df.reset_index(drop=True), preds], axis=1)
            df_pred = anomaly.detect(df_pred)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            df_pred = df.copy()
            df_pred["waste_probability"] = df_pred.get("waste_risk_score", 0.3)
            df_pred["risk_level_pred"]   = "medium"

    # ── KPI Row ────────────────────────────────────────────────────────────────
    risk_col  = "risk_level_pred" if "risk_level_pred" in df_pred.columns else "risk_level"
    risk_counts = df_pred[risk_col].value_counts() if risk_col in df_pred.columns else pd.Series()
    high_risk_df = df_pred[df_pred.get(risk_col, pd.Series()).isin(["high", "critical"])]
    waste_val    = high_risk_df.get("potential_waste_value", pd.Series(dtype=float)).sum()
    anomaly_count = int(df_pred.get("is_anomaly", pd.Series(dtype=bool)).sum())

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card blue">
            <div class="kpi-label">Total Items</div>
            <div class="kpi-value">{len(df_pred):,}</div>
            <div class="kpi-sub">in inventory</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-label">🔴 Critical Risk</div>
            <div class="kpi-value">{int(risk_counts.get('critical', 0))}</div>
            <div class="kpi-sub">Act today</div>
        </div>
        <div class="kpi-card orange">
            <div class="kpi-label">🟠 High Risk</div>
            <div class="kpi-value">{int(risk_counts.get('high', 0))}</div>
            <div class="kpi-sub">Use within 2 days</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-label">💸 Waste Value at Risk</div>
            <div class="kpi-value">₹{waste_val:,.0f}</div>
            <div class="kpi-sub">estimated loss</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-label">⚠️ Anomalies</div>
            <div class="kpi-value">{anomaly_count}</div>
            <div class="kpi-sub">unusual entries</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
        "🔍 Risk Analysis", "📈 Demand Forecast",
        "⚠️ Anomalies", "👨‍🍳 Chef Specials",
        "🔬 Explainability", "🛒 Operations",
        "📊 Insights", "🗄️ Data"
    ])

    # ── Tab 1: Risk Analysis ───────────────────────────────────────────────────
    with t1:
        st.markdown('<div class="section-title">Waste Risk Overview</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            if risk_col in df_pred.columns:
                rd = df_pred[risk_col].value_counts().reset_index()
                rd.columns = ["Risk Level", "Count"]
                color_map = {"critical": "#fc5c7d", "high": "#f7971e", "medium": "#f6d365", "low": "#56ab2f"}
                fig = px.donut if hasattr(px, "donut") else px.pie
                fig = px.pie(rd, values="Count", names="Risk Level",
                             color="Risk Level", color_discrete_map=color_map,
                             hole=0.55, title="Risk Distribution")
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#a0aec0", title_font_color="#fff",
                    legend=dict(font=dict(color="#a0aec0")),
                    margin=dict(t=50, b=20)
                )
                fig.update_traces(textfont_color="#fff")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "waste_probability" in df_pred.columns:
                fig2 = px.histogram(df_pred, x="waste_probability", nbins=30,
                                    title="Waste Probability Distribution",
                                    labels={"waste_probability": "Waste Probability"},
                                    color_discrete_sequence=["#667eea"])
                fig2.add_vline(x=0.5, line_dash="dash", line_color="#fc5c7d",
                               annotation_text="Risk Threshold", annotation_font_color="#fc5c7d")
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#a0aec0", title_font_color="#fff",
                    xaxis=dict(gridcolor="#2d3748"), yaxis=dict(gridcolor="#2d3748"),
                    margin=dict(t=50, b=20)
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">🚨 Top At-Risk Items</div>', unsafe_allow_html=True)
        show_cols = [c for c in ["ingredient_name", "category", "quantity", "unit",
                                  "days_to_expiry", "waste_probability", risk_col,
                                  "potential_waste_value"] if c in df_pred.columns]
        top = df_pred.nlargest(20, "waste_probability") if "waste_probability" in df_pred.columns else df_pred.head(20)
        top_display = top[show_cols].copy()
        if "waste_probability" in top_display.columns:
            top_display["waste_probability"] = top_display["waste_probability"].apply(lambda x: f"{x:.1%}")
        if "potential_waste_value" in top_display.columns:
            top_display["potential_waste_value"] = top_display["potential_waste_value"].apply(lambda x: f"₹{x:,.0f}")
        top_display.columns = [c.replace("_", " ").title() for c in top_display.columns]
        st.dataframe(top_display, use_container_width=True, height=380, hide_index=True)

        st.markdown('<div class="section-title">Risk by Category</div>', unsafe_allow_html=True)
        if "category" in df_pred.columns and "waste_probability" in df_pred.columns:
            cat = df_pred.groupby("category")["waste_probability"].mean().sort_values(ascending=True).reset_index()
            cat.columns = ["Category", "Avg Waste Probability"]
            fig3 = px.bar(cat, x="Avg Waste Probability", y="Category", orientation="h",
                          color="Avg Waste Probability", color_continuous_scale="RdYlGn_r",
                          title="Average Waste Risk by Category")
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#a0aec0", title_font_color="#fff",
                xaxis=dict(gridcolor="#2d3748"), yaxis=dict(gridcolor="#2d3748"),
                coloraxis_showscale=False, margin=dict(t=50, b=20)
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 2: Demand Forecast ─────────────────────────────────────────────────
    with t2:
        st.markdown('<div class="section-title">Ingredient Demand Forecasting</div>', unsafe_allow_html=True)

        available = list(forecaster.models.keys())
        if not available:
            st.warning("No demand models trained yet.")
        else:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                ingredient = st.selectbox("Select Ingredient", available[:50])
            with col_b:
                horizon = st.slider("Forecast Days", 7, 60, 14)

            if st.button("📈 Generate Forecast", type="primary"):
                with st.spinner("Forecasting..."):
                    try:
                        fc = forecaster.forecast(ingredient, horizon)
                        fc["ds"] = pd.to_datetime(fc["ds"])

                        fig_fc = go.Figure()
                        fig_fc.add_trace(go.Scatter(
                            x=fc["ds"], y=fc["yhat_upper"],
                            fill=None, mode="lines", line_color="rgba(102,126,234,0)",
                            showlegend=False, name="Upper bound"
                        ))
                        fig_fc.add_trace(go.Scatter(
                            x=fc["ds"], y=fc["yhat_lower"],
                            fill="tonexty", mode="lines",
                            line_color="rgba(102,126,234,0)",
                            fillcolor="rgba(102,126,234,0.15)",
                            name="Confidence Interval"
                        ))
                        fig_fc.add_trace(go.Scatter(
                            x=fc["ds"], y=fc["yhat"],
                            mode="lines+markers",
                            line=dict(color="#667eea", width=3),
                            marker=dict(size=5),
                            name="Forecast"
                        ))
                        fig_fc.update_layout(
                            title=f"Demand Forecast — {ingredient}",
                            xaxis_title="Date", yaxis_title="Quantity",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#a0aec0", title_font_color="#fff",
                            xaxis=dict(gridcolor="#2d3748"), yaxis=dict(gridcolor="#2d3748"),
                            hovermode="x unified", legend=dict(font=dict(color="#a0aec0")),
                            margin=dict(t=50, b=20)
                        )
                        st.plotly_chart(fig_fc, use_container_width=True)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Peak Demand",    f"{fc['yhat'].max():.1f} units")
                        c2.metric("Daily Average",  f"{fc['yhat'].mean():.1f} units")
                        c3.metric("Total Forecast", f"{fc['yhat'].sum():.1f} units")
                    except Exception as e:
                        st.error(f"Forecast failed: {e}")

        st.markdown("---")
        st.markdown('<div class="section-title">📦 Overstock & Shortage Detection</div>', unsafe_allow_html=True)
        if st.button("🔎 Run Overstock Analysis"):
            with st.spinner("Analysing stock levels vs forecasted demand..."):
                try:
                    os_df = forecaster.detect_overstock(df_pred, horizon_days=14)
                    over  = os_df[os_df["overstock_risk"]]
                    short = os_df[os_df["shortage_risk"]]
                    col_o, col_s = st.columns(2)
                    with col_o:
                        st.markdown(f"**🔴 Overstock Items ({len(over)})**")
                        if not over.empty:
                            disp = over[["ingredient_name","current_quantity","overstock_quantity"]].copy()
                            disp.columns = ["Ingredient","Current Stock","Excess Qty"]
                            st.dataframe(disp, use_container_width=True, hide_index=True)
                        else:
                            st.success("No overstock detected!")
                    with col_s:
                        st.markdown(f"**🟡 Shortage Risk Items ({len(short)})**")
                        if not short.empty:
                            disp = short[["ingredient_name","current_quantity","shortage_quantity"]].copy()
                            disp.columns = ["Ingredient","Current Stock","Shortfall"]
                            st.dataframe(disp, use_container_width=True, hide_index=True)
                        else:
                            st.success("No shortage risks detected!")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    # ── Tab 3: Anomalies ───────────────────────────────────────────────────────
    with t3:
        st.markdown('<div class="section-title">Inventory Anomaly Detection</div>', unsafe_allow_html=True)
        st.caption("Items flagged as statistically unusual — possible data errors, theft, demand spikes, or spoilage events.")

        if "is_anomaly" in df_pred.columns:
            anom  = df_pred[df_pred["is_anomaly"]]
            c1, c2, c3 = st.columns(3)
            c1.metric("Anomalies Found",  len(anom))
            c2.metric("Anomaly Rate",     f"{len(anom)/len(df_pred)*100:.1f}%")
            c3.metric("Severity",         "High" if not anom.empty and anom.get("anomaly_score", pd.Series()).mean() < -0.6 else "Medium")

            if "anomaly_score" in df_pred.columns:
                fig_an = px.scatter(
                    df_pred, x="quantity", y="daily_consumption",
                    color="is_anomaly",
                    color_discrete_map={True: "#fc5c7d", False: "#4facfe"},
                    hover_data=["ingredient_name", "category"],
                    title="Anomalies: Quantity vs Daily Consumption",
                    labels={"quantity": "Quantity in Stock", "daily_consumption": "Daily Usage", "is_anomaly": "Anomaly"},
                    opacity=0.6,
                )
                fig_an.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#a0aec0", title_font_color="#fff",
                    xaxis=dict(gridcolor="#2d3748"), yaxis=dict(gridcolor="#2d3748"),
                    margin=dict(t=50, b=20)
                )
                st.plotly_chart(fig_an, use_container_width=True)

            if not anom.empty:
                st.markdown('<div class="section-title">Flagged Items</div>', unsafe_allow_html=True)
                show = [c for c in ["ingredient_name","category","quantity","daily_consumption","anomaly_reason"] if c in anom.columns]
                disp = anom[show].copy()
                disp.columns = [c.replace("_"," ").title() for c in disp.columns]
                st.dataframe(disp, use_container_width=True, hide_index=True, height=350)
        else:
            st.info("Anomaly data not available.")

    # ── Tab 4: Chef Specials ───────────────────────────────────────────────────
    with t4:
        st.markdown('<div class="section-title">👨‍🍳 Chef Specials Generator</div>', unsafe_allow_html=True)
        st.caption("Select ingredients nearing expiry — AI will suggest dishes that use them up before they go to waste.")

        suggested = []
        if risk_col in df_pred.columns:
            suggested = df_pred[df_pred[risk_col].isin(["critical","high"])]["ingredient_name"].unique().tolist()[:5]

        col_l, col_r = st.columns([3, 1])
        with col_l:
            selected = st.multiselect(
                "Expiring Ingredients",
                options=df_pred["ingredient_name"].unique().tolist(),
                default=suggested[:4],
                help="Pick the ingredients you need to use up urgently"
            )
            cuisine = st.text_input("Cuisine Preference", placeholder="e.g. Indian, Italian, Asian, Fusion...")
        with col_r:
            n_dishes = st.slider("No. of Dishes", 1, 5, 3)
            dietary  = st.multiselect("Dietary", ["Vegetarian","Vegan","Gluten-Free","Dairy-Free"])

        if st.button("✨ Generate Chef Specials", type="primary"):
            if not selected:
                st.warning("Please select at least one ingredient.")
            else:
                with st.spinner("AI is creating dishes for you..."):
                    try:
                        groq_key = get_groq_key()
                        os.environ["GROQ_API_KEY"] = groq_key

                        from src.recommendations.chef_specials import ChefSpecialsEngine
                        chef   = ChefSpecialsEngine()
                        know   = rag.get_ingredient_knowledge(selected[:6])
                        ctx    = " | ".join(k["text"] for k in know[:3]) if know else None
                        result = chef.generate(
                            ingredients=selected,
                            cuisine_preference=cuisine or None,
                            dietary_restrictions=dietary or None,
                            n_suggestions=n_dishes,
                            context=ctx,
                        )

                        if result.get("_source") == "fallback":
                            st.warning("⚠️ Using template suggestions — add GROQ_API_KEY in app secrets for AI-generated dishes.")
                        else:
                            st.success("✅ Dishes generated by AI")

                        urgency_icon = {"use_today":"🔴","use_within_2_days":"🟠","use_this_week":"🟡"}

                        for special in result.get("chef_specials", []):
                            icon = urgency_icon.get(special.get("urgency",""), "🟢")
                            urgency_label = special.get("urgency","").replace("_"," ").title()
                            with st.expander(f"{icon} **{special.get('name','')}** — {special.get('prep_time_minutes','?')} min | {urgency_label}", expanded=True):
                                c1, c2 = st.columns([2,1])
                                with c1:
                                    st.markdown(f"**About this dish**")
                                    st.markdown(special.get("description",""))
                                    st.markdown(f"**🥬 Ingredients used:** {', '.join(special.get('ingredients_used',[]))}")
                                    st.markdown(f"**♻️ Why it reduces waste:** {special.get('waste_reduction_rationale','')}")
                                with c2:
                                    st.info(f"💡 **Storage Tip**\n\n{special.get('storage_tip','')}")

                        if result.get("general_recommendation"):
                            st.markdown("---")
                            st.markdown(f"**📋 Overall Strategy:** {result['general_recommendation']}")

                        if result.get("estimated_waste_reduction_pct"):
                            st.metric("Estimated Waste Reduction", f"{result['estimated_waste_reduction_pct']}%")

                        if know:
                            with st.expander("📚 Knowledge base used"):
                                for k in know[:3]:
                                    st.markdown(f"- {k['text']}")

                    except Exception as e:
                        st.error(f"Error: {e}")

    # ── Tab 5: Explainability ──────────────────────────────────────────────────
    with t5:
        st.markdown('<div class="section-title">Why is an Item Risky?</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Feature Importance — What drives waste predictions**")
            fi = wastage.get_feature_importance()
            if not fi.empty:
                fi_top = fi.head(10).copy()
                fi_top["feature"] = fi_top["feature"].str.replace("_", " ").str.title()
                fig_fi = px.bar(fi_top, x="importance", y="feature", orientation="h",
                                color="importance", color_continuous_scale="Viridis",
                                labels={"importance":"Importance Score","feature":"Feature"})
                fig_fi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#a0aec0", yaxis=dict(autorange="reversed", gridcolor="#2d3748"),
                    xaxis=dict(gridcolor="#2d3748"), coloraxis_showscale=False,
                    margin=dict(t=10, b=20)
                )
                st.plotly_chart(fig_fi, use_container_width=True)

        with col_b:
            st.markdown("**Item-Level Explanation**")
            sel = st.selectbox("Choose an ingredient to explain", sorted(df_pred["ingredient_name"].unique().tolist()))
            row_df = df_pred[df_pred["ingredient_name"] == sel].head(1)
            if not row_df.empty:
                row = row_df.iloc[0]
                shap_exp = wastage.explain(row_df, max_items=1)
                from src.explainability.explainer import explain_item_risk
                shap_top = shap_exp[0]["top_features"] if shap_exp else None
                expl = explain_item_risk(row, shap_top)

                risk_colors = {"critical":"#fc5c7d","high":"#f7971e","medium":"#f6d365","low":"#56ab2f"}
                lvl = expl["risk_level"]
                color = risk_colors.get(lvl, "#a0aec0")
                st.markdown(
                    f"**Risk Level:** <span style='color:{color};font-weight:700;font-size:1.1rem'>"
                    f"{lvl.upper()}</span> &nbsp; Score: `{expl['risk_score']:.2f}`",
                    unsafe_allow_html=True,
                )

                # ── SHAP summary sentence (model output) ──────────────────────
                shap_summary = shap_exp[0].get("summary_sentence") if shap_exp else None
                display_sentence = shap_summary or _nl_explain_row(row)
                source_label = "Model explanation" if shap_summary else "Rule-based explanation"
                st.markdown(f"""
<div class="nl-card">
  <div class="nl-label">{source_label}</div>
  {display_sentence}
</div>""", unsafe_allow_html=True)

                # ── SHAP factor sentences (model-driven) ──────────────────────
                if shap_exp:
                    top = shap_exp[0].get("top_features", [])
                    risk_factors, safe_factors = _deduplicate_factors(top)
                    risk_sentences = [f["natural_language"] for f in risk_factors if f.get("natural_language")]
                    safe_sentences = [f["natural_language"] for f in safe_factors if f.get("natural_language")]
                    # If deduplication removed everything, show raw top features
                    if not risk_sentences and not safe_sentences:
                        risk_sentences = [f["natural_language"] for f in top[:3] if f.get("natural_language") and f["direction"] == "increases_risk"]
                        safe_sentences  = [f["natural_language"] for f in top[:3] if f.get("natural_language") and f["direction"] == "decreases_risk"]
                else:
                    risk_sentences, safe_sentences = _nl_factor_sentences(row)

                if risk_sentences:
                    with st.expander("🔴 Why the model flagged this item", expanded=True):
                        for s in risk_sentences:
                            st.markdown(f"- {s}")
                if safe_sentences:
                    with st.expander("🟢 What's reducing the risk"):
                        for s in safe_sentences:
                            st.markdown(f"- {s}")
                if not risk_sentences and not safe_sentences:
                    st.caption("No dominant risk factors detected by the model for this item.")

                st.success(f"✅ **Recommended Action:** {expl['recommended_action']}")


    # ── Tab 6: Operations ─────────────────────────────────────────────────────
    with t6:
        st.markdown('<div class="section-title">🛒 Reorder Suggestions</div>', unsafe_allow_html=True)
        st.caption("Based on current stock and forecasted demand — exact quantities and timing to reorder.")

        reorder_rows = []
        for _, row in df_pred.iterrows():
            daily = float(row.get("daily_consumption", 0.01))
            qty   = float(row.get("quantity", 0))
            dte   = int(row.get("days_to_expiry", 0))
            rp    = float(row.get("reorder_point", 0))
            price = float(row.get("price_per_unit", 0))
            stock_days = qty / max(daily, 0.01)
            lead_time = 3  # assumed days

            below_reorder = qty <= rp
            will_run_out  = stock_days <= lead_time + 2
            if below_reorder or will_run_out:
                reorder_qty   = round(daily * 14 - qty, 2)  # restock to 2 weeks supply
                reorder_qty   = max(reorder_qty, daily * 3)
                days_until    = max(0, int(stock_days - lead_time))
                reorder_rows.append({
                    "Ingredient":     row.get("ingredient_name", ""),
                    "Category":       row.get("category", ""),
                    "Current Stock":  f"{qty:.1f} {row.get('unit','')}",
                    "Days of Stock":  int(stock_days),
                    "Order Qty":      f"{reorder_qty:.1f} {row.get('unit','')}",
                    "Order By":       f"in {days_until} day(s)",
                    "Est. Cost":      f"₹{reorder_qty * price:,.0f}",
                    "Urgency":        "🔴 Today" if days_until == 0 else ("🟠 Soon" if days_until <= 2 else "🟡 Plan"),
                })

        if reorder_rows:
            ro_df = pd.DataFrame(reorder_rows).sort_values("Days of Stock")
            st.dataframe(ro_df, use_container_width=True, hide_index=True, height=320)
            total_cost = sum(
                float(r["Est. Cost"].replace("₹","").replace(",",""))
                for r in reorder_rows
            )
            c1, c2 = st.columns(2)
            c1.metric("Items to Reorder", len(reorder_rows))
            c2.metric("Total Reorder Cost", f"₹{total_cost:,.0f}")
        else:
            st.success("✅ All items are adequately stocked — no reorders needed right now.")

        st.markdown("---")
        st.markdown('<div class="section-title">💰 Waste Cost Tracker</div>', unsafe_allow_html=True)
        st.caption("Simulated waste prevented by acting on AI recommendations over the past 30 days.")

        np.random.seed(99)
        days_back = 30
        dates = pd.date_range(end=date.today(), periods=days_back)
        base_waste  = float(df_pred.get("potential_waste_value", pd.Series([0])).mean()) * 0.15
        daily_waste  = np.random.normal(base_waste, base_waste * 0.3, days_back).clip(0)
        daily_saved  = daily_waste * np.random.uniform(0.4, 0.75, days_back)
        tracker_df   = pd.DataFrame({"date": dates, "Waste Prevented (₹)": daily_saved, "Waste Occurred (₹)": daily_waste - daily_saved})

        fig_wt = go.Figure()
        fig_wt.add_trace(go.Bar(x=tracker_df["date"], y=tracker_df["Waste Prevented (₹)"],
                                name="Prevented", marker_color="#56ab2f"))
        fig_wt.add_trace(go.Bar(x=tracker_df["date"], y=tracker_df["Waste Occurred (₹)"],
                                name="Still Wasted", marker_color="#fc5c7d"))
        fig_wt.update_layout(
            barmode="stack", title="Daily Waste Cost — Last 30 Days",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", title_font_color="#fff",
            xaxis=dict(gridcolor="#2d3748"), yaxis=dict(gridcolor="#2d3748", tickprefix="₹"),
            legend=dict(font=dict(color="#a0aec0")), margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_wt, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Waste Prevented", f"₹{daily_saved.sum():,.0f}")
        c2.metric("Still Wasted",          f"₹{(daily_waste - daily_saved).sum():,.0f}")
        c3.metric("Prevention Rate",       f"{daily_saved.sum()/max(daily_waste.sum(),1)*100:.0f}%")

        st.markdown("---")
        st.markdown('<div class="section-title">🔄 Ingredient Substitution</div>', unsafe_allow_html=True)
        st.caption("Overstocked ingredients and dishes that can substitute them into the menu.")

        _SUBSTITUTIONS = {
            "Tomatoes":       ["Tomato Shorba", "Tomato Rice", "Bruschetta", "Shakshuka"],
            "Onions":         ["French Onion Soup", "Onion Bhaji", "Caramelised Onion Tart"],
            "Spinach":        ["Palak Paneer", "Spinach Dal", "Green Smoothie Bowl", "Spanakopita"],
            "Mushrooms":      ["Mushroom Risotto", "Mushroom Soup", "Mushroom Stir Fry"],
            "Bell Peppers":   ["Stuffed Peppers", "Peperonata", "Pepper Stir Fry"],
            "Chicken Breast": ["Grilled Chicken Bowl", "Chicken Tikka", "Chicken Caesar Wrap"],
            "Paneer":         ["Palak Paneer", "Paneer Tikka", "Kadai Paneer", "Paneer Bhurji"],
            "Basmati Rice":   ["Biryani", "Fried Rice", "Khichdi", "Rice Pudding"],
            "Bread":          ["French Toast", "Bread Pakoda", "Croutons", "Bread Pudding"],
            "Carrots":        ["Carrot Halwa", "Carrot Soup", "Glazed Carrots", "Gajar Ka Halwa"],
            "Potatoes":       ["Aloo Tikki", "Mashed Potato", "Hash Browns", "Potato Wedges"],
            "Lentils":        ["Dal Tadka", "Lentil Soup", "Dal Makhani", "Masoor Dal"],
        }

        overstock_items = df_pred[
            (df_pred.get("overstock_flag", pd.Series(False, index=df_pred.index)).astype(bool)) |
            (df_pred.get("stock_expiry_ratio", pd.Series(1, index=df_pred.index)) > 1.3)
        ]["ingredient_name"].value_counts().head(8).index.tolist()

        if overstock_items:
            for ing in overstock_items[:6]:
                dishes = _SUBSTITUTIONS.get(ing)
                if not dishes:
                    # generic fallback
                    dishes = [f"{ing} Stir Fry", f"{ing} Soup", f"{ing} Curry"]
                qty_row = df_pred[df_pred["ingredient_name"] == ing].head(1)
                qty_val = f"{qty_row['quantity'].values[0]:.1f} {qty_row['unit'].values[0]}" if not qty_row.empty else ""
                st.markdown(f"""
<div class="nl-card">
  <div class="nl-label">Overstock: {ing} &nbsp;·&nbsp; {qty_val}</div>
  <b>Use in:</b> {' &nbsp;·&nbsp; '.join(f'<span style="color:#4facfe">{d}</span>' for d in dishes)}
</div>""", unsafe_allow_html=True)
        else:
            st.success("No significant overstock detected.")

    # ── Tab 7: Insights ───────────────────────────────────────────────────────
    with t7:
        st.markdown('<div class="section-title">📊 Multi-Restaurant Waste Risk Comparison</div>', unsafe_allow_html=True)
        st.caption("Side-by-side waste risk profile across all restaurants in the network.")

        from src.data.generator import generate_inventory_dataset
        from src.data.preprocessor import clean_inventory, encode_categoricals

        @st.cache_data(show_spinner=False)
        def get_all_restaurants_data(n: int = 1200):
            df_raw, _ = clean_inventory(generate_inventory_dataset(n_records=n, seed=42))
            df_all    = encode_categoricals(df_raw)
            preds_all = wastage.predict(df_all)
            return pd.concat([df_all.reset_index(drop=True), preds_all], axis=1)

        df_all = get_all_restaurants_data()

        # Stacked bar — risk distribution per restaurant
        rc = df_all.groupby(["restaurant", "risk_level_pred"]).size().reset_index(name="count")
        fig_rc = px.bar(
            rc, x="restaurant", y="count", color="risk_level_pred",
            color_discrete_map={"critical":"#fc5c7d","high":"#f7971e","medium":"#f6d365","low":"#56ab2f"},
            title="Risk Distribution by Restaurant",
            labels={"restaurant":"Restaurant","count":"Items","risk_level_pred":"Risk"},
            barmode="stack",
        )
        fig_rc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", title_font_color="#fff",
            xaxis=dict(gridcolor="#2d3748", tickangle=-20),
            yaxis=dict(gridcolor="#2d3748"),
            legend=dict(font=dict(color="#a0aec0")),
            margin=dict(t=50, b=60),
        )
        st.plotly_chart(fig_rc, use_container_width=True)

        # KPI table per restaurant
        rest_summary = df_all.groupby("restaurant").agg(
            Total=("ingredient_name","count"),
            Critical=(      "risk_level_pred", lambda x: (x=="critical").sum()),
            High=(          "risk_level_pred", lambda x: (x=="high").sum()),
            Avg_Risk=(      "waste_probability","mean"),
            Waste_Value=(   "potential_waste_value","sum"),
        ).reset_index()
        rest_summary["Avg_Risk"]     = rest_summary["Avg_Risk"].apply(lambda x: f"{x:.1%}")
        rest_summary["Waste_Value"]  = rest_summary["Waste_Value"].apply(lambda x: f"₹{x:,.0f}")
        rest_summary.columns = ["Restaurant","Total Items","Critical","High Risk","Avg Risk","Waste Value at Risk"]
        st.dataframe(rest_summary, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<div class="section-title">📉 Historical Waste Risk Trend — Last 30 Days</div>', unsafe_allow_html=True)
        st.caption("Simulated daily waste risk score per category to show how risk evolves over time.")

        categories = df_pred["category"].dropna().unique().tolist()
        sel_cats   = st.multiselect("Select categories", categories, default=categories[:4])

        if sel_cats:
            np.random.seed(7)
            trend_dates = pd.date_range(end=date.today(), periods=30)
            trend_rows  = []
            for cat in sel_cats:
                base = df_pred[df_pred["category"]==cat]["waste_probability"].mean() if "waste_probability" in df_pred.columns else 0.4
                noise = np.random.normal(0, 0.03, 30)
                trend = np.linspace(0, np.random.choice([-0.05, 0.05]), 30)
                vals  = (base + trend + noise).clip(0, 1)
                for d, v in zip(trend_dates, vals):
                    trend_rows.append({"Date": d, "Category": cat, "Avg Waste Risk": round(v, 3)})

            trend_df = pd.DataFrame(trend_rows)
            fig_tr = px.line(
                trend_df, x="Date", y="Avg Waste Risk", color="Category",
                title="Waste Risk Trend by Category",
                labels={"Avg Waste Risk": "Avg Waste Probability"},
                markers=True,
            )
            fig_tr.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#a0aec0", title_font_color="#fff",
                xaxis=dict(gridcolor="#2d3748"), yaxis=dict(gridcolor="#2d3748", tickformat=".0%"),
                legend=dict(font=dict(color="#a0aec0")),
                margin=dict(t=50, b=20),
                hovermode="x unified",
            )
            fig_tr.add_hline(y=0.45, line_dash="dash", line_color="#f7971e",
                             annotation_text="High Risk Threshold", annotation_font_color="#f7971e")
            st.plotly_chart(fig_tr, use_container_width=True)

    # ── Tab 8: Raw Data ────────────────────────────────────────────────────────
    with t8:
        st.markdown('<div class="section-title">Inventory Data</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records",      f"{len(df_pred):,}")
        c2.metric("Data Quality",       f"{quality.get('data_quality_pct',100):.1f}%")
        c3.metric("Issues Auto-Fixed",  sum(v for v in quality.get("issues",{}).values() if isinstance(v,int)))

        risk_filter = st.multiselect("Filter by Risk", ["critical","high","medium","low"], default=["critical","high"])
        disp_df = df_pred[df_pred[risk_col].isin(risk_filter)] if risk_filter and risk_col in df_pred.columns else df_pred

        show_cols = [c for c in ["ingredient_name","category","restaurant","quantity","unit",
                                  "days_to_expiry","waste_probability",risk_col,"potential_waste_value"] if c in disp_df.columns]
        out = disp_df[show_cols].copy()
        if "waste_probability" in out.columns:
            out["waste_probability"] = out["waste_probability"].apply(lambda x: f"{x:.1%}")
        if "potential_waste_value" in out.columns:
            out["potential_waste_value"] = out["potential_waste_value"].apply(lambda x: f"₹{x:,.0f}")
        out.columns = [c.replace("_"," ").title() for c in out.columns]

        st.dataframe(out, use_container_width=True, height=450, hide_index=True)
        st.download_button(
            "📥 Download CSV", data=disp_df.to_csv(index=False),
            file_name=f"ecomeal_inventory_{date.today()}.csv", mime="text/csv"
        )



if __name__ == "__main__":
    main()
