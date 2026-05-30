"""
Ecomeal AI — Streamlit Dashboard
Interactive visualization of food waste predictions, forecasts, anomalies,
and AI-generated Chef Specials.
"""

import os
import json
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from typing import List, Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import get_settings

settings = get_settings()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ecomeal AI — Food Waste Intelligence",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card { background: #1e1e2e; border-radius: 12px; padding: 16px; border-left: 4px solid; }
.critical { border-color: #ff4444; }
.high { border-color: #ff8800; }
.medium { border-color: #ffcc00; }
.low { border-color: #00cc44; }
.stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
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

    model_loaded = wastage.load() and anomaly.load()
    forecaster_loaded = forecaster.load()
    rag.load() or rag.build_index()

    if not model_loaded:
        df_raw = generate_inventory_dataset(n_records=settings.dataset_size, seed=settings.random_seed)
        df, _ = clean_inventory(df_raw)
        df = encode_categoricals(df)
        wastage.train(df)
        wastage.save()
        anomaly.fit(df)
        anomaly.save()

    if not forecaster_loaded:
        df_raw = generate_inventory_dataset(n_records=settings.dataset_size, seed=settings.random_seed)
        df, _ = clean_inventory(df_raw)
        ingredients = df["ingredient_name"].unique().tolist()[:30]
        demand_df = generate_demand_history(ingredients=ingredients, n_days=180)
        forecaster.fit(demand_df)
        forecaster.save()

    return wastage, forecaster, anomaly, chef, rag


@st.cache_data(show_spinner="Generating inventory data...")
def get_inventory_data(n_records: int = 1200):
    from src.data.generator import generate_inventory_dataset
    from src.data.preprocessor import clean_inventory, encode_categoricals
    df_raw = generate_inventory_dataset(n_records=n_records, seed=settings.random_seed)
    df, quality = clean_inventory(df_raw)
    df = encode_categoricals(df)
    return df, quality


def run_predictions(df, wastage_model, anomaly_model):
    preds = wastage_model.predict(df)
    df_out = pd.concat([df.reset_index(drop=True), preds], axis=1)
    df_out = anomaly_model.detect(df_out)
    return df_out


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    st.title("🥗 Ecomeal AI — Food Waste Intelligence Platform")
    st.caption("AI-powered inventory analytics, waste prediction, demand forecasting & Chef Specials")

    # Load models
    try:
        wastage, forecaster, anomaly, chef, rag = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    n_records = st.sidebar.slider("Inventory Size", 200, 2000, 1200, 100)
    restaurant_filter = st.sidebar.selectbox(
        "Restaurant Filter",
        ["All Restaurants", "The Spice Garden", "Urban Kitchen", "Green Leaf Bistro",
         "The Curry House", "Bay Leaf Restaurant", "Savanna Grill"],
    )

    # Load data
    df, quality = get_inventory_data(n_records)

    # Filter by restaurant
    if restaurant_filter != "All Restaurants" and "restaurant" in df.columns:
        df = df[df["restaurant"] == restaurant_filter]

    # Run predictions
    with st.spinner("Running AI predictions..."):
        try:
            df_pred = run_predictions(df, wastage, anomaly)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            df_pred = df.copy()
            df_pred["waste_probability"] = df_pred.get("waste_risk_score", 0.3)
            df_pred["risk_level_pred"] = "medium"

    # ── KPI Row ────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Inventory Overview")
    col1, col2, col3, col4, col5 = st.columns(5)

    risk_counts = df_pred.get("risk_level_pred", pd.Series()).value_counts()
    waste_val = df_pred.get("potential_waste_value", pd.Series(dtype=float))
    high_risk_val = df_pred[df_pred.get("risk_level_pred", pd.Series()).isin(["high", "critical"])].get(
        "potential_waste_value", pd.Series(dtype=float)
    ).sum()

    col1.metric("Total Items", len(df_pred), help="Items in current inventory")
    col2.metric("🔴 Critical", int(risk_counts.get("critical", 0)), delta="Act now", delta_color="inverse")
    col3.metric("🟠 High Risk", int(risk_counts.get("high", 0)), delta="Within 2 days", delta_color="inverse")
    col4.metric("💸 Waste Value at Risk", f"₹{high_risk_val:,.0f}", help="Estimated loss from high/critical items")
    col5.metric("📦 Anomalies", int(df_pred.get("is_anomaly", pd.Series(dtype=bool)).sum()), help="Unusual inventory entries")

    # ── Tabs ────────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🔍 Risk Analysis", "📈 Demand Forecast", "⚠️ Anomalies",
        "👨‍🍳 Chef Specials", "📊 Explainability", "🗄️ Raw Data"
    ])

    # ── Tab 1: Risk Analysis ───────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("#### Waste Risk Distribution")
        col_a, col_b = st.columns(2)

        with col_a:
            risk_col = "risk_level_pred" if "risk_level_pred" in df_pred.columns else "risk_level"
            if risk_col in df_pred.columns:
                risk_dist = df_pred[risk_col].value_counts().reset_index()
                risk_dist.columns = ["risk_level", "count"]
                color_map = {"critical": "#FF4444", "high": "#FF8800", "medium": "#FFCC00", "low": "#00CC44"}
                fig = px.pie(
                    risk_dist, values="count", names="risk_level",
                    color="risk_level", color_discrete_map=color_map,
                    title="Risk Level Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if "waste_probability" in df_pred.columns:
                fig2 = px.histogram(
                    df_pred, x="waste_probability", nbins=30,
                    title="Waste Probability Distribution",
                    color_discrete_sequence=["#FF6B35"],
                )
                fig2.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Risk Threshold")
                st.plotly_chart(fig2, use_container_width=True)

        # Top risk items table
        st.markdown("#### 🚨 Top At-Risk Items")
        top_cols = [c for c in ["ingredient_name", "category", "quantity", "unit",
                                 "days_to_expiry", "waste_probability", "risk_level_pred",
                                 "potential_waste_value"] if c in df_pred.columns]
        top_risk = df_pred.nlargest(20, "waste_probability") if "waste_probability" in df_pred.columns else df_pred.head(20)

        def style_risk(val):
            colors = {"critical": "background-color: #FF4444; color: white",
                      "high": "background-color: #FF8800; color: white",
                      "medium": "background-color: #FFCC00",
                      "low": "background-color: #CCFFCC"}
            return colors.get(str(val), "")

        style_fn = getattr(top_risk[top_cols].style, "map", None) or getattr(top_risk[top_cols].style, "applymap")
        styled = style_fn(style_risk, subset=["risk_level_pred"] if "risk_level_pred" in top_cols else [])
        st.dataframe(styled, use_container_width=True, height=400)

        # Risk by category
        st.markdown("#### Risk by Category")
        if "category" in df_pred.columns and "waste_probability" in df_pred.columns:
            cat_risk = df_pred.groupby("category")["waste_probability"].mean().sort_values(ascending=False).reset_index()
            fig3 = px.bar(
                cat_risk, x="category", y="waste_probability",
                title="Average Waste Probability by Category",
                color="waste_probability", color_continuous_scale="RdYlGn_r",
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 2: Demand Forecast ─────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("#### Demand Forecasting")
        available_ingredients = list(forecaster.models.keys())

        if not available_ingredients:
            st.warning("No trained demand models available. Run the pipeline first.")
        else:
            sel_ingredient = st.selectbox("Select Ingredient", available_ingredients[:50])
            horizon = st.slider("Forecast Horizon (days)", 7, 60, 14)

            if st.button("Generate Forecast", type="primary"):
                with st.spinner(f"Forecasting demand for {sel_ingredient}..."):
                    try:
                        fc = forecaster.forecast(sel_ingredient, horizon)
                        fc["ds"] = pd.to_datetime(fc["ds"])

                        fig_fc = go.Figure()
                        fig_fc.add_trace(go.Scatter(
                            x=fc["ds"], y=fc["yhat"],
                            name="Forecast", line=dict(color="#2196F3", width=2)
                        ))
                        fig_fc.add_trace(go.Scatter(
                            x=pd.concat([fc["ds"], fc["ds"].iloc[::-1]]),
                            y=pd.concat([fc["yhat_upper"], fc["yhat_lower"].iloc[::-1]]),
                            fill="toself", fillcolor="rgba(33,150,243,0.2)",
                            line=dict(color="rgba(255,255,255,0)"),
                            name="80% Confidence Interval",
                        ))
                        fig_fc.update_layout(
                            title=f"Demand Forecast: {sel_ingredient}",
                            xaxis_title="Date", yaxis_title="Quantity",
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig_fc, use_container_width=True)

                        col_f1, col_f2, col_f3 = st.columns(3)
                        col_f1.metric("Peak Demand", f"{fc['yhat'].max():.1f}")
                        col_f2.metric("Avg Daily Demand", f"{fc['yhat'].mean():.1f}")
                        col_f3.metric("Total Forecast", f"{fc['yhat'].sum():.1f}")

                    except Exception as e:
                        st.error(f"Forecast error: {e}")

        # Overstock detection
        st.markdown("---")
        st.markdown("#### Overstock & Shortage Detection")
        if st.button("Run Overstock Analysis"):
            with st.spinner("Analyzing stock levels vs forecasted demand..."):
                try:
                    overstock_df = forecaster.detect_overstock(df_pred, horizon_days=14)
                    os_items = overstock_df[overstock_df["overstock_risk"]]
                    sh_items = overstock_df[overstock_df["shortage_risk"]]

                    col_os, col_sh = st.columns(2)
                    with col_os:
                        st.markdown(f"**🔴 Overstock Items ({len(os_items)})**")
                        if not os_items.empty:
                            st.dataframe(os_items[["ingredient_name", "current_quantity", "overstock_quantity"]],
                                        use_container_width=True)
                        else:
                            st.success("No overstock items detected!")

                    with col_sh:
                        st.markdown(f"**🟡 Shortage Risk Items ({len(sh_items)})**")
                        if not sh_items.empty:
                            st.dataframe(sh_items[["ingredient_name", "current_quantity", "shortage_quantity"]],
                                        use_container_width=True)
                        else:
                            st.success("No shortage risks detected!")
                except Exception as e:
                    st.error(f"Overstock analysis failed: {e}")

    # ── Tab 3: Anomalies ───────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("#### 🚨 Anomaly Detection Report")

        if "is_anomaly" in df_pred.columns:
            anomalies = df_pred[df_pred["is_anomaly"]]
            normal = df_pred[~df_pred["is_anomaly"]]

            col_an1, col_an2 = st.columns([1, 3])
            with col_an1:
                st.metric("Total Anomalies", len(anomalies))
                st.metric("Anomaly Rate", f"{len(anomalies)/len(df_pred)*100:.1f}%")
                st.metric("Severity", "High" if not anomalies.empty and anomalies.get("anomaly_score", pd.Series()).mean() < -0.6 else "Medium",
                          help="How unusual the flagged items are compared to normal inventory")

            with col_an2:
                if "anomaly_score" in df_pred.columns and len(df_pred) > 0:
                    fig_an = px.scatter(
                        df_pred, x="quantity", y="daily_consumption",
                        color="is_anomaly",
                        color_discrete_map={True: "#FF4444", False: "#4488FF"},
                        hover_data=["ingredient_name", "category"],
                        title="Anomalies: Quantity vs Daily Consumption",
                        opacity=0.6,
                    )
                    st.plotly_chart(fig_an, use_container_width=True)

            if not anomalies.empty:
                st.markdown("**Anomalous Items:**")
                cols_show = [c for c in ["ingredient_name", "category", "quantity", "daily_consumption",
                                          "anomaly_score", "anomaly_reason"] if c in anomalies.columns]
                st.dataframe(anomalies[cols_show].sort_values("anomaly_score"), use_container_width=True)
        else:
            st.info("Anomaly detection results not available.")

    # ── Tab 4: Chef Specials ───────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("#### 👨‍🍳 AI Chef Specials Generator")
        st.info("Select expiring ingredients and generate AI-powered dish suggestions.")

        # Auto-suggest from high risk items
        if "risk_level_pred" in df_pred.columns:
            suggested = (
                df_pred[df_pred["risk_level_pred"].isin(["critical", "high"])]
                ["ingredient_name"].unique().tolist()[:8]
            )
        else:
            suggested = df_pred["ingredient_name"].unique().tolist()[:5]

        col_cs1, col_cs2 = st.columns([2, 1])
        with col_cs1:
            selected_ingredients = st.multiselect(
                "Select Expiring Ingredients",
                options=df_pred["ingredient_name"].unique().tolist(),
                default=suggested[:4],
            )
            cuisine = st.text_input("Cuisine Preference (optional)", placeholder="e.g. Indian, Italian, Asian...")
            n_dishes = st.slider("Number of Suggestions", 1, 5, 3)

        with col_cs2:
            dietary = st.multiselect(
                "Dietary Restrictions",
                ["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Nut-Free"],
            )

        if st.button("✨ Generate Chef Specials", type="primary"):
            if not selected_ingredients:
                st.warning("Please select at least one ingredient.")
            else:
                with st.spinner("Generating Chef Specials with Claude AI..."):
                    try:
                        knowledge = rag.get_ingredient_knowledge(selected_ingredients[:6])
                        ctx = " | ".join(k["text"] for k in knowledge[:3]) if knowledge else None
                        result = chef.generate(
                            ingredients=selected_ingredients,
                            cuisine_preference=cuisine or None,
                            dietary_restrictions=dietary or None,
                            n_suggestions=n_dishes,
                            context=ctx,
                        )

                        if result.get("_source") == "fallback":
                            st.warning("⚠️ Using fallback suggestions (AI key not configured)")
                        else:
                            st.success("✅ Generated using AI")

                        # Display specials
                        for i, special in enumerate(result.get("chef_specials", [])):
                            urgency_colors = {
                                "use_today": "🔴",
                                "use_within_2_days": "🟠",
                                "use_this_week": "🟡",
                            }
                            urgency_icon = urgency_colors.get(special.get("urgency", ""), "🟢")

                            with st.expander(
                                f"{urgency_icon} {special.get('name', f'Dish {i+1}')} "
                                f"({special.get('prep_time_minutes', '?')} min)",
                                expanded=(i == 0),
                            ):
                                col_d1, col_d2 = st.columns([2, 1])
                                with col_d1:
                                    st.markdown(f"**Description:** {special.get('description', '')}")
                                    st.markdown(f"**Ingredients Used:** {', '.join(special.get('ingredients_used', []))}")
                                    st.markdown(f"**Why this helps:** {special.get('waste_reduction_rationale', '')}")
                                with col_d2:
                                    st.markdown(f"**💡 Storage Tip:** {special.get('storage_tip', '')}")
                                    st.markdown(f"**Urgency:** {special.get('urgency', '').replace('_', ' ').title()}")

                        if "general_recommendation" in result:
                            st.info(f"💬 **Overall Recommendation:** {result['general_recommendation']}")
                        if "estimated_waste_reduction_pct" in result:
                            st.metric("Estimated Waste Reduction", f"{result['estimated_waste_reduction_pct']}%")

                        # Show RAG knowledge used
                        if knowledge:
                            with st.expander("📚 Knowledge Base Used"):
                                for k in knowledge[:3]:
                                    st.markdown(f"- {k['text']}")

                    except Exception as e:
                        st.error(f"Error generating specials: {e}")

    # ── Tab 5: Explainability ──────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("#### 🔬 Model Explainability")

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.markdown("**Feature Importance (XGBoost + LightGBM Ensemble)**")
            fi_df = wastage.get_feature_importance()
            if not fi_df.empty:
                fig_fi = px.bar(
                    fi_df.head(12), x="importance", y="feature",
                    orientation="h", title="Top Predictive Features",
                    color="importance", color_continuous_scale="Viridis",
                )
                fig_fi.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_fi, use_container_width=True)

        with col_ex2:
            st.markdown("**Item-Level Risk Explanation**")
            if len(df_pred) > 0:
                item_options = df_pred["ingredient_name"].tolist()
                sel_item = st.selectbox("Select Item to Explain", item_options[:50])
                sel_df = df_pred[df_pred["ingredient_name"] == sel_item].head(1)

                if not sel_df.empty:
                    shap_exp = wastage.explain(sel_df, max_items=1)
                    from src.explainability.explainer import explain_item_risk
                    row = sel_df.iloc[0]
                    explanation = explain_item_risk(
                        row,
                        shap_exp[0]["top_features"] if shap_exp else None
                    )

                    st.metric("Risk Score", f"{explanation['risk_score']:.3f}")
                    st.metric("Risk Level", explanation["risk_level"].upper())
                    st.markdown(f"**Primary Reason:** {explanation['primary_reason']}")
                    st.markdown("**All Reasons:**")
                    for reason in explanation["all_reasons"]:
                        st.markdown(f"  - {reason}")
                    st.info(f"**Recommended Action:** {explanation['recommended_action']}")

        # Model metrics
        st.markdown("---")
        st.markdown("**Model Performance Metrics**")
        metrics = wastage.metrics
        if metrics:
            m_cols = st.columns(len(metrics))
            for i, (k, v) in enumerate(metrics.items()):
                if isinstance(v, (int, float)):
                    m_cols[i].metric(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else v)

    # ── Tab 6: Raw Data ────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("#### 🗄️ Raw Inventory Data")
        st.markdown(f"**{len(df_pred)} records** | Data Quality Score: **{quality.get('data_quality_pct', 100):.1f}%**")

        if st.checkbox("Show quality report"):
            st.json(quality)

        # Filter controls
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            ["critical", "high", "medium", "low"],
            default=["critical", "high"],
        )
        if risk_filter and "risk_level_pred" in df_pred.columns:
            display_df = df_pred[df_pred["risk_level_pred"].isin(risk_filter)]
        else:
            display_df = df_pred

        st.download_button(
            "📥 Download as CSV",
            data=display_df.to_csv(index=False),
            file_name=f"ecomeal_inventory_{date.today().isoformat()}.csv",
            mime="text/csv",
        )

        display_cols = [c for c in [
            "item_id", "restaurant", "ingredient_name", "category", "quantity", "unit",
            "days_to_expiry", "waste_probability", "risk_level_pred", "potential_waste_value",
            "anomaly_score",
        ] if c in display_df.columns]
        st.dataframe(display_df[display_cols], use_container_width=True, height=500)


if __name__ == "__main__":
    main()
