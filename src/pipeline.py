"""
Main Ecomeal AI pipeline.
Run this to generate data, train all models, and produce a full analysis report.
"""

import json
import os
from pathlib import Path
from datetime import date

import pandas as pd

from src.utils.logger import logger
from src.utils.config import get_settings
from src.data.generator import generate_inventory_dataset, generate_demand_history
from src.data.preprocessor import clean_inventory, encode_categoricals
from src.models.wastage_predictor import WastagePredictor
from src.models.demand_forecaster import DemandForecaster
from src.models.anomaly_detector import InventoryAnomalyDetector
from src.recommendations.chef_specials import ChefSpecialsEngine
from src.recommendations.rag_engine import RAGRecommendationEngine
from src.explainability.explainer import generate_portfolio_summary

settings = get_settings()


def run_full_pipeline(
    n_records: int = None,
    save_models: bool = True,
    save_data: bool = True,
    generate_report: bool = True,
) -> dict:
    n_records = n_records or settings.dataset_size
    Path(settings.model_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ECOMEAL AI — FULL PIPELINE RUN")
    logger.info("=" * 60)

    # ── Step 1: Generate dataset ──────────────────────────────────────────────
    logger.info(f"[1/7] Generating {n_records} inventory records...")
    df_raw = generate_inventory_dataset(n_records=n_records, seed=settings.random_seed)
    if save_data:
        raw_path = Path(settings.data_dir) / "inventory_raw.parquet"
        df_raw.to_parquet(raw_path, index=False)
        logger.info(f"Raw data saved: {raw_path}")

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    logger.info("[2/7] Preprocessing and feature engineering...")
    df, quality_report = clean_inventory(df_raw)
    df = encode_categoricals(df)
    if save_data:
        clean_path = Path(settings.data_dir) / "inventory_clean.parquet"
        df.to_parquet(clean_path, index=False)
    logger.info(f"Quality report: {quality_report}")

    # ── Step 3: Train wastage predictor ──────────────────────────────────────
    logger.info("[3/7] Training wastage predictor (XGBoost + LightGBM ensemble)...")
    wastage = WastagePredictor()
    wastage_metrics = wastage.train(df)
    if save_models:
        wastage.save()
    logger.info(f"Wastage model metrics: {wastage_metrics}")

    # ── Step 4: Train anomaly detector ───────────────────────────────────────
    logger.info("[4/7] Fitting anomaly detector (Isolation Forest)...")
    anomaly = InventoryAnomalyDetector()
    anomaly_results = anomaly.fit(df)
    if save_models:
        anomaly.save()

    # ── Step 5: Train demand forecaster ──────────────────────────────────────
    logger.info("[5/7] Generating demand history and fitting forecasters (Prophet)...")
    ingredients = df["ingredient_name"].unique().tolist()[:40]
    demand_df = generate_demand_history(ingredients=ingredients, n_days=180, seed=settings.random_seed)
    forecaster = DemandForecaster()
    forecaster.fit(demand_df)
    if save_models:
        forecaster.save()

    # ── Step 6: Build RAG index ───────────────────────────────────────────────
    logger.info("[6/7] Building RAG knowledge index...")
    rag = RAGRecommendationEngine()
    rag.build_index()
    if save_models:
        rag.save()

    # ── Step 7: Generate predictions and report ───────────────────────────────
    logger.info("[7/7] Running full predictions and generating report...")
    predictions_df = wastage.predict(df)
    df_with_preds = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)

    anomaly_df = anomaly.detect(df_with_preds)
    overstock_df = forecaster.detect_overstock(df_with_preds, horizon_days=14)
    portfolio = generate_portfolio_summary(df_with_preds)

    # Chef Specials for top risky ingredients
    chef = ChefSpecialsEngine()
    high_risk_ingredients = (
        df_with_preds[df_with_preds["risk_level_pred"].isin(["critical", "high"])]
        ["ingredient_name"]
        .unique()
        .tolist()[:8]
    )
    chef_specials = {}
    if high_risk_ingredients:
        knowledge = rag.get_ingredient_knowledge(high_risk_ingredients[:6])
        ctx = " | ".join(k["text"] for k in knowledge[:3])
        chef_specials = chef.generate(
            ingredients=high_risk_ingredients[:6],
            n_suggestions=3,
            context=ctx,
        )

    # Feature importance
    fi = wastage.get_feature_importance().head(10).to_dict("records")

    report = {
        "pipeline_date": date.today().isoformat(),
        "dataset": {
            "total_records": len(df_raw),
            "clean_records": len(df),
            "quality_report": quality_report,
        },
        "wastage_model": wastage_metrics,
        "anomaly_detection": anomaly_results,
        "portfolio_summary": portfolio,
        "forecast_ingredients": len(forecaster.models),
        "overstock_risks": int(overstock_df["overstock_risk"].sum()),
        "shortage_risks": int(overstock_df["shortage_risk"].sum()),
        "chef_specials_generated": len(chef_specials.get("chef_specials", [])),
        "top_feature_importance": fi,
        "high_risk_ingredients": high_risk_ingredients,
    }

    if generate_report:
        report_path = Path(settings.data_dir) / "pipeline_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved: {report_path}")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Total records: {len(df)}")
    logger.info(f"  Critical items: {portfolio.get('critical_count', 0)}")
    logger.info(f"  High risk items: {portfolio.get('high_risk_count', 0)}")
    logger.info(f"  Waste value at risk: ₹{portfolio.get('waste_value_at_risk_inr', 0):,.2f}")
    logger.info(f"  ROC-AUC: {wastage_metrics.get('roc_auc', 'N/A')}")
    logger.info("=" * 60)

    return report


if __name__ == "__main__":
    run_full_pipeline()
