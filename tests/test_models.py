"""Tests for ML models."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generator import generate_inventory_dataset, generate_demand_history
from src.data.preprocessor import clean_inventory, encode_categoricals
from src.models.wastage_predictor import WastagePredictor
from src.models.anomaly_detector import InventoryAnomalyDetector


@pytest.fixture(scope="module")
def clean_df():
    df_raw = generate_inventory_dataset(n_records=300, seed=42)
    df, _ = clean_inventory(df_raw)
    return encode_categoricals(df)


class TestWastagePredictor:
    def test_train_returns_metrics(self, clean_df):
        predictor = WastagePredictor()
        metrics = predictor.train(clean_df)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] > 0.5  # better than random

    def test_predict_returns_probabilities(self, clean_df):
        predictor = WastagePredictor()
        predictor.train(clean_df)
        preds = predictor.predict(clean_df)
        assert "waste_probability" in preds.columns
        assert preds["waste_probability"].between(0, 1).all()

    def test_predict_returns_risk_levels(self, clean_df):
        predictor = WastagePredictor()
        predictor.train(clean_df)
        preds = predictor.predict(clean_df)
        valid_levels = {"low", "medium", "high", "critical"}
        assert set(preds["risk_level_pred"].unique()).issubset(valid_levels)

    def test_explain_returns_features(self, clean_df):
        predictor = WastagePredictor()
        predictor.train(clean_df)
        explanations = predictor.explain(clean_df.head(3), max_items=3)
        assert len(explanations) == 3
        for exp in explanations:
            assert "top_features" in exp
            assert len(exp["top_features"]) > 0

    def test_predict_on_single_row(self, clean_df):
        predictor = WastagePredictor()
        predictor.train(clean_df)
        single = clean_df.head(1)
        preds = predictor.predict(single)
        assert len(preds) == 1

    def test_feature_importance_not_empty(self, clean_df):
        predictor = WastagePredictor()
        predictor.train(clean_df)
        fi = predictor.get_feature_importance()
        assert not fi.empty
        assert "feature" in fi.columns


class TestAnomalyDetector:
    def test_fit_and_detect(self, clean_df):
        detector = InventoryAnomalyDetector()
        result = detector.fit(clean_df)
        assert "n_anomalies_detected" in result

        detected = detector.detect(clean_df)
        assert "is_anomaly" in detected.columns
        assert "anomaly_score" in detected.columns

    def test_anomaly_rate_reasonable(self, clean_df):
        detector = InventoryAnomalyDetector()
        detector.fit(clean_df)
        detected = detector.detect(clean_df)
        anomaly_rate = detected["is_anomaly"].mean()
        assert anomaly_rate < 0.15  # less than 15%

    def test_anomaly_reasons_provided(self, clean_df):
        detector = InventoryAnomalyDetector()
        detector.fit(clean_df)
        detected = detector.detect(clean_df)
        anomalies = detected[detected["is_anomaly"]]
        if not anomalies.empty:
            assert "anomaly_reason" in anomalies.columns
