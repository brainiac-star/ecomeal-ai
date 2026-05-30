"""Tests for data generation and preprocessing."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generator import generate_inventory_dataset, generate_demand_history
from src.data.preprocessor import clean_inventory, encode_categoricals, get_feature_columns
from src.data.validator import InventoryItem, BatchInventoryRequest


class TestDataGenerator:
    def test_generates_correct_count(self):
        df = generate_inventory_dataset(n_records=100, seed=42)
        assert len(df) == 100

    def test_all_required_columns_present(self):
        df = generate_inventory_dataset(n_records=50, seed=42)
        required = ["item_id", "ingredient_name", "category", "quantity", "expiry_date",
                    "daily_consumption", "storage_type", "wastage_history_pct", "risk_level"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_risk_distribution_realistic(self):
        df = generate_inventory_dataset(n_records=500, seed=42)
        risk_counts = df["risk_level"].value_counts()
        assert len(risk_counts) >= 2  # at least 2 distinct levels
        # No single level should dominate excessively
        max_pct = risk_counts.max() / len(df)
        assert max_pct < 0.85

    def test_dirty_records_injected(self):
        df = generate_inventory_dataset(n_records=200, seed=42)
        # Some nulls should exist before cleaning
        has_nulls = df.isnull().any().any()
        assert has_nulls, "Expected dirty records to be injected"

    def test_demand_history_shape(self):
        ingredients = ["Tomatoes", "Paneer", "Chicken"]
        df = generate_demand_history(ingredients=ingredients, n_days=30, seed=42)
        assert len(df) == 3 * 30
        assert "quantity_used" in df.columns
        assert (df["quantity_used"] >= 0).all()


class TestPreprocessor:
    def test_cleans_null_quantities(self):
        df = generate_inventory_dataset(n_records=100, seed=42)
        df_clean, report = clean_inventory(df)
        assert df_clean["quantity"].isna().sum() == 0

    def test_negative_values_fixed(self):
        df = generate_inventory_dataset(n_records=50, seed=42)
        df.loc[0, "price_per_unit"] = -100
        df_clean, report = clean_inventory(df)
        assert (df_clean["price_per_unit"] >= 0).all()

    def test_invalid_dates_handled(self):
        df = generate_inventory_dataset(n_records=50, seed=42)
        df.loc[0, "expiry_date"] = "INVALID"
        df_clean, report = clean_inventory(df)
        # Should not raise, invalid date replaced
        assert df_clean.shape[0] > 0

    def test_derived_features_created(self):
        df = generate_inventory_dataset(n_records=50, seed=42)
        df_clean, _ = clean_inventory(df)
        for feat in ["stock_days_available", "stock_expiry_ratio", "shelf_life_consumed_pct"]:
            assert feat in df_clean.columns

    def test_never_crashes_on_empty(self):
        df = pd.DataFrame()
        # Should handle gracefully
        df_clean, report = clean_inventory(df)
        assert isinstance(df_clean, pd.DataFrame)

    def test_encode_categoricals(self):
        df = generate_inventory_dataset(n_records=50, seed=42)
        df_clean, _ = clean_inventory(df)
        df_enc = encode_categoricals(df_clean)
        assert "category_code" in df_enc.columns
        assert df_enc["category_code"].dtype in [int, np.int8, np.int64]


class TestValidator:
    def test_valid_item_passes(self):
        item = InventoryItem(
            ingredient_name="Tomatoes",
            category="vegetables",
            quantity=10.5,
            daily_consumption=2.0,
        )
        assert item.ingredient_name == "Tomatoes"

    def test_negative_quantity_rejected(self):
        with pytest.raises(Exception):
            InventoryItem(
                ingredient_name="Bad Item",
                category="vegetables",
                quantity=-5,
                daily_consumption=1.0,
            )

    def test_xss_sanitized(self):
        item = InventoryItem(
            ingredient_name="<script>alert('xss')</script>",
            category="vegetables",
            quantity=1.0,
            daily_consumption=0.5,
        )
        assert "<script>" not in item.ingredient_name

    def test_invalid_storage_type_defaults(self):
        item = InventoryItem(
            ingredient_name="Test",
            category="test",
            quantity=1.0,
            daily_consumption=0.5,
            storage_type="spaceship",
        )
        assert item.storage_type == "ambient"
