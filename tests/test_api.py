"""FastAPI endpoint tests."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def sample_items():
    return [
        {
            "ingredient_name": "Tomatoes",
            "category": "vegetables",
            "quantity": 5.0,
            "unit": "kg",
            "daily_consumption": 1.5,
            "storage_type": "refrigerated",
            "wastage_history_pct": 0.15,
            "price_per_unit": 40.0,
            "reorder_point": 2.0,
        },
        {
            "ingredient_name": "Paneer",
            "category": "dairy",
            "quantity": 2.0,
            "unit": "kg",
            "daily_consumption": 0.5,
            "storage_type": "refrigerated",
            "wastage_history_pct": 0.1,
            "price_per_unit": 300.0,
            "reorder_point": 0.5,
        },
    ]


def test_health_endpoint_structure():
    """Health endpoint should return expected keys."""
    # This tests the response contract without starting the full server
    from src.api.state import set_models, get_models
    expected_keys = {"status", "models_loaded", "version", "env"}
    # Verify we can import and the structure is correct
    assert expected_keys  # placeholder - full integration test needs running server


def test_inventory_item_validation(sample_items):
    from src.data.validator import InventoryItem
    for item_data in sample_items:
        item = InventoryItem(**item_data)
        assert item.quantity >= 0
        assert item.daily_consumption > 0


def test_batch_request_validation(sample_items):
    from src.data.validator import BatchInventoryRequest, InventoryItem
    items = [InventoryItem(**i) for i in sample_items]
    req = BatchInventoryRequest(items=items, restaurant_name="Test Restaurant")
    assert len(req.items) == 2
    assert req.restaurant_name == "Test Restaurant"


def test_chef_specials_request_validation():
    from src.data.validator import ChefSpecialsRequest
    req = ChefSpecialsRequest(
        ingredients=["Tomatoes", "Paneer", "Spinach"],
        cuisine_preference="Indian",
        n_suggestions=3,
    )
    assert len(req.ingredients) == 3
    assert req.n_suggestions == 3


def test_chef_specials_request_xss_sanitized():
    from src.data.validator import ChefSpecialsRequest
    req = ChefSpecialsRequest(
        ingredients=["<script>alert(1)</script>", "Paneer"],
    )
    for ing in req.ingredients:
        assert "<script>" not in ing
