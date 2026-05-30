"""
Synthetic restaurant inventory dataset generator.
Simulates realistic multi-restaurant inventory with seasonal patterns,
waste histories, and demand variability.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from faker import Faker
from typing import Optional
import random

from src.utils.logger import logger

fake = Faker("en_IN")

CATEGORIES = {
    "vegetables": {
        "items": ["Tomatoes", "Onions", "Potatoes", "Spinach", "Mushrooms", "Bell Peppers",
                  "Carrots", "Broccoli", "Cauliflower", "Peas", "Cabbage", "Zucchini",
                  "Eggplant", "Corn", "Cucumbers", "Lettuce", "Kale", "Beets"],
        "shelf_life_days": (2, 14),
        "units": ["kg"],
        "storage": ["refrigerated", "ambient"],
        "waste_rate": (0.05, 0.30),
        "price_range": (20, 120),
    },
    "dairy": {
        "items": ["Paneer", "Butter", "Cream", "Curd", "Milk", "Cheese", "Ghee", "Khoa"],
        "shelf_life_days": (3, 30),
        "units": ["kg", "liters"],
        "storage": ["refrigerated"],
        "waste_rate": (0.03, 0.20),
        "price_range": (50, 600),
    },
    "meat_seafood": {
        "items": ["Chicken Breast", "Mutton", "Fish Fillet", "Prawns", "Crab",
                  "Chicken Wings", "Lamb Chops", "Tuna", "Salmon"],
        "shelf_life_days": (1, 5),
        "units": ["kg"],
        "storage": ["frozen", "refrigerated"],
        "waste_rate": (0.05, 0.25),
        "price_range": (150, 900),
    },
    "grains_pulses": {
        "items": ["Basmati Rice", "Wheat Flour", "Chickpeas", "Lentils", "Black Beans",
                  "Quinoa", "Oats", "Semolina", "Moong Dal", "Rajma"],
        "shelf_life_days": (90, 365),
        "units": ["kg"],
        "storage": ["ambient"],
        "waste_rate": (0.01, 0.08),
        "price_range": (40, 300),
    },
    "spices_condiments": {
        "items": ["Turmeric", "Cumin", "Coriander Powder", "Chili Powder", "Garam Masala",
                  "Black Pepper", "Cardamom", "Cinnamon", "Soy Sauce", "Vinegar"],
        "shelf_life_days": (180, 730),
        "units": ["kg", "liters"],
        "storage": ["ambient"],
        "waste_rate": (0.01, 0.05),
        "price_range": (80, 1200),
    },
    "oils_fats": {
        "items": ["Sunflower Oil", "Olive Oil", "Coconut Oil", "Mustard Oil", "Butter Oil"],
        "shelf_life_days": (180, 540),
        "units": ["liters"],
        "storage": ["ambient"],
        "waste_rate": (0.02, 0.10),
        "price_range": (100, 900),
    },
    "beverages": {
        "items": ["Orange Juice", "Coconut Water", "Sugarcane Juice", "Mango Pulp",
                  "Tomato Juice", "Mixed Fruit Juice"],
        "shelf_life_days": (3, 30),
        "units": ["liters"],
        "storage": ["refrigerated"],
        "waste_rate": (0.05, 0.25),
        "price_range": (40, 200),
    },
    "bakery": {
        "items": ["Bread", "Burger Buns", "Pizza Dough", "Croissants", "Pita Bread",
                  "Naan Dough", "Cake Base"],
        "shelf_life_days": (1, 7),
        "units": ["kg", "pieces"],
        "storage": ["ambient", "refrigerated"],
        "waste_rate": (0.08, 0.35),
        "price_range": (30, 300),
    },
}

SUPPLIERS = [
    "FreshFarm Co.", "Metro Cash & Carry", "BigBasket B2B", "Ninjacart",
    "Jumbotail", "WayCool Foods", "Grofers B2B", "ITC Foods",
    "Mother Dairy", "Amul Wholesale",
]

RESTAURANTS = [
    "The Spice Garden", "Urban Kitchen", "Green Leaf Bistro",
    "The Curry House", "Bay Leaf Restaurant", "Savanna Grill",
]


def _compute_waste_risk(row: pd.Series) -> float:
    """
    Rule-based waste risk score (0-1) accounting for:
    - days to expiry relative to shelf life
    - quantity vs consumption rate
    - historical waste rate
    - storage type risk
    """
    dte_ratio = row["days_to_expiry"] / max(row["total_shelf_life_days"], 1)
    if dte_ratio < 0.1:
        expiry_score = 1.0
    elif dte_ratio < 0.25:
        expiry_score = 0.8
    elif dte_ratio < 0.5:
        expiry_score = 0.4
    else:
        expiry_score = 0.1

    days_of_stock = row["quantity"] / max(row["daily_consumption"], 0.01)
    if days_of_stock > row["days_to_expiry"] * 1.5:
        stock_score = 0.9
    elif days_of_stock > row["days_to_expiry"]:
        stock_score = 0.6
    elif days_of_stock > row["days_to_expiry"] * 0.8:
        stock_score = 0.3
    else:
        stock_score = 0.1

    storage_penalty = {"frozen": 0.0, "ambient": 0.1, "refrigerated": 0.05}.get(
        row["storage_type"], 0.1
    )

    score = (
        expiry_score * 0.45
        + stock_score * 0.30
        + row["wastage_history_pct"] * 0.20
        + storage_penalty * 0.05
    )
    return min(round(float(score), 4), 1.0)


def _risk_label(score: float) -> str:
    if score >= 0.7:
        return "critical"
    elif score >= 0.45:
        return "high"
    elif score >= 0.20:
        return "medium"
    return "low"


def generate_inventory_dataset(
    n_records: int = 1200,
    seed: int = 42,
    reference_date: Optional[date] = None,
) -> pd.DataFrame:
    """Generate a synthetic restaurant inventory dataset with n_records rows."""
    np.random.seed(seed)
    random.seed(seed)

    if reference_date is None:
        reference_date = date.today()

    logger.info(f"Generating {n_records} inventory records (seed={seed})")

    rows = []
    for i in range(n_records):
        category = random.choices(
            list(CATEGORIES.keys()),
            weights=[5, 4, 3, 4, 3, 2, 2, 3],
            k=1,
        )[0]
        meta = CATEGORIES[category]

        ingredient = random.choice(meta["items"])
        unit = random.choice(meta["units"])
        storage_type = random.choice(meta["storage"])
        supplier = random.choice(SUPPLIERS)
        restaurant = random.choice(RESTAURANTS)

        shelf_life_min, shelf_life_max = meta["shelf_life_days"]
        total_shelf_life = int(np.random.randint(shelf_life_min, shelf_life_max + 1))

        # Purchase date: between 0 and 60% of shelf life ago
        days_since_purchase = int(np.random.uniform(0, total_shelf_life * 0.6))
        purchase_date = reference_date - timedelta(days=days_since_purchase)
        expiry_date = purchase_date + timedelta(days=total_shelf_life)
        days_to_expiry = max(0, (expiry_date - reference_date).days)

        # Daily consumption with noise
        base_consumption = np.random.uniform(0.5, 15)
        daily_consumption = round(
            base_consumption * np.random.lognormal(0, 0.2), 3
        )

        # Quantity: plausible relative to consumption and remaining shelf life
        expected_usage = daily_consumption * max(days_to_expiry, 1)
        quantity = round(
            expected_usage * np.random.uniform(0.5, 2.5), 3
        )
        quantity = max(quantity, 0.1)

        waste_min, waste_max = meta["waste_rate"]
        wastage_history_pct = round(np.random.uniform(waste_min, waste_max), 4)

        price_min, price_max = meta["price_range"]
        price_per_unit = round(np.random.uniform(price_min, price_max), 2)

        reorder_point = round(daily_consumption * np.random.randint(3, 8), 3)

        row = {
            "item_id": f"ITEM-{i+1:05d}",
            "restaurant": restaurant,
            "ingredient_name": ingredient,
            "category": category,
            "quantity": quantity,
            "unit": unit,
            "purchase_date": purchase_date.isoformat(),
            "expiry_date": expiry_date.isoformat(),
            "days_since_purchase": days_since_purchase,
            "days_to_expiry": days_to_expiry,
            "total_shelf_life_days": total_shelf_life,
            "daily_consumption": daily_consumption,
            "storage_type": storage_type,
            "supplier": supplier,
            "wastage_history_pct": wastage_history_pct,
            "price_per_unit": price_per_unit,
            "reorder_point": reorder_point,
        }

        row["waste_risk_score"] = _compute_waste_risk(pd.Series(row))
        row["risk_level"] = _risk_label(row["waste_risk_score"])
        # Add 15% label noise to simulate real-world uncertainty
        if np.random.random() < 0.15:
            row["will_waste"] = int(np.random.random() < 0.5)
        else:
            row["will_waste"] = int(row["waste_risk_score"] >= 0.45)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Intentionally inject ~5% dirty records to test robustness
    n_dirty = max(10, int(n_records * 0.05))
    dirty_idx = np.random.choice(df.index, n_dirty, replace=False)
    df.loc[dirty_idx[:n_dirty // 4], "quantity"] = np.nan
    df.loc[dirty_idx[n_dirty // 4: n_dirty // 2], "daily_consumption"] = 0
    df.loc[dirty_idx[n_dirty // 2: 3 * n_dirty // 4], "expiry_date"] = "INVALID"
    df.loc[dirty_idx[3 * n_dirty // 4:], "price_per_unit"] = -1

    logger.info(
        f"Dataset generated: {len(df)} rows, "
        f"{df['risk_level'].value_counts().to_dict()} risk distribution"
    )
    return df


def generate_demand_history(
    ingredients: list,
    n_days: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate daily demand history for time-series forecasting."""
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"Generating {n_days}-day demand history for {len(ingredients)} ingredients")

    rows = []
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=n_days - 1)

    for ingredient in ingredients:
        # Each ingredient gets a base demand + weekly seasonality + trend + noise
        base = np.random.uniform(5, 50)
        trend = np.random.uniform(-0.05, 0.05)
        weekly_amp = np.random.uniform(0, 0.3) * base

        for i, d in enumerate(
            (start_date + timedelta(days=j) for j in range(n_days))
        ):
            dow = d.weekday()
            is_weekend = int(dow >= 5)
            is_monday = int(dow == 0)

            weekly_effect = weekly_amp * np.sin(2 * np.pi * dow / 7)
            trend_effect = base * trend * i
            noise = np.random.normal(0, base * 0.1)
            spike = base * 0.5 if np.random.random() < 0.03 else 0  # random demand spike

            qty = max(0.0, round(base + weekly_effect + trend_effect + noise + spike, 3))
            rows.append(
                {
                    "date": d.isoformat(),
                    "ingredient_name": ingredient,
                    "quantity_used": qty,
                    "day_of_week": dow,
                    "is_weekend": is_weekend,
                    "is_monday": is_monday,
                }
            )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
