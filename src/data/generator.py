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

# Each restaurant has a management quality profile that affects ordering behaviour,
# waste rates, and pricing tier. High-quality ops = tighter stock, lower waste.
RESTAURANT_PROFILES = {
    "The Spice Garden":    {"mgmt": 0.85, "price_tier": 1.2, "order_freq_days": 3},
    "Urban Kitchen":       {"mgmt": 0.70, "price_tier": 1.0, "order_freq_days": 4},
    "Green Leaf Bistro":   {"mgmt": 0.90, "price_tier": 1.4, "order_freq_days": 2},
    "The Curry House":     {"mgmt": 0.55, "price_tier": 0.9, "order_freq_days": 5},
    "Bay Leaf Restaurant": {"mgmt": 0.75, "price_tier": 1.1, "order_freq_days": 3},
    "Savanna Grill":       {"mgmt": 0.60, "price_tier": 1.0, "order_freq_days": 5},
}

# Reorder lead times and order cycle by category (days)
CATEGORY_ORDER_PROFILE = {
    "vegetables":         {"lead_days": 1, "order_cycle": 3,  "batch_multiplier": (1.5, 3.0)},
    "dairy":              {"lead_days": 1, "order_cycle": 3,  "batch_multiplier": (1.5, 3.0)},
    "meat_seafood":       {"lead_days": 1, "order_cycle": 2,  "batch_multiplier": (1.2, 2.5)},
    "grains_pulses":      {"lead_days": 3, "order_cycle": 21, "batch_multiplier": (14.0, 30.0)},
    "spices_condiments":  {"lead_days": 3, "order_cycle": 30, "batch_multiplier": (20.0, 45.0)},
    "oils_fats":          {"lead_days": 2, "order_cycle": 14, "batch_multiplier": (10.0, 20.0)},
    "beverages":          {"lead_days": 1, "order_cycle": 4,  "batch_multiplier": (2.0, 5.0)},
    "bakery":             {"lead_days": 1, "order_cycle": 2,  "batch_multiplier": (1.0, 2.0)},
}


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
        cat_profile = CATEGORY_ORDER_PROFILE[category]

        ingredient = random.choice(meta["items"])
        unit = random.choice(meta["units"])
        storage_type = random.choice(meta["storage"])

        # Each restaurant has preferred suppliers — 70% chance of using top-2
        restaurant = random.choice(RESTAURANTS)
        rp = RESTAURANT_PROFILES[restaurant]
        preferred_suppliers = SUPPLIERS[:4] if rp["price_tier"] >= 1.1 else SUPPLIERS[3:]
        supplier = random.choices(
            preferred_suppliers + SUPPLIERS,
            weights=[3] * len(preferred_suppliers) + [1] * len(SUPPLIERS),
            k=1,
        )[0]

        shelf_life_min, shelf_life_max = meta["shelf_life_days"]
        total_shelf_life = int(np.random.randint(shelf_life_min, shelf_life_max + 1))

        # Daily consumption: category-scaled with lognormal noise
        base_consumption = np.random.uniform(0.5, 15)
        daily_consumption = round(base_consumption * np.random.lognormal(0, 0.15), 3)

        # Order cycle: poorly managed restaurants order less frequently
        order_cycle = cat_profile["order_cycle"] * np.random.uniform(0.8, 1.3)
        order_cycle *= (1.0 + (1.0 - rp["mgmt"]) * 0.5)  # poor mgmt → longer cycles

        # Days since last order: 0 to one full order cycle
        days_since_purchase = int(np.random.uniform(0, max(order_cycle, 1)))
        # Cap at 65% of shelf life so item isn't already expired
        days_since_purchase = min(days_since_purchase, int(total_shelf_life * 0.65))

        purchase_date = reference_date - timedelta(days=days_since_purchase)
        expiry_date = purchase_date + timedelta(days=total_shelf_life)
        days_to_expiry = max(0, (expiry_date - reference_date).days)

        # Ordered qty = one order cycle worth + safety buffer
        # Poorly managed restaurants over-order (buffer 1.5-2.5x vs 1.0-1.5x)
        if rp["mgmt"] >= 0.75:
            buffer = np.random.uniform(1.0, 1.5)
        else:
            buffer = np.random.uniform(1.4, 2.5)
        ordered_qty = daily_consumption * order_cycle * buffer

        # Consumed since purchase (with ±10% usage variability)
        consumed = daily_consumption * days_since_purchase * np.random.uniform(0.9, 1.1)
        quantity = round(max(ordered_qty - consumed, daily_consumption * 0.3), 3)

        # Reorder point: lead_time days + safety stock
        safety_days = cat_profile["lead_days"] + cat_profile["order_cycle"] * 0.3
        reorder_point = round(daily_consumption * safety_days * np.random.uniform(0.9, 1.1), 3)

        # Waste history: correlated with management quality and storage type
        waste_min, waste_max = meta["waste_rate"]
        # Poor management → waste skews toward the high end
        waste_skew = 1.0 - rp["mgmt"] * 0.5
        wastage_history_pct = round(
            np.random.beta(2 * waste_skew, 2 * (1 - waste_skew) + 1)
            * (waste_max - waste_min) + waste_min, 4
        )

        # Price: category range scaled by restaurant price tier + small noise
        price_min, price_max = meta["price_range"]
        price_per_unit = round(
            np.random.uniform(price_min, price_max) * rp["price_tier"]
            * np.random.uniform(0.95, 1.05), 2
        )

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

    # Force realistic risk distribution: ~8% critical, ~15% high, ~30% medium, ~47% low
    # Override a fraction of rows to represent genuinely at-risk items
    n_critical = int(n_records * 0.08)
    n_high     = int(n_records * 0.15)
    rng = np.random.default_rng(seed + 1)
    all_idx = rng.permutation(n_records)
    crit_idx = all_idx[:n_critical]
    high_idx = all_idx[n_critical:n_critical + n_high]

    # Critical: expiring in 0-2 days, overstocked
    for idx in crit_idx:
        df.loc[idx, "days_to_expiry"]  = int(rng.integers(0, 3))
        df.loc[idx, "quantity"]        = round(df.loc[idx, "daily_consumption"] * rng.uniform(8, 20), 3)
        df.loc[idx, "will_waste"]      = 1
        df.loc[idx, "waste_risk_score"] = round(rng.uniform(0.70, 0.95), 4)
        df.loc[idx, "risk_level"]      = "critical"

    # High: expiring in 3-7 days with excess stock
    for idx in high_idx:
        df.loc[idx, "days_to_expiry"]  = int(rng.integers(3, 8))
        df.loc[idx, "quantity"]        = round(df.loc[idx, "daily_consumption"] * rng.uniform(5, 12), 3)
        df.loc[idx, "will_waste"]      = int(rng.random() < 0.75)
        df.loc[idx, "waste_risk_score"] = round(rng.uniform(0.45, 0.70), 4)
        df.loc[idx, "risk_level"]      = "high"

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
