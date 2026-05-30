"""
Data preprocessing pipeline — handles missing values, invalid formats,
outliers, and feature engineering. Designed to NEVER crash on bad data.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Tuple

from src.utils.logger import logger


_NUMERIC_DEFAULTS = {
    "quantity": 0.0,
    "daily_consumption": 0.01,
    "days_to_expiry": 0,
    "days_since_purchase": 0,
    "total_shelf_life_days": 30,
    "wastage_history_pct": 0.1,
    "price_per_unit": 50.0,
    "reorder_point": 1.0,
}

_CATEGORY_DEFAULTS = {
    "category": "unknown",
    "storage_type": "ambient",
    "unit": "kg",
    "supplier": "Unknown Supplier",
    "restaurant": "Unknown Restaurant",
    "ingredient_name": "Unknown Ingredient",
    "risk_level": "medium",
}


def _safe_parse_date(val, fallback: date = None) -> date:
    if fallback is None:
        fallback = date.today()
    try:
        if isinstance(val, (date, datetime)):
            return val if isinstance(val, date) else val.date()
        if pd.isna(val) or str(val).strip().upper() in ("INVALID", "NAN", "NONE", ""):
            return fallback
        return pd.to_datetime(str(val)).date()
    except Exception:
        return fallback


def clean_inventory(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Clean and validate inventory dataframe.
    Returns (cleaned_df, quality_report).
    """
    original_size = len(df)
    report = {"original_rows": original_size, "issues": {}}

    df = df.copy()

    # ── Drop full duplicates ──────────────────────────────────────────────────
    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    report["issues"]["duplicate_rows_removed"] = int(n_dupes)

    # ── Fix numeric columns ───────────────────────────────────────────────────
    for col, default in _NUMERIC_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_bad = df[col].isna().sum()
        if n_bad:
            report["issues"][f"{col}_nulls_filled"] = int(n_bad)
        df[col] = df[col].fillna(default)
        # Negative values get defaulted
        if col in ("quantity", "daily_consumption", "price_per_unit", "reorder_point"):
            mask = df[col] < 0
            if mask.any():
                report["issues"][f"{col}_negatives_fixed"] = int(mask.sum())
            df.loc[mask, col] = default

    # Avoid division-by-zero later
    df["daily_consumption"] = df["daily_consumption"].replace(0, 0.01)

    # ── Fix categorical columns ───────────────────────────────────────────────
    for col, default in _CATEGORY_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            continue
        df[col] = df[col].fillna(default).astype(str).str.strip()
        empty_mask = df[col] == ""
        df.loc[empty_mask, col] = default
        if empty_mask.any():
            report["issues"][f"{col}_empty_filled"] = int(empty_mask.sum())

    # ── Fix date columns ──────────────────────────────────────────────────────
    today = date.today()
    for date_col in ("purchase_date", "expiry_date"):
        if date_col not in df.columns:
            df[date_col] = today.isoformat()
            continue
        df[date_col] = df[date_col].apply(lambda v: _safe_parse_date(v, today).isoformat())

    # Recompute days_to_expiry from fixed expiry_date
    df["expiry_date_parsed"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    df["days_to_expiry"] = (df["expiry_date_parsed"] - pd.Timestamp(today)).dt.days.fillna(0).astype(int).clip(lower=0)
    df.drop(columns=["expiry_date_parsed"], inplace=True)

    # ── Feature engineering ───────────────────────────────────────────────────
    df = _engineer_features(df)

    report["cleaned_rows"] = len(df)
    report["data_quality_pct"] = round(len(df) / max(original_size, 1) * 100, 2)

    n_issues = sum(v for v in report["issues"].values() if isinstance(v, int))
    logger.info(
        f"Preprocessing complete: {original_size}→{len(df)} rows, "
        f"{n_issues} total issues fixed"
    )
    return df, report


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features used by ML models."""

    # Ratio: how many days of stock do we have relative to expiry?
    df["stock_days_available"] = (df["quantity"] / df["daily_consumption"]).round(3)
    df["stock_expiry_ratio"] = (
        df["stock_days_available"] / df["days_to_expiry"].replace(0, 0.5)
    ).round(4)

    # Shelf life consumed fraction
    df["shelf_life_consumed_pct"] = (
        df["days_since_purchase"] / df["total_shelf_life_days"].replace(0, 1)
    ).clip(0, 1).round(4)

    # Estimated waste quantity in money
    df["potential_waste_value"] = (
        df["quantity"]
        * df["wastage_history_pct"]
        * df["price_per_unit"]
    ).round(2)

    # Is stock below reorder point?
    df["below_reorder_point"] = (df["quantity"] < df["reorder_point"]).astype(int)

    # Days of stock vs expiry urgency flag
    df["overstock_flag"] = (df["stock_days_available"] > df["days_to_expiry"] * 1.2).astype(int)

    # Encode storage type
    storage_map = {"frozen": 0, "refrigerated": 1, "ambient": 2}
    df["storage_type_enc"] = df["storage_type"].map(storage_map).fillna(2).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode category columns for ML models."""
    for col in ("category", "storage_type"):
        if col in df.columns:
            df[f"{col}_code"] = df[col].astype("category").cat.codes
    return df


def get_feature_columns() -> list:
    """Return ordered list of numeric feature columns for ML models."""
    return [
        "days_to_expiry",
        "days_since_purchase",
        "shelf_life_consumed_pct",
        "quantity",
        "daily_consumption",
        "stock_days_available",
        "stock_expiry_ratio",
        "overstock_flag",
        "below_reorder_point",
        "wastage_history_pct",
        "potential_waste_value",
        "price_per_unit",
        "reorder_point",
        "total_shelf_life_days",
        "storage_type_enc",
        "category_code",
    ]
