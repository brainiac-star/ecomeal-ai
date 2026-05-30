"""
Explainability layer: SHAP-based and rule-based explanations.
Answers WHY an ingredient is risky and WHY a recommendation was made.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


_FEATURE_DESCRIPTIONS = {
    "days_to_expiry": "days remaining before expiry",
    "shelf_life_consumed_pct": "fraction of shelf life already consumed",
    "stock_expiry_ratio": "ratio of stock days available to days until expiry",
    "overstock_flag": "whether current stock exceeds expiry-adjusted demand",
    "wastage_history_pct": "historical waste rate for this ingredient",
    "quantity": "current quantity in stock",
    "daily_consumption": "average daily usage",
    "stock_days_available": "how many days current stock will last at current consumption",
    "potential_waste_value": "estimated monetary value at risk of wastage",
    "below_reorder_point": "whether stock is below the reorder threshold",
    "storage_type_enc": "storage environment (frozen=0, refrigerated=1, ambient=2)",
    "category_code": "ingredient category",
    "price_per_unit": "unit price of ingredient",
    "reorder_point": "quantity level at which to reorder",
    "total_shelf_life_days": "total shelf life of ingredient",
    "days_since_purchase": "days since the ingredient was purchased",
}


def explain_item_risk(row: pd.Series, shap_impacts: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive human-readable explanation for why
    an inventory item is at risk of being wasted.
    """
    reasons = []
    risk_score = float(row.get("waste_risk_score", row.get("waste_probability", 0.5)))
    dte = int(row.get("days_to_expiry", 0))
    qty = float(row.get("quantity", 0))
    daily = float(row.get("daily_consumption", 0.01))
    shelf_pct = float(row.get("shelf_life_consumed_pct", 0))
    stock_ratio = float(row.get("stock_expiry_ratio", 1))
    waste_hist = float(row.get("wastage_history_pct", 0))
    overstock = bool(row.get("overstock_flag", False))

    # Expiry-based reasons
    if dte == 0:
        reasons.append("Item expires TODAY — immediate action required")
    elif dte <= 2:
        reasons.append(f"Item expires in {dte} day(s) — use urgently")
    elif dte <= 5:
        reasons.append(f"Item expires in {dte} days — plan usage within 2 days")

    if shelf_pct >= 0.8:
        reasons.append(f"{shelf_pct*100:.0f}% of shelf life consumed")

    # Stock vs consumption reasons
    stock_days = qty / max(daily, 0.01)
    if overstock or stock_days > dte * 1.5:
        reasons.append(
            f"Overstock: {stock_days:.1f} days of supply but only {dte} days until expiry"
        )
    elif stock_ratio > 2:
        reasons.append(f"Stock-to-expiry ratio is {stock_ratio:.1f}x (ideal: <1.0)")

    # Historical patterns
    if waste_hist > 0.25:
        reasons.append(f"High historical waste rate: {waste_hist*100:.0f}%")

    # SHAP-based reasons (from ML model)
    if shap_impacts:
        for impact in shap_impacts[:3]:
            feat = impact.get("feature", "")
            direction = impact.get("direction", "")
            feat_desc = _FEATURE_DESCRIPTIONS.get(feat, feat)
            if direction == "increases_risk":
                reasons.append(f"ML model: {feat_desc} increases waste probability")

    if not reasons:
        reasons.append(f"Moderate risk based on combined inventory signals (score: {risk_score:.2f})")

    return {
        "risk_score": round(risk_score, 3),
        "risk_level": _score_to_level(risk_score),
        "primary_reason": reasons[0] if reasons else "No specific risk detected",
        "all_reasons": reasons,
        "recommended_action": _get_action(risk_score, dte, overstock),
    }


def explain_recommendation(
    ingredient: str,
    dish: str,
    usage_rationale: str,
    retrieved_knowledge: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Explain why a specific dish was recommended for an ingredient."""
    knowledge_used = []
    if retrieved_knowledge:
        for k in retrieved_knowledge[:3]:
            if any(tag.lower() in ingredient.lower() for tag in k.get("tags", [])):
                knowledge_used.append(k["text"])

    return {
        "dish": dish,
        "ingredient": ingredient,
        "rationale": usage_rationale,
        "knowledge_references": knowledge_used,
        "explanation": (
            f"'{dish}' was recommended because it efficiently uses '{ingredient}' "
            f"which is at risk of wastage. "
            + (f"Based on culinary knowledge: {knowledge_used[0]}" if knowledge_used else "")
        ),
    }


def generate_portfolio_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize the waste risk portfolio for an entire inventory.
    """
    if df.empty:
        return {"error": "Empty inventory"}

    risk_col = "risk_level_pred" if "risk_level_pred" in df.columns else "risk_level"
    prob_col = "waste_probability" if "waste_probability" in df.columns else "waste_risk_score"

    risk_counts = df[risk_col].value_counts().to_dict() if risk_col in df.columns else {}

    waste_value_at_risk = 0.0
    if "potential_waste_value" in df.columns:
        high_risk_mask = df.get(risk_col, pd.Series()).isin(["high", "critical"])
        waste_value_at_risk = df.loc[high_risk_mask, "potential_waste_value"].sum()

    top_at_risk = []
    if risk_col in df.columns and prob_col in df.columns:
        top_df = (
            df[df[risk_col].isin(["critical", "high"])]
            .nlargest(5, prob_col)[["ingredient_name", prob_col, "days_to_expiry", "quantity"]]
        )
        top_at_risk = top_df.to_dict("records")

    return {
        "total_items": len(df),
        "risk_distribution": risk_counts,
        "critical_count": risk_counts.get("critical", 0),
        "high_risk_count": risk_counts.get("high", 0),
        "waste_value_at_risk_inr": round(waste_value_at_risk, 2),
        "top_at_risk_items": top_at_risk,
        "overall_risk_score": round(
            df[prob_col].mean() if prob_col in df.columns else 0.5, 3
        ),
    }


def _score_to_level(score: float) -> str:
    if score >= 0.70:
        return "critical"
    elif score >= 0.45:
        return "high"
    elif score >= 0.20:
        return "medium"
    return "low"


def _get_action(risk_score: float, dte: int, overstock: bool) -> str:
    if risk_score >= 0.70 or dte <= 1:
        return "Use immediately today — prepare or freeze now"
    elif risk_score >= 0.45 or dte <= 3:
        return "Create Chef Specials featuring this ingredient within 2 days"
    elif overstock:
        return "Plan high-volume usage and reduce next order quantity"
    elif risk_score >= 0.20:
        return "Monitor closely and include in upcoming menu planning"
    return "No immediate action needed — continue normal usage"
