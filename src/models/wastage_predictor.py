"""
Wastage prediction model using XGBoost + LightGBM ensemble.
Predicts waste probability and risk level per inventory item.
Includes SHAP-based explainability with plain-English sentences.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

from src.utils.logger import logger
from src.utils.config import get_settings
from src.data.preprocessor import get_feature_columns, encode_categoricals


settings = get_settings()


class WastagePredictor:
    """
    Ensemble wastage classifier combining XGBoost and LightGBM.
    Outputs probability of waste + risk level + SHAP explanations.
    """

    def __init__(self):
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.shap_explainer = None
        self.feature_cols: List[str] = get_feature_columns()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.is_trained: bool = False
        self.model_dir = Path(settings.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, Any] = {}

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = encode_categoricals(df)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_cols].fillna(0)

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Training wastage predictor...")
        df = encode_categoricals(df.copy())
        target = "will_waste"

        if target not in df.columns:
            # Derive from waste_risk_score if missing
            df[target] = (df.get("waste_risk_score", 0.5) >= 0.45).astype(int)

        X = self._prepare_features(df)
        y = df[target].astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=settings.random_seed, stratify=y
        )

        # ── XGBoost ───────────────────────────────────────────────────────────
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=settings.random_seed,
            verbosity=0,
        )
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # ── LightGBM ──────────────────────────────────────────────────────────
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=settings.random_seed,
            verbose=-1,
        )
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        # ── Mark explainer ready — use LightGBM native pred_contrib ─────────
        self.shap_explainer = True  # sentinel: lgb_model.predict with pred_contrib

        # ── Metrics ───────────────────────────────────────────────────────────
        y_prob = self._ensemble_proba(X_val)
        y_pred = (y_prob >= 0.5).astype(int)
        self.metrics = {
            "roc_auc": round(roc_auc_score(y_val, y_prob), 4),
            "avg_precision": round(average_precision_score(y_val, y_prob), 4),
            "f1": round(f1_score(y_val, y_pred), 4),
            "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_val, y_pred, zero_division=0), 4),
            "train_size": len(X_train),
            "val_size": len(X_val),
        }

        self.is_trained = True
        logger.info(f"Training complete: {self.metrics}")
        return self.metrics

    def _ensemble_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of XGB and LGB probabilities."""
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        p_lgb = self.lgb_model.predict_proba(X)[:, 1]
        return 0.55 * p_xgb + 0.45 * p_lgb

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict waste probability and risk level for each row."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X = self._prepare_features(df)
        probs = self._ensemble_proba(X)
        predictions = []
        for i, prob in enumerate(probs):
            predictions.append({
                "waste_probability": round(float(prob), 4),
                "will_waste_pred": int(prob >= 0.5),
                "risk_level_pred": _prob_to_risk(prob),
            })

        return pd.DataFrame(predictions)

    def explain(self, df: pd.DataFrame, max_items: int = 5) -> List[Dict[str, Any]]:
        """Return feature contribution explanations with plain-English sentences per item.

        Uses LightGBM's native pred_contrib (leaf-based SHAP values) — no external
        shap package dependency, works reliably across all environments.
        """
        if self.lgb_model is None:
            return []

        X = self._prepare_features(df).head(max_items)
        try:
            # pred_contrib returns shape (n, n_features + 1); last col is bias
            contribs = self.lgb_model.predict(X, pred_contrib=True)
            sv_all = contribs[:, :-1]  # drop bias column
        except Exception as e:
            logger.warning(f"Feature contribution computation failed: {e}")
            return []

        explanations = []
        for idx in range(len(X)):
            sv = sv_all[idx]
            # Normalise to [−1, 1] so direction is clear regardless of scale
            max_abs = np.abs(sv).max()
            if max_abs > 0:
                sv = sv / max_abs

            feature_impacts = sorted(
                zip(self.feature_cols, sv),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:6]
            top_features = [
                {
                    "feature": feat,
                    "impact": round(float(val), 4),
                    "direction": "increases_risk" if val > 0 else "decreases_risk",
                    "natural_language": _shap_to_sentence(feat, val),
                }
                for feat, val in feature_impacts
            ]
            row_vals = df.iloc[idx] if idx < len(df) else pd.Series()
            explanations.append({
                "item_index": idx,
                "top_features": top_features,
                "summary_sentence": _build_summary_sentence(top_features, row_vals),
            })
        return explanations

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_trained:
            return pd.DataFrame()
        xgb_imp = self.xgb_model.feature_importances_
        lgb_imp = self.lgb_model.feature_importances_
        avg_imp = (xgb_imp + lgb_imp) / 2
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": avg_imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model.")
        joblib.dump(self.xgb_model, self.model_dir / "xgb_wastage.pkl")
        joblib.dump(self.lgb_model, self.model_dir / "lgb_wastage.pkl")
        joblib.dump(self.metrics, self.model_dir / "wastage_metrics.pkl")
        logger.info(f"Wastage predictor saved to {self.model_dir}")

    def load(self) -> bool:
        try:
            self.xgb_model = joblib.load(self.model_dir / "xgb_wastage.pkl")
            self.lgb_model = joblib.load(self.model_dir / "lgb_wastage.pkl")
            self.metrics = joblib.load(self.model_dir / "wastage_metrics.pkl")
            self.shap_explainer = True
            self.is_trained = True
            logger.info("Wastage predictor loaded successfully")
            return True
        except FileNotFoundError:
            logger.warning("No saved wastage model found, need to train first")
            return False
        except Exception as e:
            logger.error(f"Error loading wastage model: {e}")
            return False


_SHAP_TEMPLATES = {
    "days_to_expiry": {
        "increases_risk": "Very few days remain before this item expires, pushing waste risk up.",
        "decreases_risk": "Plenty of time remains before expiry, which keeps waste risk low.",
    },
    "shelf_life_consumed_pct": {
        "increases_risk": "Most of the shelf life has already been used up, increasing spoilage risk.",
        "decreases_risk": "Only a small fraction of shelf life has elapsed, so the item is still fresh.",
    },
    "stock_expiry_ratio": {
        "increases_risk": "Current stock far exceeds what can be consumed before the expiry date.",
        "decreases_risk": "Stock levels are well matched to remaining shelf life.",
    },
    "overstock_flag": {
        "increases_risk": "This item is overstocked relative to expected demand before it expires.",
        "decreases_risk": "Stock quantity is appropriate for the remaining usage window.",
    },
    "wastage_history_pct": {
        "increases_risk": "Historical data shows this ingredient is frequently wasted.",
        "decreases_risk": "This ingredient has a strong track record of being fully used.",
    },
    "quantity": {
        "increases_risk": "A large quantity is on hand that may not be used in time.",
        "decreases_risk": "Only a small quantity remains, reducing the risk of it going to waste.",
    },
    "daily_consumption": {
        "increases_risk": "Daily usage is too low to consume this stock before expiry.",
        "decreases_risk": "High daily usage means stock will be consumed well before it expires.",
    },
    "stock_days_available": {
        "increases_risk": "Current stock will last longer than the item's remaining shelf life.",
        "decreases_risk": "Stock will be consumed before the expiry date at the current usage rate.",
    },
    "potential_waste_value": {
        "increases_risk": "The monetary value at risk is significant if this item is not used soon.",
        "decreases_risk": "The financial impact of potential waste is relatively small.",
    },
    "below_reorder_point": {
        "increases_risk": "Stock is below the reorder threshold, signalling possible supply gaps.",
        "decreases_risk": "Stock levels are above the reorder point, indicating healthy supply.",
    },
    "storage_type_enc": {
        "increases_risk": "The storage conditions for this item raise the spoilage probability.",
        "decreases_risk": "Storage conditions help preserve this item and reduce spoilage risk.",
    },
    "price_per_unit": {
        "increases_risk": "High unit price means wasting even a small amount has a large financial impact.",
        "decreases_risk": "Low unit price limits the financial loss if some quantity is wasted.",
    },
    "total_shelf_life_days": {
        "increases_risk": "This is a short-shelf-life ingredient that spoils quickly by nature.",
        "decreases_risk": "This ingredient has a long shelf life, giving more time to use it.",
    },
    "days_since_purchase": {
        "increases_risk": "The item has been in storage for a long time relative to its shelf life.",
        "decreases_risk": "The item was purchased recently, so it is still well within its usable period.",
    },
    "category_code": {
        "increases_risk": "This ingredient category tends to have higher-than-average wastage rates.",
        "decreases_risk": "This ingredient category typically has low wastage rates.",
    },
}

_SHAP_DEFAULT = {
    "increases_risk": "This factor pushes waste risk higher based on historical patterns.",
    "decreases_risk": "This factor helps keep waste risk lower.",
}


def _shap_to_sentence(feature: str, shap_val: float) -> str:
    direction = "increases_risk" if shap_val > 0 else "decreases_risk"
    templates = _SHAP_TEMPLATES.get(feature, _SHAP_DEFAULT)
    return templates[direction]


def _deduplicate_factors(top_features: List[Dict]) -> tuple:
    """
    Split features into risk/safe lists, keeping only the dominant direction
    when the same semantic concept appears on both sides.
    """
    # Features that measure the same underlying concept — only keep the stronger signal
    _SEMANTIC_GROUPS = [
        {"stock_days_available", "stock_expiry_ratio", "overstock_flag"},
        {"days_to_expiry", "shelf_life_consumed_pct", "days_since_purchase"},
        {"quantity", "potential_waste_value"},
        {"daily_consumption"},
    ]

    seen_groups: set = set()
    risk_drivers, safe_drivers = [], []

    for f in top_features:
        feat = f["feature"]
        group_id = next(
            (i for i, g in enumerate(_SEMANTIC_GROUPS) if feat in g), None
        )
        direction = f["direction"]

        if group_id is not None:
            key = (group_id, direction)
            conflict_key = (group_id, "decreases_risk" if direction == "increases_risk" else "increases_risk")
            if conflict_key in seen_groups:
                # Opposite direction already recorded for this group — skip weaker signal
                continue
            seen_groups.add(key)

        if direction == "increases_risk":
            risk_drivers.append(f)
        else:
            safe_drivers.append(f)

    return risk_drivers, safe_drivers


def _build_summary_sentence(top_features: List[Dict], row: "pd.Series") -> str:
    """Combine top SHAP drivers into a single plain-English summary sentence."""
    risk_drivers, safe_drivers = _deduplicate_factors(top_features)

    _labels = {
        "days_to_expiry": "expiry proximity",
        "shelf_life_consumed_pct": "shelf-life consumption",
        "stock_expiry_ratio": "excess stock vs expiry",
        "overstock_flag": "overstock",
        "wastage_history_pct": "waste history",
        "quantity": "stock quantity",
        "daily_consumption": "low daily usage",
        "stock_days_available": "stock duration",
        "potential_waste_value": "high value at risk",
        "total_shelf_life_days": "short shelf life",
        "days_since_purchase": "time in storage",
        "storage_type_enc": "storage type",
        "price_per_unit": "unit price",
        "category_code": "ingredient category",
        "below_reorder_point": "reorder level",
    }

    if not risk_drivers:
        return "No dominant waste risk factors detected — item appears to be in good shape."

    top_risk_labels = [_labels.get(f["feature"], f["feature"].replace("_", " ")) for f in risk_drivers[:2]]
    parts = " and ".join(top_risk_labels)

    if safe_drivers:
        safe_label = _labels.get(safe_drivers[0]["feature"], safe_drivers[0]["feature"].replace("_", " "))
        return f"Waste risk is mainly driven by {parts}, partially offset by favourable {safe_label}."
    return f"Waste risk is primarily driven by {parts}."


def _prob_to_risk(prob: float) -> str:
    if prob >= 0.70:
        return "critical"
    elif prob >= 0.45:
        return "high"
    elif prob >= 0.20:
        return "medium"
    return "low"
