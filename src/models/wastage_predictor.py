"""
Wastage prediction model using XGBoost + LightGBM ensemble.
Predicts waste probability and risk level per inventory item.
Includes SHAP-based explainability.
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
import shap

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
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
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
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
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

        # ── SHAP explainer — use LightGBM which has no version issue ─────────
        try:
            self.shap_explainer = shap.TreeExplainer(self.lgb_model)
        except Exception as e:
            logger.warning(f"SHAP explainer init failed: {e}, continuing without SHAP")
            self.shap_explainer = None

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
        """
        Return SHAP-based explanations: top contributing features per item.
        """
        if self.shap_explainer is None:
            return []

        X = self._prepare_features(df).head(max_items)
        try:
            sv_all = self.shap_explainer.shap_values(X)
            if isinstance(sv_all, list):
                sv_all = sv_all[1]  # class 1 for binary LGB
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return []

        explanations = []
        for idx in range(len(X)):
            sv = sv_all[idx]
            feature_impacts = sorted(
                zip(self.feature_cols, sv),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]
            explanations.append({
                "item_index": idx,
                "top_features": [
                    {
                        "feature": feat,
                        "impact": round(float(val), 4),
                        "direction": "increases_risk" if val > 0 else "decreases_risk",
                    }
                    for feat, val in feature_impacts
                ],
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
            try:
                self.shap_explainer = shap.TreeExplainer(self.lgb_model)
            except Exception:
                self.shap_explainer = None
            self.is_trained = True
            logger.info("Wastage predictor loaded successfully")
            return True
        except FileNotFoundError:
            logger.warning("No saved wastage model found, need to train first")
            return False
        except Exception as e:
            logger.error(f"Error loading wastage model: {e}")
            return False


def _prob_to_risk(prob: float) -> str:
    if prob >= 0.70:
        return "critical"
    elif prob >= 0.45:
        return "high"
    elif prob >= 0.20:
        return "medium"
    return "low"
