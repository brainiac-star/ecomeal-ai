"""
Anomaly detection for inventory data using Isolation Forest.
Flags items with unusual quantity, consumption, or pricing patterns.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils.logger import logger
from src.utils.config import get_settings

settings = get_settings()


class InventoryAnomalyDetector:
    """
    Detects anomalous inventory records via Isolation Forest.
    Anomalies may indicate: data entry errors, theft, spoilage events,
    demand spikes, or supplier issues.
    """

    ANOMALY_FEATURES = [
        "quantity",
        "daily_consumption",
        "stock_expiry_ratio",
        "wastage_history_pct",
        "price_per_unit",
        "potential_waste_value",
        "days_to_expiry",
        "shelf_life_consumed_pct",
    ]

    def __init__(self):
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False
        self.model_dir = Path(settings.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.anomaly_threshold: float = -0.5

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.ANOMALY_FEATURES if c in df.columns]
        X = df[available].copy().fillna(0)
        return X

    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Fitting anomaly detector...")
        X = self._get_features(df)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        contamination = min(0.05, max(0.01, 10 / max(len(df), 100)))
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples="auto",
            random_state=settings.random_seed,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_fitted = True

        scores = self.model.score_samples(X_scaled)
        n_anomalies = (scores < self.anomaly_threshold).sum()
        result = {
            "n_samples": len(df),
            "n_anomalies_detected": int(n_anomalies),
            "anomaly_rate": round(n_anomalies / len(df), 4),
        }
        logger.info(f"Anomaly detector fitted: {result}")
        return result

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns df with added columns:
        - anomaly_score: lower = more anomalous
        - is_anomaly: bool
        - anomaly_reason: short human-readable explanation
        """
        if not self.is_fitted:
            raise RuntimeError("Anomaly detector not fitted. Call fit() or load() first.")

        X = self._get_features(df)
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)
        is_anomaly = scores < self.anomaly_threshold

        result = df.copy()
        result["anomaly_score"] = np.round(scores, 4)
        result["is_anomaly"] = is_anomaly

        reasons = []
        for i, row in result.iterrows():
            if not row["is_anomaly"]:
                reasons.append(None)
            else:
                reasons.append(_infer_anomaly_reason(row))
        result["anomaly_reason"] = reasons

        return result

    def get_top_anomalies(self, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        detected = self.detect(df)
        return (
            detected[detected["is_anomaly"]]
            .sort_values("anomaly_score")
            .head(n)
        )

    def save(self) -> None:
        joblib.dump(self.model, self.model_dir / "anomaly_detector.pkl")
        joblib.dump(self.scaler, self.model_dir / "anomaly_scaler.pkl")
        logger.info("Anomaly detector saved")

    def load(self) -> bool:
        try:
            self.model = joblib.load(self.model_dir / "anomaly_detector.pkl")
            self.scaler = joblib.load(self.model_dir / "anomaly_scaler.pkl")
            self.is_fitted = True
            logger.info("Anomaly detector loaded")
            return True
        except FileNotFoundError:
            logger.warning("No saved anomaly detector found")
            return False
        except Exception as e:
            logger.error(f"Error loading anomaly detector: {e}")
            return False


def _infer_anomaly_reason(row: pd.Series) -> str:
    """Provide a human-readable reason for why this item is anomalous."""
    reasons = []

    qty = row.get("quantity", 0)
    consumption = row.get("daily_consumption", 0.01)
    dte = row.get("days_to_expiry", 0)
    price = row.get("price_per_unit", 0)

    stock_days = qty / max(consumption, 0.01)
    if stock_days > dte * 3:
        reasons.append(f"extreme overstock ({stock_days:.0f} days of stock, expires in {dte} days)")
    elif qty == 0:
        reasons.append("zero quantity — possible data entry error or theft")

    if consumption > 100:
        reasons.append(f"unusually high daily consumption ({consumption:.1f})")
    elif consumption < 0.01:
        reasons.append("near-zero consumption — item may be unused")

    if price < 0:
        reasons.append("negative price — data error")
    elif price > 5000:
        reasons.append(f"unusually high price (₹{price:.0f})")

    waste_pct = row.get("wastage_history_pct", 0)
    if waste_pct > 0.5:
        reasons.append(f"very high historical waste ({waste_pct*100:.0f}%)")

    return "; ".join(reasons) if reasons else "statistical outlier in feature space"
