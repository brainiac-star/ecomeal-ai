"""
Demand forecasting using Prophet (primary) with a statsmodels SARIMA fallback.
Generates forward-looking consumption predictions per ingredient.
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import timedelta

from src.utils.logger import logger
from src.utils.config import get_settings

warnings.filterwarnings("ignore")
settings = get_settings()


class DemandForecaster:
    """
    Per-ingredient demand forecaster backed by Facebook Prophet.
    Falls back to simple linear trend when Prophet fails or data is sparse.
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.forecasts_cache: Dict[str, pd.DataFrame] = {}
        self.is_fitted: bool = False
        self.model_dir = Path(settings.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, demand_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit a Prophet model per ingredient.
        demand_df columns: date, ingredient_name, quantity_used
        """
        try:
            from prophet import Prophet
            prophet_available = True
        except ImportError:
            logger.warning("Prophet not installed, falling back to linear trend")
            prophet_available = False

        if "date" not in demand_df.columns or "quantity_used" not in demand_df.columns:
            raise ValueError("demand_df must have 'date' and 'quantity_used' columns")

        demand_df = demand_df.copy()
        demand_df["date"] = pd.to_datetime(demand_df["date"], errors="coerce")
        demand_df = demand_df.dropna(subset=["date", "quantity_used"])
        demand_df["quantity_used"] = pd.to_numeric(demand_df["quantity_used"], errors="coerce").fillna(0)

        ingredients = demand_df["ingredient_name"].unique()
        logger.info(f"Fitting demand forecasters for {len(ingredients)} ingredients")

        fit_results = {}
        for ingredient in ingredients:
            sub = (
                demand_df[demand_df["ingredient_name"] == ingredient]
                .sort_values("date")
                .drop_duplicates("date")
            )
            if len(sub) < 7:
                logger.debug(f"Skipping {ingredient}: only {len(sub)} data points")
                continue

            try:
                if prophet_available:
                    model = self._fit_prophet(sub)
                    self.models[ingredient] = {"type": "prophet", "model": model}
                else:
                    model = self._fit_linear(sub)
                    self.models[ingredient] = {"type": "linear", "model": model}
                fit_results[ingredient] = "ok"
            except Exception as e:
                logger.warning(f"Forecast fit failed for {ingredient}: {e}")
                fit_results[ingredient] = f"failed: {e}"

        self.is_fitted = True
        logger.info(f"Forecasters fitted: {sum(v == 'ok' for v in fit_results.values())}/{len(ingredients)}")
        return fit_results

    def _fit_prophet(self, sub: pd.DataFrame):
        from prophet import Prophet
        prophet_df = sub[["date", "quantity_used"]].rename(
            columns={"date": "ds", "quantity_used": "y"}
        )
        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=len(sub) > 90,
            changepoint_prior_scale=0.05,
            interval_width=0.80,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(prophet_df)
        return m

    def _fit_linear(self, sub: pd.DataFrame) -> Dict:
        """Simple linear trend fallback."""
        y = sub["quantity_used"].values
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 1)
        return {
            "slope": coeffs[0],
            "intercept": coeffs[1],
            "last_date": sub["date"].max(),
            "mean": y.mean(),
            "std": y.std(),
        }

    def forecast(self, ingredient: str, horizon_days: int = 14) -> pd.DataFrame:
        """
        Return forecast DataFrame with columns:
        ds, yhat, yhat_lower, yhat_upper, demand_level
        """
        if ingredient not in self.models:
            logger.warning(f"No model for {ingredient}, returning naive forecast")
            return self._naive_forecast(horizon_days)

        entry = self.models[ingredient]
        try:
            if entry["type"] == "prophet":
                future = entry["model"].make_future_dataframe(periods=horizon_days)
                fc = entry["model"].predict(future).tail(horizon_days)
                fc = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                fc["yhat"] = fc["yhat"].clip(lower=0)
                fc["yhat_lower"] = fc["yhat_lower"].clip(lower=0)
                fc["yhat_upper"] = fc["yhat_upper"].clip(lower=0)
            else:
                m = entry["model"]
                last_n = len(
                    [k for k in self.models if self.models[k]["type"] == "linear"]
                )
                rows = []
                for i in range(1, horizon_days + 1):
                    ds = m["last_date"] + timedelta(days=i)
                    yhat = max(0, m["slope"] * (last_n + i) + m["intercept"])
                    std = max(m["std"], yhat * 0.1)
                    rows.append({"ds": ds, "yhat": yhat,
                                 "yhat_lower": max(0, yhat - std),
                                 "yhat_upper": yhat + std})
                fc = pd.DataFrame(rows)
        except Exception as e:
            logger.warning(f"Forecast failed for {ingredient}: {e}")
            return self._naive_forecast(horizon_days)

        fc["demand_level"] = fc["yhat"].apply(_demand_level)
        return fc.reset_index(drop=True)

    def _naive_forecast(self, horizon_days: int) -> pd.DataFrame:
        today = pd.Timestamp.today()
        rows = []
        for i in range(1, horizon_days + 1):
            rows.append({
                "ds": today + timedelta(days=i),
                "yhat": 10.0,
                "yhat_lower": 5.0,
                "yhat_upper": 15.0,
                "demand_level": "medium",
            })
        return pd.DataFrame(rows)

    def detect_overstock(
        self, inventory_df: pd.DataFrame, horizon_days: int = 14
    ) -> pd.DataFrame:
        """
        Compare forecast demand vs current stock and flag overstock/shortage risks.
        """
        results = []
        for _, row in inventory_df.iterrows():
            ingredient = row.get("ingredient_name", "")
            current_qty = float(row.get("quantity", 0))
            daily_cons = float(row.get("daily_consumption", 0.01))
            dte = int(row.get("days_to_expiry", 0))

            if ingredient in self.models:
                fc = self.forecast(ingredient, horizon_days)
                forecast_demand = fc["yhat"].sum()
            else:
                forecast_demand = daily_cons * horizon_days

            expected_usage = min(forecast_demand, daily_cons * dte)
            excess = current_qty - expected_usage
            shortage = max(0, expected_usage - current_qty)

            results.append({
                "item_id": row.get("item_id", ""),
                "ingredient_name": ingredient,
                "current_quantity": current_qty,
                "forecast_demand": round(forecast_demand, 3),
                "expected_usage_before_expiry": round(expected_usage, 3),
                "overstock_quantity": round(max(0, excess), 3),
                "shortage_quantity": round(shortage, 3),
                "overstock_risk": excess > current_qty * 0.3,
                "shortage_risk": shortage > 0,
            })

        return pd.DataFrame(results)

    def save(self) -> None:
        joblib.dump(self.models, self.model_dir / "demand_models.pkl")
        logger.info(f"Demand forecasters saved ({len(self.models)} models)")

    def load(self) -> bool:
        try:
            self.models = joblib.load(self.model_dir / "demand_models.pkl")
            self.is_fitted = bool(self.models)
            logger.info(f"Demand forecasters loaded ({len(self.models)} models)")
            return True
        except FileNotFoundError:
            logger.warning("No saved demand models found")
            return False
        except Exception as e:
            logger.error(f"Error loading demand models: {e}")
            return False


def _demand_level(yhat: float) -> str:
    if yhat <= 5:
        return "low"
    elif yhat <= 20:
        return "medium"
    elif yhat <= 50:
        return "high"
    return "very_high"
