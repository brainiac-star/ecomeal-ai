"""
Input validation layer for API requests.
All external inputs must pass through here before touching ML models.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List
from datetime import date
import re


class InventoryItem(BaseModel):
    ingredient_name: str = Field(..., min_length=1, max_length=200)
    category: str = Field(..., min_length=1, max_length=100)
    quantity: float = Field(..., ge=0, le=1_000_000)
    unit: str = Field("kg", max_length=20)
    purchase_date: Optional[date] = None
    expiry_date: Optional[date] = None
    daily_consumption: float = Field(..., gt=0, le=100_000)
    storage_type: str = Field("ambient", max_length=30)
    supplier: Optional[str] = Field(None, max_length=200)
    wastage_history_pct: float = Field(0.1, ge=0.0, le=1.0)
    price_per_unit: float = Field(50.0, ge=0)
    reorder_point: float = Field(1.0, ge=0)

    @field_validator("ingredient_name", "category", "supplier", mode="before")
    @classmethod
    def strip_and_sanitize(cls, v):
        if v is None:
            return v
        cleaned = re.sub(r"[<>\"';&]", "", str(v)).strip()
        return cleaned

    @field_validator("storage_type", mode="before")
    @classmethod
    def validate_storage(cls, v):
        allowed = {"frozen", "refrigerated", "ambient"}
        v = str(v).strip().lower()
        return v if v in allowed else "ambient"

    @model_validator(mode="after")
    def validate_dates(self):
        today = date.today()
        if self.purchase_date and self.purchase_date > today:
            self.purchase_date = today
        if self.expiry_date and self.expiry_date < today:
            pass  # Expired items are valid inputs
        if self.purchase_date and self.expiry_date:
            if self.purchase_date > self.expiry_date:
                self.expiry_date = self.purchase_date
        return self


class BatchInventoryRequest(BaseModel):
    items: List[InventoryItem] = Field(..., min_length=1, max_length=10_000)
    restaurant_name: Optional[str] = Field(None, max_length=200)


class ChefSpecialsRequest(BaseModel):
    ingredients: List[str] = Field(..., min_length=1, max_length=20)
    cuisine_preference: Optional[str] = Field(None, max_length=100)
    dietary_restrictions: Optional[List[str]] = None
    n_suggestions: int = Field(3, ge=1, le=10)

    @field_validator("ingredients", mode="before")
    @classmethod
    def sanitize_ingredients(cls, items):
        return [re.sub(r"[<>\"';&]", "", str(i)).strip() for i in items if str(i).strip()]


class DemandForecastRequest(BaseModel):
    ingredient_name: str = Field(..., min_length=1, max_length=200)
    horizon_days: int = Field(14, ge=1, le=90)
