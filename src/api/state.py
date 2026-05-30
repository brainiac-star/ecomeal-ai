"""
Global model registry for the FastAPI app.
Models are loaded once at startup and shared across requests.
"""

from typing import Dict, Any, Optional

_models: Dict[str, Any] = {}


def set_models(models: Dict[str, Any]) -> None:
    global _models
    _models = models


def get_models() -> Dict[str, Any]:
    return _models
