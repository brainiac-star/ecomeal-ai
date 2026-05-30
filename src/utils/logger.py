import sys
import os
from loguru import logger

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
APP_ENV = os.getenv("APP_ENV", "development")

logger.remove()

logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

logger.add(
    "logs/ecomeal_{time:YYYY-MM-DD}.log",
    level=LOG_LEVEL,
    rotation="00:00",
    retention="7 days",
    compression="gz",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)

__all__ = ["logger"]
