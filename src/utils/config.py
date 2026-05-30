import os
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from functools import lru_cache


def _get_secret(key: str, default: str = "") -> str:
    """Read from env vars first, then Streamlit secrets if available."""
    val = os.environ.get(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # App
    app_env: str = Field("development", validation_alias="APP_ENV")
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")
    debug: bool = Field(False, validation_alias="DEBUG")
    random_seed: int = Field(42, validation_alias="RANDOM_SEED")
    dataset_size: int = Field(1200, validation_alias="DATASET_SIZE")

    # API
    api_host: str = Field("0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(8000, validation_alias="API_PORT")
    api_workers: int = Field(2, validation_alias="API_WORKERS")

    # LLM providers
    anthropic_api_key: str = Field("", validation_alias="ANTHROPIC_API_KEY")
    groq_api_key: str = Field("", validation_alias="GROQ_API_KEY")
    gemini_api_key: str = Field("", validation_alias="GEMINI_API_KEY")

    # Model names
    claude_model: str = Field("claude-sonnet-4-6", validation_alias="CLAUDE_MODEL")
    groq_model: str = Field("llama-3.1-8b-instant", validation_alias="GROQ_MODEL")
    gemini_model: str = Field("gemini-1.5-flash", validation_alias="GEMINI_MODEL")
    ollama_model: str = Field("llama3", validation_alias="OLLAMA_MODEL")

    # Paths
    model_dir: str = Field("data/models", validation_alias="MODEL_DIR")
    data_dir: str = Field("data/processed", validation_alias="DATA_DIR")
    raw_data_dir: str = Field("data/raw", validation_alias="RAW_DATA_DIR")

    # Rate limiting
    rate_limit_per_minute: int = Field(60, validation_alias="RATE_LIMIT_PER_MINUTE")


@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    # Override with Streamlit secrets if keys are empty
    if not s.groq_api_key:
        s.groq_api_key = _get_secret("GROQ_API_KEY")
    if not s.anthropic_api_key:
        s.anthropic_api_key = _get_secret("ANTHROPIC_API_KEY")
    if not s.gemini_api_key:
        s.gemini_api_key = _get_secret("GEMINI_API_KEY")
    return s
