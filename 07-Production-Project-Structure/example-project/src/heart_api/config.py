"""
All configuration in one place, read from the environment.

Rule: no magic strings scattered through the codebase, no secrets in source.
The same image runs in dev, staging and prod - only the environment differs.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "heart-risk-api"
    model_path: Path = Path("models/model.joblib")
    log_level: str = "INFO"

    # The threshold is a *business* decision, not a model detail - so it is config.
    # See module 02, notebook 05: 0.5 is a default, not a law.
    decision_threshold: float = 0.35


settings = Settings()
