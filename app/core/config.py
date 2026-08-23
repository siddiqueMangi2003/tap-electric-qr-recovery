from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Tap Electric QR Recovery Service"
    app_env: Literal["development", "test", "production"] = "development"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./data/tap_qr.db"

    object_storage_backend: Literal["local", "s3"] = "local"
    local_storage_path: Path = Path("./data/images")
    s3_bucket: str = "tap-electric-scans"
    s3_endpoint_url: str | None = None

    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    accepted_image_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")
    inference_radius_meters: float = Field(default=500.0, gt=0)

    enable_trocr: bool = False
    trocr_model_name: str = "microsoft/trocr-base-printed"
    trocr_local_files_only: bool = True
    model_artifact_path: Path = Path("./models")

    raw_image_retention_days: int = Field(default=30, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
