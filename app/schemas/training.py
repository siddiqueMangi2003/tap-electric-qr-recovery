from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TrainingRunRequest(BaseModel):
    dry_run: bool = True
    epochs: int = Field(default=3, ge=1, le=20)


class TrainingRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    dry_run: bool
    dataset_version: str | None
    model_version: str | None
    error_message: str | None
    created_at: datetime


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_name: str
    version: str
    dataset_version: str | None
    created_at: datetime
    exact_match_accuracy: float
    character_error_rate: float
    qr_recovery_rate: float
    degraded_scan_accuracy: float
    clean_scan_accuracy: float
    model_path: str
    promoted: bool
