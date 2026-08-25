from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.inference import ChargerCandidateResponse


class ScanCreate(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timestamp: datetime
    gps_accuracy_meters: float | None = Field(default=None, ge=0)
    client_scan_id: str | None = Field(default=None, min_length=1, max_length=100)
    capture_session_id: str | None = Field(default=None, min_length=1, max_length=100)
    device_model: str | None = Field(default=None, max_length=200)
    app_version: str | None = Field(default=None, max_length=50)
    native_qr_result: str | None = Field(default=None, max_length=4096)
    native_qr_success: bool = False
    native_failure_reason: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def result_matches_success(self) -> "ScanCreate":
        if self.native_qr_success and not self.native_qr_result:
            raise ValueError("native_qr_result is required when native_qr_success is true")
        return self


class ScanAccepted(BaseModel):
    scan_id: str
    status: str
    image_uri: str


class ScanLabelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    correct_qr_payload: str
    charger_id: str
    confirmation_source: str
    review_status: str
    training_eligible: bool
    confirmed_at: datetime


class ScanResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    client_scan_id: str | None
    image_uri: str
    latitude: float
    longitude: float
    gps_accuracy_meters: float | None
    captured_at: datetime
    device_model: str | None
    app_version: str | None
    native_qr_success: bool
    native_qr_result: str | None
    blur_score: float | None
    brightness_score: float | None
    brightness_category: str | None
    preprocessing_strategy: str | None
    qr_decoder_result: str | None
    ml_prediction: str | None
    final_prediction: str | None
    resolved_charger_id: str | None
    prediction_source: str | None
    confidence: float | None
    resolution_explanation: str | None
    status: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    label: ScanLabelResponse | None = None
    candidates: list[ChargerCandidateResponse] = Field(default_factory=list)


class ScanConfirmationRequest(BaseModel):
    correct_qr_payload: str = Field(min_length=1, max_length=4096)
    charger_id: str = Field(min_length=1, max_length=100)
    confirmation_source: str = Field(default="operator", min_length=1, max_length=50)
    confirmed_by: str | None = Field(default=None, max_length=100)
    verified: bool = False
