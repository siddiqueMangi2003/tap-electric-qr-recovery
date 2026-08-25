from pydantic import BaseModel, ConfigDict, Field


class ChargerCandidateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    charger_id: str
    qr_payload: str
    distance_meters: float
    text_similarity: float
    match_score: float


class InferenceResponse(BaseModel):
    qr_decoder_result: str | None
    ml_prediction: str | None
    final_prediction: str | None
    resolved_charger_id: str | None
    prediction_source: str | None
    confidence: float
    preprocessing_strategy: str
    blur_score: float
    brightness_score: float
    brightness_category: str
    explanation: str
    candidates: list[ChargerCandidateResponse] = Field(default_factory=list)
