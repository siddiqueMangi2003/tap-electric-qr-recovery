from pydantic import BaseModel


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
