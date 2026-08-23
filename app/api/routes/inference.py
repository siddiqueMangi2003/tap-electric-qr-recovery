from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.repositories.chargers import ChargerRepository
from app.schemas.inference import InferenceResponse
from app.services.inference_pipeline import InvalidImageError

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("", response_model=InferenceResponse)
async def run_inference(
    request: Request,
    image: Annotated[UploadFile, File()],
    latitude: Annotated[float, Form(ge=-90, le=90)],
    longitude: Annotated[float, Form(ge=-180, le=180)],
    db: Annotated[Session, Depends(get_db)],
) -> InferenceResponse:
    settings = request.app.state.settings
    if image.content_type not in settings.accepted_image_types:
        raise HTTPException(status_code=415, detail="Unsupported image media type")
    data = await image.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds configured upload limit")
    try:
        result = request.app.state.inference_pipeline.run(
            data,
            latitude=latitude,
            longitude=longitude,
            charger_repository=ChargerRepository(db),
        )
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return InferenceResponse(
        qr_decoder_result=result.qr_decoder_result,
        ml_prediction=result.ml_prediction,
        final_prediction=result.final_prediction,
        resolved_charger_id=result.resolved_charger_id,
        prediction_source=result.prediction_source,
        confidence=result.confidence,
        preprocessing_strategy=result.preprocessing_strategy,
        blur_score=result.blur_score,
        brightness_score=result.brightness_score,
        brightness_category=result.brightness_category,
        explanation=result.resolution_explanation,
    )
