import hashlib
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.db.models import Scan
from app.repositories.chargers import ChargerRepository
from app.repositories.scans import ScanRepository
from app.schemas.scans import (
    ScanAccepted,
    ScanConfirmationRequest,
    ScanCreate,
    ScanLabelResponse,
    ScanResponse,
)
from app.services.charger_resolver import ChargerResolver
from app.services.inference_pipeline import InvalidImageError, decode_image

router = APIRouter(prefix="/scans", tags=["scans"])


@router.post("", response_model=ScanAccepted, status_code=202)
async def create_scan(
    request: Request,
    background_tasks: BackgroundTasks,
    image: Annotated[UploadFile, File(description="JPEG, PNG, or WebP charger sticker image")],
    latitude: Annotated[float, Form()],
    longitude: Annotated[float, Form()],
    timestamp: Annotated[datetime, Form()],
    db: Annotated[Session, Depends(get_db)],
    gps_accuracy_meters: Annotated[float | None, Form()] = None,
    client_scan_id: Annotated[str | None, Form()] = None,
    capture_session_id: Annotated[str | None, Form()] = None,
    device_model: Annotated[str | None, Form()] = None,
    app_version: Annotated[str | None, Form()] = None,
    native_qr_result: Annotated[str | None, Form()] = None,
    native_qr_success: Annotated[bool, Form()] = False,
    native_failure_reason: Annotated[str | None, Form()] = None,
) -> ScanAccepted:
    settings = request.app.state.settings
    content_type = image.content_type or "application/octet-stream"
    if content_type not in settings.accepted_image_types:
        raise HTTPException(status_code=415, detail="Unsupported image media type")

    data = await image.read(settings.max_upload_bytes + 1)
    if not data:
        raise HTTPException(status_code=422, detail="Image must not be empty")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds configured upload limit")
    try:
        decode_image(data)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        payload = ScanCreate(
            latitude=latitude,
            longitude=longitude,
            timestamp=timestamp,
            gps_accuracy_meters=gps_accuracy_meters,
            client_scan_id=client_scan_id,
            capture_session_id=capture_session_id,
            device_model=device_model,
            app_version=app_version,
            native_qr_result=native_qr_result,
            native_qr_success=native_qr_success,
            native_failure_reason=native_failure_reason,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    scans = ScanRepository(db)
    if payload.client_scan_id:
        existing = scans.get_by_client_id(payload.client_scan_id)
        if existing:
            return ScanAccepted(
                scan_id=existing.id, status=existing.status, image_uri=existing.image_uri
            )

    scan_id = str(uuid.uuid4())
    storage = request.app.state.object_storage
    image_uri = storage.upload_image(scan_id, data, content_type)
    scan = Scan(
        id=scan_id,
        client_scan_id=payload.client_scan_id,
        capture_session_id=payload.capture_session_id,
        image_uri=image_uri,
        image_sha256=hashlib.sha256(data).hexdigest(),
        content_type=content_type,
        latitude=payload.latitude,
        longitude=payload.longitude,
        gps_accuracy_meters=payload.gps_accuracy_meters,
        captured_at=payload.timestamp,
        device_model=payload.device_model,
        app_version=payload.app_version,
        native_qr_success=payload.native_qr_success,
        native_qr_result=payload.native_qr_result,
        native_failure_reason=payload.native_failure_reason,
    )
    try:
        scans.add(scan)
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        storage.delete_image(image_uri)
        raise HTTPException(status_code=409, detail="Duplicate client_scan_id") from exc
    except Exception:
        db.rollback()
        storage.delete_image(image_uri)
        raise

    background_tasks.add_task(request.app.state.scan_processor.process, scan.id)
    return ScanAccepted(scan_id=scan.id, status=scan.status, image_uri=scan.image_uri)


@router.get("/{scan_id}", response_model=ScanResponse)
def get_scan(
    scan_id: str,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> ScanResponse:
    scan = ScanRepository(db).get(scan_id, with_label=True)
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    response = ScanResponse.model_validate(scan)
    if scan.resolved_charger_id is None and scan.final_prediction:
        resolver = ChargerResolver(
            ChargerRepository(db),
            radius_meters=request.app.state.settings.inference_radius_meters,
        )
        response.candidates = [
            candidate
            for candidate in resolver.rank_candidates(
                scan.final_prediction,
                latitude=scan.latitude,
                longitude=scan.longitude,
            )
        ]
    return response


@router.post("/{scan_id}/confirm", response_model=ScanLabelResponse, status_code=201)
def confirm_scan(
    scan_id: str,
    confirmation: ScanConfirmationRequest,
    db: Annotated[Session, Depends(get_db)],
) -> object:
    scans = ScanRepository(db)
    scan = scans.get(scan_id, with_label=True)
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    if scan.label is not None:
        raise HTTPException(status_code=409, detail="Scan already has a confirmation label")
    label = scans.add_label(scan, **confirmation.model_dump())
    db.commit()
    db.refresh(label)
    return label
