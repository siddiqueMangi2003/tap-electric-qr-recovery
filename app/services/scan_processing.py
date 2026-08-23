import logging

from sqlalchemy.orm import Session, sessionmaker

from app.core.logging import log_event
from app.db.models import ScanStatus
from app.repositories.chargers import ChargerRepository
from app.repositories.scans import ScanRepository
from app.services.inference_pipeline import InferencePipeline
from app.services.storage.base import ObjectStorage

logger = logging.getLogger(__name__)


class ScanProcessor:
    """Worker boundary: owns a fresh DB session for every inference job."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        storage: ObjectStorage,
        pipeline: InferencePipeline,
    ) -> None:
        self.session_factory = session_factory
        self.storage = storage
        self.pipeline = pipeline

    def process(self, scan_id: str) -> None:
        session = self.session_factory()
        try:
            scans = ScanRepository(session)
            scan = scans.get(scan_id)
            if scan is None:
                log_event(logger, "scan_not_found", scan_id=scan_id)
                return
            scan.status = ScanStatus.PROCESSING.value
            scan.error_message = None
            session.commit()

            image_bytes = self.storage.get_image(scan.image_uri)
            result = self.pipeline.run(
                image_bytes,
                latitude=scan.latitude,
                longitude=scan.longitude,
                charger_repository=ChargerRepository(session),
                native_qr_success=scan.native_qr_success,
                native_qr_result=scan.native_qr_result,
            )
            scan.blur_score = result.blur_score
            scan.brightness_score = result.brightness_score
            scan.brightness_category = result.brightness_category
            scan.preprocessing_strategy = result.preprocessing_strategy
            scan.qr_decoder_result = result.qr_decoder_result
            scan.ml_prediction = result.ml_prediction
            scan.final_prediction = result.final_prediction
            scan.resolved_charger_id = result.resolved_charger_id
            scan.prediction_source = result.prediction_source
            scan.confidence = result.confidence
            scan.resolution_explanation = result.resolution_explanation
            scan.status = ScanStatus.COMPLETED.value
            session.commit()
            log_event(logger, "scan_processed", scan_id=scan_id, source=result.prediction_source)
        except Exception as exc:
            session.rollback()
            scan = ScanRepository(session).get(scan_id)
            if scan:
                scan.status = ScanStatus.FAILED.value
                scan.error_message = str(exc)[:1000]
                session.commit()
            logger.exception("scan_processing_failed scan_id=%s", scan_id)
        finally:
            session.close()
