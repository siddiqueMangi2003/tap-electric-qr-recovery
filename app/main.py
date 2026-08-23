from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import inference, models, scans, training
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.base import Base
from app.db.session import build_engine, build_session_factory
from app.services.image_quality import ImageQualityService
from app.services.inference_pipeline import InferencePipeline
from app.services.ml_recognizer import TrocrRecognizer
from app.services.preprocessing import ImagePreprocessor
from app.services.qr_decoder import QRDecoder
from app.services.scan_processing import ScanProcessor
from app.services.storage import LocalObjectStorage, S3ObjectStorage
from app.services.training_service import TrainingService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging()
    engine = build_engine(settings.database_url)
    session_factory = build_session_factory(engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        Base.metadata.create_all(engine)
        yield
        engine.dispose()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Collect, recover, confirm, and learn from degraded EV charger stickers.",
        lifespan=lifespan,
    )
    storage = (
        LocalObjectStorage(settings.local_storage_path)
        if settings.object_storage_backend == "local"
        else S3ObjectStorage(settings.s3_bucket, settings.s3_endpoint_url)
    )
    pipeline = InferencePipeline(
        image_quality=ImageQualityService(),
        preprocessor=ImagePreprocessor(),
        qr_decoder=QRDecoder(),
        recognizer=TrocrRecognizer(
            settings.trocr_model_name,
            enabled=settings.enable_trocr,
            local_files_only=settings.trocr_local_files_only,
        ),
        resolver_radius_meters=settings.inference_radius_meters,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.object_storage = storage
    app.state.inference_pipeline = pipeline
    app.state.scan_processor = ScanProcessor(session_factory, storage, pipeline)
    app.state.training_service = TrainingService(session_factory, storage, settings)

    app.include_router(scans.router, prefix=settings.api_prefix)
    app.include_router(inference.router, prefix=settings.api_prefix)
    app.include_router(training.router, prefix=settings.api_prefix)
    app.include_router(models.router, prefix=settings.api_prefix)

    @app.get("/health", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
