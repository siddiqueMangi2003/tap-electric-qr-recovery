import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.models import (
    DatasetVersion,
    ModelVersion,
    TrainingRun,
    TrainingRunStatus,
)
from app.ml.dataset import DatasetBuilder, DatasetManifest
from app.ml.evaluate import EvaluationReport, evaluate_predictions
from app.ml.model_registry import EvaluationMetrics, ModelPromotionPolicy
from app.ml.train import TrocrFineTuner
from app.repositories.chargers import ChargerRepository
from app.repositories.model_versions import ModelVersionRepository
from app.repositories.scans import ScanRepository
from app.services.image_quality import ImageQualityService
from app.services.inference_pipeline import InferencePipeline
from app.services.ml_recognizer import TrocrRecognizer
from app.services.preprocessing import ImagePreprocessor
from app.services.qr_decoder import QRDecoder
from app.services.storage.base import ObjectStorage

logger = logging.getLogger(__name__)


class TrainingService:
    """Offline job boundary for dataset creation, fine-tuning, and promotion."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        storage: ObjectStorage,
        settings: Settings,
    ) -> None:
        self.session_factory = session_factory
        self.storage = storage
        self.settings = settings

    def run(self, training_run_id: str, *, epochs: int) -> None:
        session = self.session_factory()
        try:
            training_run = session.get(TrainingRun, training_run_id)
            if training_run is None:
                return
            training_run.status = TrainingRunStatus.RUNNING.value
            training_run.started_at = datetime.now(UTC)
            session.commit()

            scans = ScanRepository(session).list_training_eligible()
            builder = DatasetBuilder()
            manifest = builder.build(scans)
            manifest_path = builder.save(manifest, self.settings.model_artifact_path / "datasets")
            dataset = session.scalar(
                select(DatasetVersion).where(DatasetVersion.version == manifest.version)
            )
            if dataset is None:
                dataset = DatasetVersion(
                    version=manifest.version,
                    manifest_path=str(manifest_path),
                    example_count=len(manifest.examples),
                    grouping_strategy=manifest.grouping_strategy,
                )
                session.add(dataset)
            training_run.dataset_version = manifest.version

            if training_run.dry_run:
                training_run.status = TrainingRunStatus.COMPLETED.value
                training_run.completed_at = datetime.now(UTC)
                session.commit()
                return

            if len(manifest.examples) < 3:
                raise ValueError("Real training requires at least three verified examples")

            model_version_name = f"trocr-{uuid.uuid4().hex[:12]}"
            output_dir = self.settings.model_artifact_path / model_version_name
            model_path, training_metrics = TrocrFineTuner(
                self.settings.trocr_model_name, self.storage
            ).train(manifest, output_dir, epochs=epochs)
            evaluation = self._evaluate_end_to_end(manifest, model_path, session)
            metrics = evaluation.metrics
            models = ModelVersionRepository(session)
            incumbent = models.promoted()
            incumbent_metrics = (
                EvaluationMetrics(
                    exact_match_accuracy=incumbent.exact_match_accuracy,
                    character_error_rate=incumbent.character_error_rate,
                    qr_recovery_rate=incumbent.qr_recovery_rate,
                    degraded_scan_accuracy=incumbent.degraded_scan_accuracy,
                    clean_scan_accuracy=incumbent.clean_scan_accuracy,
                )
                if incumbent
                else None
            )
            decision = ModelPromotionPolicy().decide(metrics, incumbent_metrics)
            model = models.add(
                ModelVersion(
                    model_name=self.settings.trocr_model_name,
                    version=model_version_name,
                    dataset_version=manifest.version,
                    exact_match_accuracy=metrics.exact_match_accuracy,
                    character_error_rate=metrics.character_error_rate,
                    qr_recovery_rate=metrics.qr_recovery_rate,
                    degraded_scan_accuracy=metrics.degraded_scan_accuracy,
                    clean_scan_accuracy=metrics.clean_scan_accuracy,
                    metrics_json=json.dumps(
                        {
                            "training_stage": training_metrics,
                            "end_to_end_accuracy_by_degradation": (
                                evaluation.accuracy_by_degradation
                            ),
                            "promotion_reason": decision.reason,
                        },
                        sort_keys=True,
                    ),
                    model_path=str(model_path),
                    promoted=False,
                )
            )
            if decision.promote:
                models.promote(model)
            training_run.model_version = model.version
            training_run.status = TrainingRunStatus.COMPLETED.value
            training_run.completed_at = datetime.now(UTC)
            session.commit()
        except Exception as exc:
            session.rollback()
            training_run = session.get(TrainingRun, training_run_id)
            if training_run:
                training_run.status = TrainingRunStatus.FAILED.value
                training_run.error_message = str(exc)[:1000]
                training_run.completed_at = datetime.now(UTC)
                session.commit()
            logger.exception("training_run_failed training_run_id=%s", training_run_id)
        finally:
            session.close()

    def _evaluate_end_to_end(
        self, manifest: DatasetManifest, model_path: Path, session: Session
    ) -> EvaluationReport:
        evaluation_examples = [
            example for example in manifest.examples if example.split == "test"
        ] or [example for example in manifest.examples if example.split == "validation"]
        if not evaluation_examples:
            raise ValueError(
                "Grouped split produced no validation/test examples; collect more chargers"
            )

        pipeline = InferencePipeline(
            image_quality=ImageQualityService(),
            preprocessor=ImagePreprocessor(),
            qr_decoder=QRDecoder(),
            recognizer=TrocrRecognizer(str(model_path), enabled=True, local_files_only=True),
            resolver_radius_meters=self.settings.inference_radius_meters,
        )
        scans = ScanRepository(session)
        charger_repository = ChargerRepository(session)
        predictions: list[str] = []
        references: list[str] = []
        degradations: list[str] = []
        previously_failed: list[bool] = []
        for example in evaluation_examples:
            scan = scans.get(example.scan_id)
            if scan is None:
                continue
            result = pipeline.run(
                self.storage.get_image(scan.image_uri),
                latitude=scan.latitude,
                longitude=scan.longitude,
                charger_repository=charger_repository,
                native_qr_success=scan.native_qr_success,
                native_qr_result=scan.native_qr_result,
            )
            predictions.append(result.final_prediction or "")
            references.append(example.correct_qr_payload)
            degradations.append(example.degradation_type)
            previously_failed.append(not scan.native_qr_success)
        if not predictions:
            raise ValueError("No readable validation/test examples were available")
        return evaluate_predictions(predictions, references, degradations, previously_failed)
