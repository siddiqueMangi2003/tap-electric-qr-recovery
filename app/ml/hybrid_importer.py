from __future__ import annotations

import hashlib
import json
import mimetypes
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Charger, Scan, ScanStatus
from app.repositories.chargers import ChargerRepository
from app.repositories.scans import ScanRepository
from app.services.image_quality import ImageQualityService
from app.services.inference_pipeline import decode_image
from app.services.storage.base import ObjectStorage


@dataclass(frozen=True)
class ImportSummary:
    imported: int
    skipped: int
    training_eligible: int
    pending_review: int


class HybridDatasetImporter:
    """Normalize synthetic, public, and physical manifests into the service stores."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        storage: ObjectStorage,
    ) -> None:
        self.session_factory = session_factory
        self.storage = storage
        self.quality_service = ImageQualityService()

    def import_manifest(self, manifest_path: Path) -> ImportSummary:
        manifest_path = manifest_path.resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        examples = payload.get("examples")
        if not isinstance(examples, list):
            raise ValueError("Manifest must contain an examples list")

        imported = skipped = eligible = pending = 0
        session = self.session_factory()
        try:
            for example in examples:
                client_scan_id = self._client_scan_id(example)
                scans = ScanRepository(session)
                if scans.get_by_client_id(client_scan_id):
                    skipped += 1
                    continue

                image_path = self._resolve_image(manifest_path.parent, example["image_path"])
                image_bytes = image_path.read_bytes()
                expected_hash = example.get("image_sha256")
                actual_hash = hashlib.sha256(image_bytes).hexdigest()
                if expected_hash and expected_hash != actual_hash:
                    raise ValueError(f"Checksum mismatch for {image_path.name}")
                image = decode_image(image_bytes)
                quality = self.quality_service.analyze(image)
                content_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
                scan_id = str(uuid.uuid4())
                image_uri = self.storage.upload_image(scan_id, image_bytes, content_type)
                try:
                    charger = ChargerRepository(session).get(example["charger_id"])
                    if charger is None:
                        charger = Charger(
                            charger_id=example["charger_id"],
                            qr_payload=example["qr_payload"],
                            latitude=float(example["latitude"]),
                            longitude=float(example["longitude"]),
                        )
                        session.add(charger)
                    elif charger.qr_payload != example["qr_payload"]:
                        raise ValueError(
                            f"Charger {charger.charger_id} has conflicting payload labels"
                        )

                    native_success = bool(example.get("native_qr_success", False))
                    scan = Scan(
                        id=scan_id,
                        client_scan_id=client_scan_id,
                        capture_session_id=example.get("session_id"),
                        sticker_id=example.get("sticker_id"),
                        dataset_source=example.get("source", "unknown")[:50],
                        dataset_split=example.get("split"),
                        image_uri=image_uri,
                        image_sha256=actual_hash,
                        content_type=content_type,
                        latitude=float(example["latitude"]),
                        longitude=float(example["longitude"]),
                        captured_at=self._captured_at(example),
                        native_qr_success=native_success,
                        native_qr_result=(example["qr_payload"] if native_success else None),
                        blur_score=quality.blur_score,
                        brightness_score=quality.brightness_score,
                        brightness_category=quality.brightness_category,
                        preprocessing_strategy="dataset_import",
                        status=ScanStatus.COMPLETED.value,
                    )
                    scans.add(scan)
                    verified = bool(example.get("verified", False))
                    scans.add_label(
                        scan,
                        correct_qr_payload=example["qr_payload"],
                        charger_id=example["charger_id"],
                        confirmation_source=f"dataset:{scan.dataset_source}"[:50],
                        confirmed_by="manifest-importer",
                        verified=verified,
                    )
                    session.commit()
                    imported += 1
                    if verified:
                        eligible += 1
                    else:
                        pending += 1
                except Exception:
                    session.rollback()
                    self.storage.delete_image(image_uri)
                    raise
        finally:
            session.close()
        return ImportSummary(imported, skipped, eligible, pending)

    @staticmethod
    def _resolve_image(manifest_root: Path, relative_path: str) -> Path:
        candidate = (manifest_root / relative_path).resolve()
        if not candidate.is_relative_to(manifest_root.resolve()):
            raise ValueError("Image path escapes the manifest directory")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    @staticmethod
    def _client_scan_id(example: dict[str, object]) -> str:
        source = str(example.get("source", "unknown"))
        example_id = str(example["example_id"])
        readable = f"dataset:{source}:{example_id}"
        if len(readable) <= 100:
            return readable
        digest = hashlib.sha256(readable.encode()).hexdigest()
        return f"dataset:{source[:20]}:{digest}"[:100]

    @staticmethod
    def _captured_at(example: dict[str, object]) -> datetime:
        value = example.get("captured_at")
        if not value:
            return datetime.now(UTC)
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
