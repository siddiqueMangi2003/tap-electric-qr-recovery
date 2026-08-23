from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.db.models import Scan


@dataclass(frozen=True)
class DatasetExample:
    scan_id: str
    image_uri: str
    correct_qr_payload: str
    charger_id: str
    group_id: str
    split: str
    degradation_type: str


@dataclass(frozen=True)
class DatasetManifest:
    version: str
    examples: list[DatasetExample]
    grouping_strategy: str = "charger_id with image-hash duplicate protection"

    @property
    def split_counts(self) -> dict[str, int]:
        counts = {"train": 0, "validation": 0, "test": 0}
        for example in self.examples:
            counts[example.split] += 1
        return counts


class DatasetBuilder:
    """Build immutable manifests from verified labels using group-safe splits."""

    def build(self, scans: list[Scan]) -> DatasetManifest:
        sorted_scans = sorted(scans, key=lambda scan: (scan.image_sha256, scan.id))
        digest_input = "|".join(
            (
                f"{scan.id}:{scan.image_sha256}:{scan.label.correct_qr_payload}:"
                f"{scan.label.charger_id}"
            )
            for scan in sorted_scans
            if scan.label
        )
        digest = hashlib.sha256(digest_input.encode()).hexdigest()[:12]
        version = f"dataset-{digest}"
        examples: list[DatasetExample] = []
        split_by_image_hash: dict[str, str] = {}
        for scan in sorted_scans:
            if scan.label is None or not scan.label.training_eligible:
                continue
            group_id = scan.label.charger_id
            split = split_by_image_hash.setdefault(
                scan.image_sha256, self._split_for_group(group_id)
            )
            examples.append(
                DatasetExample(
                    scan_id=scan.id,
                    image_uri=scan.image_uri,
                    correct_qr_payload=scan.label.correct_qr_payload,
                    charger_id=scan.label.charger_id,
                    group_id=group_id,
                    split=split,
                    degradation_type=self._degradation_type(scan),
                )
            )
        return DatasetManifest(version=version, examples=examples)

    def save(self, manifest: DatasetManifest, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{manifest.version}.json"
        payload = {
            "version": manifest.version,
            "created_at": datetime.now(UTC).isoformat(),
            "grouping_strategy": manifest.grouping_strategy,
            "split_counts": manifest.split_counts,
            "examples": [asdict(example) for example in manifest.examples],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _split_for_group(group_id: str) -> str:
        bucket = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 100
        if bucket < 70:
            return "train"
        if bucket < 85:
            return "validation"
        return "test"

    @staticmethod
    def _degradation_type(scan: Scan) -> str:
        if scan.brightness_category in {"too_dark", "too_bright"}:
            return scan.brightness_category
        if scan.blur_score is not None and scan.blur_score < 100:
            return "blurry"
        return "clean_or_unknown"
