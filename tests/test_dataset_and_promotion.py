from datetime import UTC, datetime

from app.db.models import Scan, ScanLabel
from app.ml.dataset import DatasetBuilder
from app.ml.model_registry import EvaluationMetrics, ModelPromotionPolicy


def _labeled_scan(scan_id: str, charger_id: str, image_hash: str) -> Scan:
    scan = Scan(
        id=scan_id,
        image_uri=f"local://{scan_id}.png",
        image_sha256=image_hash,
        content_type="image/png",
        latitude=0,
        longitude=0,
        captured_at=datetime.now(UTC),
    )
    scan.label = ScanLabel(
        correct_qr_payload=f"payload-{charger_id}",
        charger_id=charger_id,
        confirmation_source="operator",
        review_status="verified",
        training_eligible=True,
    )
    return scan


def test_dataset_keeps_same_charger_in_one_split() -> None:
    manifest = DatasetBuilder().build(
        [
            _labeled_scan("1", "charger-a", "a" * 64),
            _labeled_scan("2", "charger-a", "b" * 64),
            _labeled_scan("3", "charger-b", "c" * 64),
        ]
    )
    charger_a_splits = {
        example.split for example in manifest.examples if example.charger_id == "charger-a"
    }
    assert len(charger_a_splits) == 1


def test_model_promotion_requires_gain_without_clean_regression() -> None:
    incumbent = EvaluationMetrics(0.70, 0.10, 0.60, 0.60, 0.95)
    good_candidate = EvaluationMetrics(0.75, 0.08, 0.68, 0.66, 0.945)
    bad_candidate = EvaluationMetrics(0.76, 0.08, 0.69, 0.68, 0.90)
    policy = ModelPromotionPolicy()
    assert policy.decide(good_candidate, incumbent).promote is True
    assert policy.decide(bad_candidate, incumbent).promote is False
