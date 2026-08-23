import hashlib
import json
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models import Scan, ScanLabel
from app.db.session import build_session_factory
from app.ml.hybrid_importer import HybridDatasetImporter
from app.ml.synthetic_dataset import SyntheticDatasetConfig, SyntheticDatasetGenerator
from app.services.storage.local import LocalObjectStorage


def _generate_small_dataset(tmp_path: Path) -> Path:
    return SyntheticDatasetGenerator(
        SyntheticDatasetConfig(
            chargers=3,
            variants_per_charger=2,
            seed=17,
            width=480,
            height=360,
        )
    ).generate(tmp_path / "dataset")


def test_synthetic_generator_creates_exact_grouped_manifest(tmp_path: Path) -> None:
    manifest_path = _generate_small_dataset(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = manifest["examples"]
    assert len(examples) == 6
    assert {example["split"] for example in examples} == {
        "train",
        "validation",
        "test",
    }

    split_by_sticker: dict[str, set[str]] = {}
    for example in examples:
        split_by_sticker.setdefault(example["sticker_id"], set()).add(example["split"])
        image_path = manifest_path.parent / example["image_path"]
        assert image_path.is_file()
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == example["image_sha256"]
        assert example["charger_id"] in example["qr_payload"]
    assert all(len(splits) == 1 for splits in split_by_sticker.values())


def test_hybrid_importer_is_verified_and_idempotent(tmp_path: Path) -> None:
    manifest_path = _generate_small_dataset(tmp_path)
    engine = create_engine(f"sqlite:///{(tmp_path / 'import.db').as_posix()}")
    Base.metadata.create_all(engine)
    importer = HybridDatasetImporter(
        build_session_factory(engine), LocalObjectStorage(tmp_path / "objects")
    )

    first = importer.import_manifest(manifest_path)
    second = importer.import_manifest(manifest_path)
    assert first.imported == first.training_eligible == 6
    assert first.pending_review == 0
    assert second.skipped == 6

    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Scan)) == 6
        assert session.scalar(select(func.count()).select_from(ScanLabel)) == 6
        assert set(session.scalars(select(Scan.dataset_split))) == {
            "train",
            "validation",
            "test",
        }
        assert all(session.scalars(select(ScanLabel.training_eligible)))
    engine.dispose()
