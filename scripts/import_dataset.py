import argparse
from pathlib import Path

from app.core.config import get_settings
from app.db.base import Base
from app.db.session import build_engine, build_session_factory
from app.ml.hybrid_importer import HybridDatasetImporter
from app.services.storage import LocalObjectStorage, S3ObjectStorage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a normalized synthetic, public, or physical dataset manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    session_factory = build_session_factory(engine)
    storage = (
        LocalObjectStorage(settings.local_storage_path)
        if settings.object_storage_backend == "local"
        else S3ObjectStorage(settings.s3_bucket, settings.s3_endpoint_url)
    )
    summary = HybridDatasetImporter(session_factory, storage).import_manifest(args.manifest)
    engine.dispose()
    print(
        f"Imported={summary.imported} skipped={summary.skipped} "
        f"eligible={summary.training_eligible} pending_review={summary.pending_review}"
    )


if __name__ == "__main__":
    main()
