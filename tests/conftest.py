from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.models import Charger
from app.main import create_app


@pytest.fixture
def app_client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(
        app_env="test",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        local_storage_path=tmp_path / "images",
        model_artifact_path=tmp_path / "models",
        enable_trocr=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        session = app.state.session_factory()
        session.add(
            Charger(
                charger_id="NL-TAP-E12345",
                qr_payload="https://tap-electric.com/c/NL-TAP-E12345",
                latitude=52.3676,
                longitude=4.9041,
            )
        )
        session.commit()
        session.close()
        yield client


@pytest.fixture
def png_bytes() -> bytes:
    image = np.full((240, 320, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()
