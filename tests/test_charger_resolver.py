from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models import Charger
from app.repositories.chargers import ChargerRepository
from app.services.charger_resolver import (
    ChargerResolver,
    haversine_meters,
    text_similarity,
)


def test_haversine_distance_for_known_short_distance() -> None:
    distance = haversine_meters(52.3676, 4.9041, 52.3677, 4.9041)
    assert 10 < distance < 12


def test_text_similarity_normalizes_punctuation_and_case() -> None:
    assert text_similarity("nl-tap-e12345", "NL TAP E12345") == 1.0


def test_resolver_corrects_small_ocr_error_using_nearby_charger() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            Charger(
                charger_id="NL-TAP-E12345",
                qr_payload="NL-TAP-E12345",
                latitude=52.36765,
                longitude=4.9041,
            )
        )
        session.commit()
        result = ChargerResolver(ChargerRepository(session), radius_meters=500).resolve(
            "NL-TAP-E1234S",
            latitude=52.3676,
            longitude=4.9041,
            source_confidence=0.55,
        )
    assert result.charger_id == "NL-TAP-E12345"
    assert result.payload == "NL-TAP-E12345"
    assert "distance" in result.explanation
