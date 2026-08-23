from dataclasses import dataclass
from difflib import SequenceMatcher
from math import asin, cos, degrees, radians, sin, sqrt

from app.db.models import Charger
from app.repositories.chargers import ChargerRepository

EARTH_RADIUS_METERS = 6_371_000.0


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r, lon1_r, lat2_r, lon2_r = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    value = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_METERS * asin(sqrt(value))


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_identifier(left), normalize_identifier(right)).ratio()


def normalize_identifier(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


@dataclass(frozen=True)
class Resolution:
    payload: str
    charger_id: str | None
    confidence: float
    explanation: str


class ChargerResolver:
    def __init__(self, repository: ChargerRepository, radius_meters: float = 500.0) -> None:
        self.repository = repository
        self.radius_meters = radius_meters

    def resolve(
        self,
        candidate: str,
        *,
        latitude: float,
        longitude: float,
        source_confidence: float,
    ) -> Resolution:
        exact = self.repository.get_by_payload(candidate)
        if exact:
            return Resolution(
                payload=exact.qr_payload,
                charger_id=exact.charger_id,
                confidence=min(0.99, 0.85 + 0.15 * source_confidence),
                explanation="Exact payload match in the charger catalogue.",
            )

        nearby = self._nearby(latitude, longitude)
        if not nearby:
            return Resolution(
                payload=candidate,
                charger_id=None,
                confidence=round(source_confidence * 0.75, 4),
                explanation="No known charger was found inside the configured GPS radius.",
            )

        ranked: list[tuple[float, Charger, float, float]] = []
        for charger, distance in nearby:
            similarity = max(
                text_similarity(candidate, charger.qr_payload),
                text_similarity(candidate, charger.charger_id),
            )
            distance_score = max(0.0, 1.0 - distance / self.radius_meters)
            score = 0.55 * similarity + 0.25 * distance_score + 0.20 * source_confidence
            ranked.append((score, charger, similarity, distance))

        score, charger, similarity, distance = max(ranked, key=lambda item: item[0])
        if similarity < 0.55:
            return Resolution(
                payload=candidate,
                charger_id=None,
                confidence=round(min(score, source_confidence * 0.75), 4),
                explanation=(
                    "Nearby chargers exist, but text similarity is below the safety threshold."
                ),
            )
        return Resolution(
            payload=charger.qr_payload,
            charger_id=charger.charger_id,
            confidence=round(min(score, 0.99), 4),
            explanation=(
                f"Selected nearby charger using text similarity {similarity:.3f} "
                f"and distance {distance:.1f} m."
            ),
        )

    def _nearby(self, latitude: float, longitude: float) -> list[tuple[Charger, float]]:
        latitude_delta = degrees(self.radius_meters / EARTH_RADIUS_METERS)
        longitude_scale = max(cos(radians(latitude)), 0.01)
        longitude_delta = latitude_delta / longitude_scale
        candidates = self.repository.list_in_bounding_box(
            min_latitude=latitude - latitude_delta,
            max_latitude=latitude + latitude_delta,
            min_longitude=longitude - longitude_delta,
            max_longitude=longitude + longitude_delta,
        )
        with_distances = [
            (
                charger,
                haversine_meters(latitude, longitude, charger.latitude, charger.longitude),
            )
            for charger in candidates
        ]
        return [item for item in with_distances if item[1] <= self.radius_meters]
