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
class ChargerCandidate:
    charger_id: str
    qr_payload: str
    distance_meters: float
    text_similarity: float
    match_score: float


@dataclass(frozen=True)
class Resolution:
    payload: str
    charger_id: str | None
    confidence: float
    explanation: str
    candidates: tuple[ChargerCandidate, ...] = ()


class ChargerResolver:
    def __init__(
        self,
        repository: ChargerRepository,
        radius_meters: float = 500.0,
        *,
        maximum_candidates: int = 3,
        minimum_text_similarity: float = 0.55,
        minimum_winner_margin: float = 0.05,
    ) -> None:
        self.repository = repository
        self.radius_meters = radius_meters
        self.maximum_candidates = maximum_candidates
        self.minimum_text_similarity = minimum_text_similarity
        self.minimum_winner_margin = minimum_winner_margin

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

        ranked = self.rank_candidates(candidate, latitude=latitude, longitude=longitude)
        if not ranked:
            return Resolution(
                payload=candidate,
                charger_id=None,
                confidence=round(source_confidence * 0.75, 4),
                explanation="No known charger was found inside the configured GPS radius.",
            )

        best = ranked[0]
        score = 0.80 * best.match_score + 0.20 * source_confidence
        if best.text_similarity < self.minimum_text_similarity:
            return Resolution(
                payload=candidate,
                charger_id=None,
                confidence=round(min(score, source_confidence * 0.75), 4),
                explanation=(
                    "Nearby chargers exist, but text similarity is below the safety threshold."
                ),
                candidates=tuple(ranked),
            )
        winner_margin = best.match_score - ranked[1].match_score if len(ranked) > 1 else 1.0
        if winner_margin < self.minimum_winner_margin:
            return Resolution(
                payload=candidate,
                charger_id=None,
                confidence=round(min(score, 0.69), 4),
                explanation=(
                    "Several nearby chargers have similar match scores; "
                    "manual selection is required."
                ),
                candidates=tuple(ranked),
            )
        return Resolution(
            payload=best.qr_payload,
            charger_id=best.charger_id,
            confidence=round(min(score, 0.99), 4),
            explanation=(
                f"Selected nearby charger using text similarity {best.text_similarity:.3f} "
                f"and distance {best.distance_meters:.1f} m."
            ),
        )

    def rank_candidates(
        self,
        candidate: str,
        *,
        latitude: float,
        longitude: float,
    ) -> list[ChargerCandidate]:
        """Return the strongest nearby matches for a manual fallback choice."""
        ranked: list[ChargerCandidate] = []
        for charger, distance in self._nearby(latitude, longitude):
            similarity = max(
                text_similarity(candidate, charger.qr_payload),
                text_similarity(candidate, charger.charger_id),
            )
            distance_score = max(0.0, 1.0 - distance / self.radius_meters)
            # Preserve the resolver's existing 55:25 text/distance weighting,
            # normalized so the candidate score remains independent of model confidence.
            match_score = 0.6875 * similarity + 0.3125 * distance_score
            ranked.append(
                ChargerCandidate(
                    charger_id=charger.charger_id,
                    qr_payload=charger.qr_payload,
                    distance_meters=round(distance, 1),
                    text_similarity=round(similarity, 4),
                    match_score=round(match_score, 4),
                )
            )
        ranked.sort(key=lambda item: (-item.match_score, item.distance_meters, item.charger_id))
        return ranked[: self.maximum_candidates]

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
