from dataclasses import dataclass

import cv2
import numpy as np

from app.repositories.chargers import ChargerRepository
from app.services.charger_resolver import ChargerResolver
from app.services.image_quality import ImageQualityService
from app.services.ml_recognizer import TrocrRecognizer
from app.services.preprocessing import ImagePreprocessor, PreprocessedImage
from app.services.qr_decoder import QRDecoder


@dataclass(frozen=True)
class InferenceResult:
    blur_score: float
    brightness_score: float
    brightness_category: str
    preprocessing_strategy: str
    qr_decoder_result: str | None
    ml_prediction: str | None
    final_prediction: str | None
    resolved_charger_id: str | None
    prediction_source: str | None
    confidence: float
    resolution_explanation: str


class InvalidImageError(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise InvalidImageError("Uploaded bytes are not a supported, decodable image")
    return image


class InferencePipeline:
    def __init__(
        self,
        *,
        image_quality: ImageQualityService,
        preprocessor: ImagePreprocessor,
        qr_decoder: QRDecoder,
        recognizer: TrocrRecognizer,
        resolver_radius_meters: float,
    ) -> None:
        self.image_quality = image_quality
        self.preprocessor = preprocessor
        self.qr_decoder = qr_decoder
        self.recognizer = recognizer
        self.resolver_radius_meters = resolver_radius_meters

    def run(
        self,
        image_bytes: bytes,
        *,
        latitude: float,
        longitude: float,
        charger_repository: ChargerRepository,
        native_qr_success: bool = False,
        native_qr_result: str | None = None,
    ) -> InferenceResult:
        image = decode_image(image_bytes)
        quality = self.image_quality.analyze(image)

        qr_payload: str | None = None
        ml_prediction: str | None = None
        candidate: str | None = None
        source: str | None = None
        source_confidence = 0.0
        strategy = "none"

        if native_qr_success and native_qr_result:
            candidate = native_qr_result.strip()
            source = "native_qr"
            source_confidence = 0.99
            strategy = "native_decoder_result"
        else:
            qr_payload, strategy, source_confidence = self._try_qr_candidates(
                self.preprocessor.generate_candidates(image, quality)
            )
            if qr_payload is None:
                qr_payload, strategy, source_confidence = self._try_qr_candidates(
                    self.preprocessor.restoration_candidates(image)
                )
            if qr_payload:
                candidate = qr_payload
                source = "server_qr_decoder"
            else:
                crop = self.recognizer.crop_printed_identifier_region(image)
                prediction = self.recognizer.predict(crop)
                if prediction:
                    candidate = prediction.text
                    ml_prediction = prediction.text
                    source = "trocr_printed_identifier"
                    source_confidence = prediction.confidence
                    strategy = f"{strategy}+printed_identifier_crop"

        if not candidate:
            return InferenceResult(
                blur_score=quality.blur_score,
                brightness_score=quality.brightness_score,
                brightness_category=quality.brightness_category,
                preprocessing_strategy=strategy,
                qr_decoder_result=qr_payload,
                ml_prediction=ml_prediction,
                final_prediction=None,
                resolved_charger_id=None,
                prediction_source=None,
                confidence=0.0,
                resolution_explanation=(
                    "Neither deterministic QR decoding nor the configured OCR fallback "
                    "produced a candidate."
                ),
            )

        resolution = ChargerResolver(
            charger_repository, radius_meters=self.resolver_radius_meters
        ).resolve(
            candidate,
            latitude=latitude,
            longitude=longitude,
            source_confidence=source_confidence,
        )
        return InferenceResult(
            blur_score=quality.blur_score,
            brightness_score=quality.brightness_score,
            brightness_category=quality.brightness_category,
            preprocessing_strategy=strategy,
            qr_decoder_result=qr_payload,
            ml_prediction=ml_prediction,
            final_prediction=resolution.payload,
            resolved_charger_id=resolution.charger_id,
            prediction_source=source,
            confidence=resolution.confidence,
            resolution_explanation=resolution.explanation,
        )

    def _try_qr_candidates(
        self, candidates: list[PreprocessedImage]
    ) -> tuple[str | None, str, float]:
        last_strategy = "none"
        for candidate in candidates:
            last_strategy = candidate.strategy
            decoded = self.qr_decoder.decode(candidate.image)
            if decoded:
                strategy = f"{candidate.strategy}:{decoded.decoder}"
                return decoded.payload, strategy, decoded.confidence
        return None, last_strategy, 0.0
