from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QRDecodeResult:
    payload: str
    confidence: float


class QRDecoder:
    """Primary deterministic decoder based on OpenCV's QRCodeDetector."""

    def __init__(self) -> None:
        self.detector = cv2.QRCodeDetector()

    def decode(self, image: np.ndarray) -> QRDecodeResult | None:
        try:
            payload, points, _ = self.detector.detectAndDecode(image)
        except cv2.error:
            return None
        if not payload:
            return None
        confidence = 0.98 if points is not None else 0.90
        return QRDecodeResult(payload=payload.strip(), confidence=confidence)
