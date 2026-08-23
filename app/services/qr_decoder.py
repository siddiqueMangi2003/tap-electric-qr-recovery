from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import zxingcpp

DecoderName = Literal["opencv", "zxingcpp"]


@dataclass(frozen=True)
class QRDecodeResult:
    payload: str
    confidence: float
    decoder: DecoderName


class QRDecoder:
    """Deterministic decoder ensemble: OpenCV first, then ZXing-C++."""

    def __init__(self) -> None:
        self.detector = cv2.QRCodeDetector()

    def decode(self, image: np.ndarray) -> QRDecodeResult | None:
        return self.decode_opencv(image) or self.decode_zxing(image)

    def decode_opencv(self, image: np.ndarray) -> QRDecodeResult | None:
        try:
            payload, points, _ = self.detector.detectAndDecode(image)
        except cv2.error:
            return None
        if not payload:
            return None
        confidence = 0.98 if points is not None else 0.90
        return QRDecodeResult(
            payload=payload.strip(),
            confidence=confidence,
            decoder="opencv",
        )

    @staticmethod
    def decode_zxing(image: np.ndarray) -> QRDecodeResult | None:
        try:
            barcode = zxingcpp.read_barcode(
                image,
                formats=zxingcpp.BarcodeFormat.QRCode,
                try_rotate=True,
                try_downscale=True,
                try_invert=True,
            )
        except (IndexError, RuntimeError, TypeError, ValueError):
            return None
        if barcode is None or not barcode.text:
            return None
        return QRDecodeResult(
            payload=barcode.text.strip(),
            confidence=0.96,
            decoder="zxingcpp",
        )
