import cv2
import numpy as np
import qrcode

from app.services.image_quality import ImageQualityService
from app.services.qr_decoder import QRDecoder


def test_image_quality_detects_dark_and_bright_images() -> None:
    service = ImageQualityService()
    dark = np.full((100, 100, 3), 20, dtype=np.uint8)
    bright = np.full((100, 100, 3), 240, dtype=np.uint8)
    assert service.analyze(dark).brightness_category == "too_dark"
    assert service.analyze(bright).brightness_category == "too_bright"


def test_variance_of_laplacian_distinguishes_blur() -> None:
    service = ImageQualityService()
    checkerboard = np.indices((200, 200)).sum(axis=0) % 2 * 255
    checkerboard = checkerboard.astype(np.uint8)
    blurred = cv2.GaussianBlur(checkerboard, (21, 21), 0)
    assert service.analyze(checkerboard).blur_score > service.analyze(blurred).blur_score


def test_opencv_qr_decoder_recovers_real_payload() -> None:
    payload = "https://tap-electric.com/c/NL-TAP-E12345"
    qr_image = np.array(qrcode.make(payload).convert("RGB"))
    bgr_image = cv2.cvtColor(qr_image, cv2.COLOR_RGB2BGR)
    result = QRDecoder().decode(bgr_image)
    assert result is not None
    assert result.payload == payload
