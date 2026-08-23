from dataclasses import dataclass

import cv2
import numpy as np

from app.services.image_quality import ImageQuality


@dataclass(frozen=True)
class PreprocessedImage:
    strategy: str
    image: np.ndarray


class ImagePreprocessor:
    """Generate a small, quality-directed set of QR decoding candidates."""

    def generate_candidates(
        self, image: np.ndarray, quality: ImageQuality
    ) -> list[PreprocessedImage]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = [PreprocessedImage("original", image)]

        if quality.brightness_category == "too_dark" or quality.contrast_score < 35:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
            candidates.append(PreprocessedImage("clahe", clahe))
            thresholded = cv2.adaptiveThreshold(
                clahe,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                5,
            )
            candidates.append(PreprocessedImage("clahe_adaptive_threshold", thresholded))

        if quality.brightness_category == "too_bright":
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            _, otsu = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            candidates.append(PreprocessedImage("normalize_otsu", otsu))

        if quality.is_blurry:
            denoised = cv2.GaussianBlur(gray, (3, 3), 0)
            sharpened = cv2.addWeighted(gray, 1.8, denoised, -0.8, 0)
            candidates.append(PreprocessedImage("unsharp_mask", sharpened))

        if min(gray.shape[:2]) < 700:
            upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            candidates.append(PreprocessedImage("bicubic_upscale", upscaled))

        return candidates

    def restoration_candidates(self, image: np.ndarray) -> list[PreprocessedImage]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        enlarged = cv2.resize(denoised, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(enlarged)
        binary = cv2.adaptiveThreshold(
            clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            7,
        )
        return [
            PreprocessedImage("restore_denoise_upscale_clahe", clahe),
            PreprocessedImage("restore_adaptive_threshold", binary),
        ]
