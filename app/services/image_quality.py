from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

BrightnessCategory = Literal["too_dark", "normal", "too_bright"]


@dataclass(frozen=True)
class ImageQuality:
    blur_score: float
    brightness_score: float
    brightness_category: BrightnessCategory
    is_blurry: bool
    contrast_score: float


class ImageQualityService:
    def __init__(
        self,
        *,
        blur_threshold: float = 100.0,
        dark_threshold: float = 65.0,
        bright_threshold: float = 205.0,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold

    def analyze(self, image: np.ndarray) -> ImageQuality:
        gray = self._grayscale(image)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_score = float(gray.mean())
        contrast_score = float(gray.std())
        if brightness_score < self.dark_threshold:
            category: BrightnessCategory = "too_dark"
        elif brightness_score > self.bright_threshold:
            category = "too_bright"
        else:
            category = "normal"
        return ImageQuality(
            blur_score=blur_score,
            brightness_score=brightness_score,
            brightness_category=category,
            is_blurry=blur_score < self.blur_threshold,
            contrast_score=contrast_score,
        )

    @staticmethod
    def _grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
