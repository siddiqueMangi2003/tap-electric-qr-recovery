import random

import cv2
import numpy as np


class ScanAugmenter:
    """Simulate real charger-sticker failures without changing the text label."""

    def __init__(self, probability: float = 0.7, seed: int | None = None) -> None:
        self.probability = probability
        self.random = random.Random(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.random.random() > self.probability:
            return image
        operations = [
            self.gaussian_blur,
            self.motion_blur,
            self.adjust_brightness,
            self.add_noise,
            self.reduce_contrast,
            self.rotate,
            self.perspective,
        ]
        chosen = self.random.sample(operations, k=self.random.randint(1, 3))
        result = image.copy()
        for operation in chosen:
            result = operation(result)
        return result

    def gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        kernel = self.random.choice((3, 5, 7))
        return cv2.GaussianBlur(image, (kernel, kernel), 0)

    def motion_blur(self, image: np.ndarray) -> np.ndarray:
        size = self.random.choice((5, 7, 9))
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1.0 / size
        return cv2.filter2D(image, -1, kernel)

    def adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        factor = self.random.choice((0.4, 0.6, 1.4, 1.8))
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        sigma = self.random.uniform(5, 20)
        noise = np.random.default_rng(self.random.randrange(2**32)).normal(0, sigma, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def reduce_contrast(self, image: np.ndarray) -> np.ndarray:
        mean = image.mean(axis=(0, 1), keepdims=True)
        return np.clip(mean + 0.45 * (image - mean), 0, 255).astype(np.uint8)

    def rotate(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        angle = self.random.uniform(-12, 12)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    def perspective(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        shift = min(width, height) * self.random.uniform(0.03, 0.10)
        source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        target = np.float32(
            [[shift, 0], [width - shift, shift], [width, height - shift], [0, height]]
        )
        matrix = cv2.getPerspectiveTransform(source, target)
        return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
