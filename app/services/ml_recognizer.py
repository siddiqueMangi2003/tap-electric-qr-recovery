from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class OCRPrediction:
    text: str
    confidence: float


class TrocrRecognizer:
    """Lazy TrOCR adapter for printed sticker identifiers, not QR matrix decoding."""

    def __init__(
        self,
        model_name: str,
        *,
        enabled: bool = False,
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self.local_files_only = local_files_only
        self._processor = None
        self._model = None

    def predict(self, image: np.ndarray) -> OCRPrediction | None:
        if not self.enabled:
            return None
        self._load()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        pixel_values = self._processor(
            images=Image.fromarray(rgb), return_tensors="pt"
        ).pixel_values
        generated_ids = self._model.generate(pixel_values, max_new_tokens=64)
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return OCRPrediction(text=text, confidence=0.55) if text else None

    @staticmethod
    def crop_printed_identifier_region(image: np.ndarray) -> np.ndarray:
        """Preserve the lower sticker area where printed IDs are commonly located."""

        height = image.shape[0]
        return image[int(height * 0.55) : height, :]

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:  # pragma: no cover - optional integration
            raise RuntimeError("Install the project with the 'ml' extra to enable TrOCR") from exc
        self._processor = TrOCRProcessor.from_pretrained(
            self.model_name, local_files_only=self.local_files_only
        )
        self._model = VisionEncoderDecoderModel.from_pretrained(
            self.model_name, local_files_only=self.local_files_only
        )
        self._model.eval()
