from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.services.image_quality import ImageQuality, ImageQualityService
from app.services.preprocessing import ImagePreprocessor
from app.services.qr_decoder import QRDecoder, QRDecodeResult

DEGRADATIONS = (
    "gaussian_blur",
    "motion_blur",
    "too_dark",
    "overexposed",
    "glare",
    "faded",
    "noise",
    "rotation",
    "perspective",
    "partial_occlusion",
    "jpeg_compression",
)


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    chargers: int = 100
    variants_per_charger: int = 8
    seed: int = 42
    width: int = 960
    height: int = 720
    train_ratio: float = 0.70
    validation_ratio: float = 0.15

    def validate(self) -> None:
        if self.chargers < 3:
            raise ValueError("At least three chargers are required for grouped splits")
        if self.variants_per_charger < 2:
            raise ValueError("Use at least two variants so each sticker has a degraded example")
        if self.width < 480 or self.height < 360:
            raise ValueError("Synthetic camera frames must be at least 480x360")
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between zero and one")
        if not 0 < self.validation_ratio < 1:
            raise ValueError("validation_ratio must be between zero and one")
        if self.train_ratio + self.validation_ratio >= 1:
            raise ValueError("The split ratios must leave room for a test set")


@dataclass(frozen=True)
class SyntheticExample:
    example_id: str
    image_path: str
    source: str
    verified: bool
    charger_id: str
    qr_payload: str
    sticker_id: str
    session_id: str
    latitude: float
    longitude: float
    split: str
    degradations: list[str]
    severity: float
    blur_score: float
    brightness_score: float
    brightness_category: str
    native_qr_success: bool
    recovered_qr_success: bool
    recovery_strategy: str | None
    recovery_decoder: str | None
    image_sha256: str


class SyntheticDatasetGenerator:
    """Create reproducible charger-sticker images with exact labels."""

    def __init__(self, config: SyntheticDatasetConfig) -> None:
        config.validate()
        self.config = config
        self.random = random.Random(config.seed)
        self.numpy_rng = np.random.default_rng(config.seed)
        self.quality_service = ImageQualityService()
        self.preprocessor = ImagePreprocessor()
        self.decoder = QRDecoder()

    def generate(self, output_dir: Path) -> Path:
        output_dir = output_dir.resolve()
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        split_by_charger = self._assign_splits()
        examples: list[SyntheticExample] = []

        for charger_index in range(1, self.config.chargers + 1):
            charger_id = f"NL-TAP-E{charger_index:05d}"
            payload = f"https://example.test/charge/{charger_id}"
            split = split_by_charger[charger_id]
            clean_frame = self._render_camera_frame(charger_id, payload)

            for variant_index in range(self.config.variants_per_charger):
                example_id = f"{charger_id}-v{variant_index:02d}"
                if variant_index == 0:
                    degradations: list[str] = ["clean"]
                    severity = 0.0
                    image = clean_frame.copy()
                else:
                    count = 1 if self.random.random() < 0.65 else 2
                    degradations = self.random.sample(DEGRADATIONS, count)
                    severity = round(self.random.uniform(0.35, 0.95), 3)
                    image = clean_frame.copy()
                    for degradation in degradations:
                        image = self._degrade(image, degradation, severity)

                relative_path = Path("images") / split / charger_id / f"{example_id}.jpg"
                image_path = output_dir / relative_path
                image_path.parent.mkdir(parents=True, exist_ok=True)
                encoded_ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not encoded_ok:
                    raise RuntimeError(f"Could not encode {example_id}")
                image_bytes = encoded.tobytes()
                image_path.write_bytes(image_bytes)

                quality = self.quality_service.analyze(image)
                native_decoded = self.decoder.decode_opencv(image)
                recovered, recovery_strategy = self._recover(image, quality, payload)
                examples.append(
                    SyntheticExample(
                        example_id=example_id,
                        image_path=relative_path.as_posix(),
                        source="synthetic",
                        verified=True,
                        charger_id=charger_id,
                        qr_payload=payload,
                        sticker_id=f"sticker-{charger_id}",
                        session_id=f"synthetic-{charger_id}",
                        latitude=round(52.3676 + charger_index * 0.00005, 7),
                        longitude=round(4.9041 + charger_index * 0.00004, 7),
                        split=split,
                        degradations=degradations,
                        severity=severity,
                        blur_score=round(quality.blur_score, 3),
                        brightness_score=round(quality.brightness_score, 3),
                        brightness_category=quality.brightness_category,
                        native_qr_success=bool(
                            native_decoded and native_decoded.payload == payload
                        ),
                        recovered_qr_success=recovered is not None,
                        recovery_strategy=recovery_strategy,
                        recovery_decoder=(recovered.decoder if recovered else None),
                        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
                    )
                )

        manifest = {
            "schema_version": "1.0",
            "dataset_name": "tap-electric-synthetic-stickers",
            "generator": {
                **asdict(self.config),
                "test_ratio": round(1 - self.config.train_ratio - self.config.validation_ratio, 4),
            },
            "examples": [asdict(example) for example in examples],
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def _recover(
        self,
        image: np.ndarray,
        quality: ImageQuality,
        payload: str,
    ) -> tuple[QRDecodeResult | None, str | None]:
        for candidate in self.preprocessor.generate_candidates(image, quality):
            decoded = self.decoder.decode(candidate.image)
            if decoded and decoded.payload == payload:
                return decoded, candidate.strategy
        for candidate in self.preprocessor.restoration_candidates(image):
            decoded = self.decoder.decode(candidate.image)
            if decoded and decoded.payload == payload:
                return decoded, candidate.strategy
        return None, None

    def _assign_splits(self) -> dict[str, str]:
        charger_ids = [f"NL-TAP-E{index:05d}" for index in range(1, self.config.chargers + 1)]
        self.random.shuffle(charger_ids)
        train_count = max(1, round(len(charger_ids) * self.config.train_ratio))
        validation_count = max(1, round(len(charger_ids) * self.config.validation_ratio))
        if train_count + validation_count >= len(charger_ids):
            train_count = len(charger_ids) - 2
            validation_count = 1
        assignment: dict[str, str] = {}
        for index, charger_id in enumerate(charger_ids):
            if index < train_count:
                assignment[charger_id] = "train"
            elif index < train_count + validation_count:
                assignment[charger_id] = "validation"
            else:
                assignment[charger_id] = "test"
        return assignment

    def _render_camera_frame(self, charger_id: str, payload: str) -> np.ndarray:
        try:
            import qrcode
            from qrcode.constants import ERROR_CORRECT_H
        except ImportError as exc:  # pragma: no cover - optional data tooling
            raise RuntimeError("Install the project with the 'notebook' extra") from exc

        qr = qrcode.QRCode(error_correction=ERROR_CORRECT_H, box_size=8, border=4)
        qr.add_data(payload)
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        qr_image = qr_image.resize((300, 300), Image.Resampling.NEAREST)

        sticker = Image.new("RGB", (700, 430), (245, 245, 240))
        draw = ImageDraw.Draw(sticker)
        title_font = self._font(34)
        label_font = self._font(22)
        id_font = self._font(38)
        small_font = self._font(18)
        draw.rounded_rectangle((4, 4, 696, 426), radius=22, outline=(35, 55, 65), width=5)
        draw.text((34, 18), "TAP ELECTRIC", fill=(20, 130, 95), font=title_font)
        sticker.paste(qr_image, (32, 92))
        draw.text((370, 112), "CHARGER ID", fill=(75, 75, 75), font=label_font)
        draw.text((370, 150), charger_id, fill=(15, 15, 15), font=id_font)
        draw.text((370, 232), "Scan to start charging", fill=(45, 45, 45), font=small_font)
        draw.text((370, 272), "Connector 1", fill=(45, 45, 45), font=small_font)
        draw.text((370, 340), "Help: tap-electric.example", fill=(90, 90, 90), font=small_font)

        background_color = self.numpy_rng.integers(45, 190, size=3, dtype=np.uint8)
        frame = np.full(
            (self.config.height, self.config.width, 3), background_color, dtype=np.uint8
        )
        texture = self.numpy_rng.normal(0, 5, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + texture, 0, 255).astype(np.uint8)
        sticker_bgr = cv2.cvtColor(np.array(sticker), cv2.COLOR_RGB2BGR)
        target_width = int(self.config.width * 0.72)
        scale = target_width / sticker_bgr.shape[1]
        sticker_bgr = cv2.resize(
            sticker_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        sticker_height, sticker_width = sticker_bgr.shape[:2]
        x = (self.config.width - sticker_width) // 2
        y = (self.config.height - sticker_height) // 2
        frame[y : y + sticker_height, x : x + sticker_width] = sticker_bgr
        return frame

    def _degrade(self, image: np.ndarray, name: str, severity: float) -> np.ndarray:
        if name == "gaussian_blur":
            kernel = 3 + 2 * int(severity * 4)
            return cv2.GaussianBlur(image, (kernel, kernel), 0)
        if name == "motion_blur":
            size = 5 + 2 * int(severity * 8)
            kernel = np.zeros((size, size), dtype=np.float32)
            kernel[size // 2, :] = 1.0 / size
            return cv2.filter2D(image, -1, kernel)
        if name == "too_dark":
            return np.clip(image.astype(np.float32) * (0.72 - 0.5 * severity), 0, 255).astype(
                np.uint8
            )
        if name == "overexposed":
            return np.clip(image.astype(np.float32) * (1.1 + severity) + 25, 0, 255).astype(
                np.uint8
            )
        if name == "glare":
            overlay = image.copy()
            center = (
                self.random.randint(image.shape[1] // 4, image.shape[1] * 3 // 4),
                self.random.randint(image.shape[0] // 4, image.shape[0] * 3 // 4),
            )
            cv2.ellipse(
                overlay,
                center,
                (int(160 * severity), int(75 * severity)),
                self.random.randint(-30, 30),
                0,
                360,
                (255, 255, 255),
                -1,
            )
            return cv2.addWeighted(overlay, 0.55, image, 0.45, 0)
        if name == "faded":
            pale = np.full_like(image, 225)
            return cv2.addWeighted(image, 1 - severity * 0.55, pale, severity * 0.55, 0)
        if name == "noise":
            noise = self.numpy_rng.normal(0, 8 + severity * 24, image.shape)
            return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if name == "rotation":
            angle = self.random.uniform(-14, 14) * severity
            matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
            return cv2.warpAffine(
                image,
                matrix,
                (image.shape[1], image.shape[0]),
                borderMode=cv2.BORDER_REPLICATE,
            )
        if name == "perspective":
            height, width = image.shape[:2]
            shift = min(height, width) * 0.12 * severity
            source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            target = np.float32(
                [[shift, 0], [width - shift, shift], [width, height], [0, height - shift]]
            )
            return cv2.warpPerspective(
                image,
                cv2.getPerspectiveTransform(source, target),
                (width, height),
                borderMode=cv2.BORDER_REPLICATE,
            )
        if name == "partial_occlusion":
            result = image.copy()
            width = int(image.shape[1] * (0.04 + severity * 0.08))
            x = self.random.randint(image.shape[1] // 4, image.shape[1] * 3 // 4)
            y = self.random.randint(image.shape[0] // 4, image.shape[0] * 3 // 4)
            cv2.rectangle(result, (x, y), (x + width, y + width), (85, 85, 85), -1)
            return result
        if name == "jpeg_compression":
            quality = max(12, int(70 - severity * 58))
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else image
        raise ValueError(f"Unknown degradation: {name}")

    @staticmethod
    def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        for name in ("DejaVuSans.ttf", "arial.ttf"):
            try:
                return ImageFont.truetype(name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()
