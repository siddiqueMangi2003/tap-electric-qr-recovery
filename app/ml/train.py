from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.ml.augmentations import ScanAugmenter
from app.ml.dataset import DatasetExample, DatasetManifest
from app.ml.metrics import character_error_rate, exact_match_accuracy
from app.services.storage.base import ObjectStorage


class TrocrFineTuner:
    """Real Hugging Face fine-tuning path, imported only for non-dry runs."""

    def __init__(self, model_name: str, storage: ObjectStorage) -> None:
        self.model_name = model_name
        self.storage = storage

    def train(
        self,
        manifest: DatasetManifest,
        output_dir: Path,
        *,
        epochs: int = 3,
    ) -> tuple[Path, dict[str, float]]:
        if not manifest.examples:
            raise ValueError("At least one verified example is required for training")
        try:
            from torch.utils.data import Dataset
            from transformers import (
                Seq2SeqTrainer,
                Seq2SeqTrainingArguments,
                TrOCRProcessor,
                VisionEncoderDecoderModel,
            )
        except ImportError as exc:  # pragma: no cover - optional integration
            raise RuntimeError("Install the project with the 'ml' extra for real training") from exc

        processor = TrOCRProcessor.from_pretrained(self.model_name)
        model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

        storage = self.storage
        augmenter = ScanAugmenter()

        class OCRDataset(Dataset):
            def __init__(self, examples: list[DatasetExample], augment: bool) -> None:
                self.examples = examples
                self.augment = augment

            def __len__(self) -> int:
                return len(self.examples)

            def __getitem__(self, index: int) -> dict[str, Any]:
                example = self.examples[index]
                data = np.frombuffer(storage.get_image(example.image_uri), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Cannot decode training image {example.scan_id}")
                image = TrocrFineTuner._printed_identifier_crop(image)
                if self.augment:
                    image = augmenter(image)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pixel_values = processor(
                    images=Image.fromarray(rgb), return_tensors="pt"
                ).pixel_values[0]
                tokenized = processor.tokenizer(
                    example.charger_id,
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids[0]
                tokenized[tokenized == processor.tokenizer.pad_token_id] = -100
                return {"pixel_values": pixel_values, "labels": tokenized}

        train_examples = [item for item in manifest.examples if item.split == "train"]
        validation_examples = [item for item in manifest.examples if item.split == "validation"]
        test_examples = [item for item in manifest.examples if item.split == "test"]
        if not train_examples:
            raise ValueError("Grouped split produced no training examples; collect more chargers")
        evaluation_examples = test_examples or validation_examples

        output_dir.mkdir(parents=True, exist_ok=True)
        arguments = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=5e-5,
            predict_with_generate=True,
            generation_max_length=64,
            eval_strategy="epoch" if validation_examples else "no",
            save_strategy="epoch" if validation_examples else "no",
            load_best_model_at_end=bool(validation_examples),
            metric_for_best_model="exact_match",
            greater_is_better=True,
            report_to=[],
        )

        def compute_metrics(prediction: Any) -> dict[str, float]:
            prediction_ids = prediction.predictions
            label_ids = prediction.label_ids.copy()
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            predictions = processor.batch_decode(prediction_ids, skip_special_tokens=True)
            references = processor.batch_decode(label_ids, skip_special_tokens=True)
            return {
                "exact_match": exact_match_accuracy(predictions, references),
                "character_error_rate": character_error_rate(predictions, references),
            }

        trainer = Seq2SeqTrainer(
            model=model,
            args=arguments,
            train_dataset=OCRDataset(train_examples, augment=True),
            eval_dataset=OCRDataset(validation_examples, augment=False)
            if validation_examples
            else None,
            processing_class=processor,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        final_path = output_dir / "final"
        trainer.save_model(final_path)
        processor.save_pretrained(final_path)
        metrics = (
            trainer.evaluate(
                OCRDataset(evaluation_examples, augment=False), metric_key_prefix="test"
            )
            if evaluation_examples
            else {}
        )
        return final_path, {
            key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))
        }

    @staticmethod
    def _printed_identifier_crop(image: np.ndarray) -> np.ndarray:
        height = image.shape[0]
        return image[int(height * 0.55) : height, :]
