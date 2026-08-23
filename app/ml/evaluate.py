from dataclasses import dataclass

from app.ml.metrics import (
    accuracy_by_degradation,
    character_error_rate,
    exact_match_accuracy,
    qr_recovery_rate,
)
from app.ml.model_registry import EvaluationMetrics


@dataclass(frozen=True)
class EvaluationReport:
    metrics: EvaluationMetrics
    accuracy_by_degradation: dict[str, float]


def evaluate_predictions(
    predictions: list[str],
    references: list[str],
    degradation_types: list[str],
    previously_failed: list[bool],
) -> EvaluationReport:
    grouped = accuracy_by_degradation(predictions, references, degradation_types)
    degraded_values = [value for name, value in grouped.items() if name != "clean_or_unknown"]
    metrics = EvaluationMetrics(
        exact_match_accuracy=exact_match_accuracy(predictions, references),
        character_error_rate=character_error_rate(predictions, references),
        qr_recovery_rate=qr_recovery_rate(predictions, references, previously_failed),
        degraded_scan_accuracy=(
            sum(degraded_values) / len(degraded_values) if degraded_values else 0.0
        ),
        clean_scan_accuracy=grouped.get("clean_or_unknown", 0.0),
    )
    return EvaluationReport(metrics=metrics, accuracy_by_degradation=grouped)
