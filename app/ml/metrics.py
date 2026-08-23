from collections.abc import Sequence


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def exact_match_accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    _validate_pairs(predictions, references)
    if not references:
        return 0.0
    matches = sum(
        prediction.strip() == reference.strip()
        for prediction, reference in zip(predictions, references, strict=True)
    )
    return matches / len(references)


def character_error_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    _validate_pairs(predictions, references)
    total_characters = sum(len(reference) for reference in references)
    if total_characters == 0:
        return 0.0 if all(not prediction for prediction in predictions) else 1.0
    edits = sum(
        levenshtein_distance(prediction, reference)
        for prediction, reference in zip(predictions, references, strict=True)
    )
    return edits / total_characters


def qr_recovery_rate(
    predictions: Sequence[str], references: Sequence[str], previously_failed: Sequence[bool]
) -> float:
    _validate_pairs(predictions, references)
    if len(previously_failed) != len(references):
        raise ValueError("previously_failed must match the prediction count")
    difficult = [
        prediction.strip() == reference.strip()
        for prediction, reference, failed in zip(
            predictions, references, previously_failed, strict=True
        )
        if failed
    ]
    return sum(difficult) / len(difficult) if difficult else 0.0


def accuracy_by_degradation(
    predictions: Sequence[str], references: Sequence[str], degradation_types: Sequence[str]
) -> dict[str, float]:
    _validate_pairs(predictions, references)
    if len(degradation_types) != len(references):
        raise ValueError("degradation_types must match the prediction count")
    grouped: dict[str, list[bool]] = {}
    for prediction, reference, degradation in zip(
        predictions, references, degradation_types, strict=True
    ):
        grouped.setdefault(degradation, []).append(prediction.strip() == reference.strip())
    return {name: sum(matches) / len(matches) for name, matches in grouped.items()}


def _validate_pairs(predictions: Sequence[str], references: Sequence[str]) -> None:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
