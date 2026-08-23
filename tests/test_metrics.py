import pytest

from app.ml.evaluate import evaluate_predictions
from app.ml.metrics import character_error_rate, exact_match_accuracy, levenshtein_distance


def test_metric_calculations() -> None:
    predictions = ["ABC123", "XYZ98"]
    references = ["ABC123", "XYZ99"]
    assert exact_match_accuracy(predictions, references) == 0.5
    assert character_error_rate(predictions, references) == pytest.approx(1 / 11)
    assert levenshtein_distance("E1234S", "E12345") == 1


def test_evaluation_reports_degradation_slices() -> None:
    report = evaluate_predictions(
        predictions=["A", "wrong", "C"],
        references=["A", "B", "C"],
        degradation_types=["clean_or_unknown", "blurry", "too_dark"],
        previously_failed=[False, True, True],
    )
    assert report.metrics.exact_match_accuracy == pytest.approx(2 / 3)
    assert report.metrics.qr_recovery_rate == 0.5
    assert report.accuracy_by_degradation["blurry"] == 0.0
    assert report.accuracy_by_degradation["too_dark"] == 1.0
