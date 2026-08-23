from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationMetrics:
    exact_match_accuracy: float
    character_error_rate: float
    qr_recovery_rate: float
    degraded_scan_accuracy: float
    clean_scan_accuracy: float


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str


class ModelPromotionPolicy:
    """Prefer end-to-end recovery while guarding clean-image performance."""

    def __init__(
        self,
        minimum_exact_match_gain: float = 0.01,
        maximum_clean_accuracy_drop: float = 0.01,
    ) -> None:
        self.minimum_exact_match_gain = minimum_exact_match_gain
        self.maximum_clean_accuracy_drop = maximum_clean_accuracy_drop

    def decide(
        self,
        candidate: EvaluationMetrics,
        incumbent: EvaluationMetrics | None,
    ) -> PromotionDecision:
        if incumbent is None:
            return PromotionDecision(True, "No incumbent model exists.")
        exact_gain = candidate.exact_match_accuracy - incumbent.exact_match_accuracy
        clean_drop = incumbent.clean_scan_accuracy - candidate.clean_scan_accuracy
        if exact_gain < self.minimum_exact_match_gain:
            return PromotionDecision(False, "Exact-match improvement is below the promotion gate.")
        if clean_drop > self.maximum_clean_accuracy_drop:
            return PromotionDecision(False, "Candidate regresses too much on clean images.")
        if candidate.degraded_scan_accuracy < incumbent.degraded_scan_accuracy:
            return PromotionDecision(False, "Candidate regresses on degraded scans.")
        return PromotionDecision(
            True, "Candidate improves recovery without a material clean regression."
        )
