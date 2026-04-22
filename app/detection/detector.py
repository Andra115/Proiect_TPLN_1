"""
detection/detector.py

Main pipeline for §3.3: ties together char-level diff and entropy scoring.

Typical usage
-------------
    detector = build_detector(reference_texts)   # once at startup
    result   = detector.run(input_text, output_text)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

from .diff import Change, char_level_diff
from .scorer import DiacriticScorer, UncertaintyScore


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    position: int
    original_char: str
    corrected_char: str
    context: str
    change_type: str
    probability: float
    entropy: float
    flag_for_review: bool
    scoring_method: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectionReport:
    input_text: str
    corrected_text: str
    total_changes: int
    flagged_count: int
    avg_entropy: float
    detections: list[DetectionResult]

    def to_dict(self) -> dict:
        return {
            "input_text": self.input_text,
            "corrected_text": self.corrected_text,
            "total_changes": self.total_changes,
            "flagged_count": self.flagged_count,
            "avg_entropy": self.avg_entropy,
            "detections": [d.to_dict() for d in self.detections],
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    def __init__(self, scorer: DiacriticScorer):
        self._scorer = scorer

    def run(self, input_text: str, output_text: str) -> DetectionReport:
        changes: list[Change] = char_level_diff(input_text, output_text)
        scores: list[UncertaintyScore] = self._scorer.score_all(changes, output_text)

        detections = [
            DetectionResult(
                position=ch.position,
                original_char=ch.original_char,
                corrected_char=ch.corrected_char,
                context=ch.context,
                change_type=ch.change_type,
                probability=sc.probability,
                entropy=sc.entropy,
                flag_for_review=sc.flag_for_review,
                scoring_method=sc.method,
            )
            for ch, sc in zip(changes, scores)
        ]

        flagged = [d for d in detections if d.flag_for_review]
        avg_entropy = (
            round(sum(d.entropy for d in detections) / len(detections), 6)
            if detections else 0.0
        )

        return DetectionReport(
            input_text=input_text,
            corrected_text=output_text,
            total_changes=len(detections),
            flagged_count=len(flagged),
            avg_entropy=avg_entropy,
            detections=detections,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_detector(
    reference_texts: Sequence[str],
    review_threshold: float = 0.8,
) -> Detector:
    """
    Build and return a ready-to-use Detector.

    Parameters
    ----------
    reference_texts:  Romanian sentences/paragraphs used to train the n-gram LM.
                      More text → better probability estimates.
    review_threshold: Shannon entropy (bits) above which a change is flagged.
                      Default 0.8 works well for the standard confusion pairs.
    """
    scorer = DiacriticScorer(review_threshold=review_threshold)
    scorer.fit(reference_texts)
    return Detector(scorer)