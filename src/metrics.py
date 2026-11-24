"""Micro-averaged metrics for exact character spans and concept IDs."""

from collections.abc import Iterable
from dataclasses import dataclass

from .data_loader import Annotation


@dataclass(frozen=True)
class Scores:
    mention_f1: float
    linking_accuracy: float
    end_to_end_f1: float


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(gold: Iterable[Annotation], predicted: Iterable[Annotation]) -> Scores:
    """Score unique spans and links. Linking accuracy is conditional on exact spans.

    When a span has several gold IDs, any shared ID counts as a linking hit.
    End-to-end F1 still compares individual span/concept pairs.
    An empty predicted concept ID records a detected span with no link.
    """
    gold_links = {(row.note_id, row.start, row.end, row.concept_id) for row in gold}
    predicted_links = {(row.note_id, row.start, row.end, row.concept_id) for row in predicted}
    gold_spans = {link[:3] for link in gold_links}
    predicted_spans = {link[:3] for link in predicted_links}
    predicted_links = {link for link in predicted_links if link[3]}
    matched_spans = gold_spans & predicted_spans
    matched_links = gold_links & predicted_links
    correctly_linked_spans = {link[:3] for link in matched_links}
    return Scores(
        mention_f1=_ratio(2 * len(matched_spans), len(gold_spans) + len(predicted_spans)),
        linking_accuracy=_ratio(len(correctly_linked_spans), len(matched_spans)),
        end_to_end_f1=_ratio(2 * len(matched_links), len(gold_links) + len(predicted_links)),
    )

