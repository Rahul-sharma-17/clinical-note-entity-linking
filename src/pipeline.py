"""Detect mentions and attach the highest-ranked ontology concept on CPU."""

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

from .baseline import ExactStringLookup
from .config import BENCHMARK_METRICS_PATH
from .data_loader import Annotation, Dataset, load_dataset
from .entity_linker import EntityLinker, load_linker
from .metrics import Scores, evaluate
from .ner_predictor import TrainingDictionaryNER, TransformerNER


@dataclass(frozen=True)
class MentionCandidate:
    concept_id: str
    preferred_term: str
    confidence: float


@dataclass(frozen=True)
class LinkedMention:
    start: int
    end: int
    text: str
    concept_id: str | None
    concept_name: str | None
    confidence: float | None
    candidates: tuple[MentionCandidate, ...] = ()


class Pipeline:
    def __init__(
        self,
        detector: TrainingDictionaryNER | TransformerNER,
        linker: EntityLinker,
    ):
        self.detector = detector
        self.linker = linker

    def predict(self, note_text: str) -> list[LinkedMention]:
        if not note_text.strip():
            return []

        predictions = []
        for mention in self.detector.predict(note_text):
            surface = note_text[mention.start:mention.end]
            candidates = self.linker.link(surface, top_k=3)
            top = candidates[0] if candidates else None
            predictions.append(LinkedMention(
                start=mention.start,
                end=mention.end,
                text=surface,
                concept_id=top.concept_id if top else None,
                concept_name=top.concept_name if top else None,
                confidence=top.confidence if top else None,
                candidates=tuple(
                    MentionCandidate(item.concept_id, item.concept_name, item.confidence)
                    for item in candidates
                ),
            ))
        return predictions


def load_pipeline(dataset: Dataset | None = None) -> Pipeline:
    dataset = dataset if dataset is not None else load_dataset()
    return Pipeline(TrainingDictionaryNER(dataset), load_linker(dataset))


@lru_cache(maxsize=1)
def _default_pipeline() -> Pipeline:
    return load_pipeline()


def predict(note_text: str) -> list[LinkedMention]:
    """Predict spans and links with the configured local model or fallback."""
    if not note_text.strip():
        return []
    return _default_pipeline().predict(note_text)


def benchmark(dataset: Dataset, pipeline: Pipeline) -> tuple[Scores, Scores]:
    note_ids = dataset.splits["validation"]
    gold = [row for row in dataset.annotations if row.note_id in note_ids]
    baseline = ExactStringLookup(dataset.terminology)
    baseline_rows = []
    pipeline_rows = []
    for note_id in sorted(note_ids):
        text = dataset.notes[note_id]
        baseline_rows.extend(baseline.predict(note_id, text))
        pipeline_rows.extend(
            Annotation(note_id, hit.start, hit.end, hit.concept_id or "")
            for hit in pipeline.predict(text)
        )
    return evaluate(gold, baseline_rows), evaluate(gold, pipeline_rows)


def main() -> None:
    dataset = load_dataset()
    pipeline = load_pipeline(dataset)
    baseline, final = benchmark(dataset, pipeline)
    validation_count = len(dataset.splits["validation"])
    detector = type(pipeline.detector).__name__
    retriever = type(pipeline.linker.retriever).__name__
    metrics = {
        "validation_notes": validation_count,
        "detector": detector,
        "retriever": retriever,
        "baseline": asdict(baseline),
        "system": asdict(final),
    }
    BENCHMARK_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=BENCHMARK_METRICS_PATH.parent, delete=False
    )
    temporary = Path(output.name)
    try:
        with output:
            json.dump(metrics, output, indent=2, allow_nan=False)
            output.write("\n")
        temporary.replace(BENCHMARK_METRICS_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Validation: {validation_count:,} notes")
    print(f"Final system: {detector} + {retriever}")
    print("| System | Mention F1 | Linking Accuracy | End-to-End F1 |")
    print("| --- | ---: | ---: | ---: |")
    for name, scores in [("Exact String Lookup Baseline", baseline), ("Final System", final)]:
        print(f"| {name} | {scores.mention_f1:.6f} | {scores.linking_accuracy:.6f} "
              f"| {scores.end_to_end_f1:.6f} |")
    print(f"Saved metrics to {BENCHMARK_METRICS_PATH}")


if __name__ == "__main__":
    main()

