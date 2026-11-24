"""CPU mention detection with a train-only dictionary or a local BIO model."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, quantiles
from time import perf_counter

from .baseline import ExactStringLookup
from .config import NER_MODEL_DIR
from .data_loader import Annotation, Concept, Dataset, load_dataset
from .metrics import evaluate


@dataclass(frozen=True)
class Mention:
    start: int
    end: int


class TrainingDictionaryNER:
    """Learn literal surfaces from annotated training notes."""

    def __init__(self, dataset: Dataset):
        train_ids = dataset.splits["train"]
        surfaces = sorted({
            dataset.notes[row.note_id][row.start:row.end].strip()
            for row in dataset.annotations if row.note_id in train_ids
        })
        self.matcher = ExactStringLookup({surface: Concept(surface, ()) for surface in surfaces})

    def predict(self, text: str) -> list[Mention]:
        return [Mention(hit.start, hit.end) for hit in self.matcher.predict("", text)]


def decode_bio(tagged_offsets):
    """Decode B/I/O tags into spans. Treat an orphan I as a new mention."""
    mentions = []
    active = None
    for (start, end), tag in tagged_offsets:
        if tag in {"O", "B-MENTION"} and active is not None:
            mentions.append(active)
            active = None
        if tag != "O":
            active = Mention(active.start if active is not None else start, end)
    if active is not None:
        mentions.append(active)
    return mentions


def merge_window_tags(offset_windows, tag_windows):
    """Prefer the prediction furthest from its window edges for each token."""
    by_offset = {}
    for offsets, tags in zip(offset_windows, tag_windows, strict=True):
        tokens = [(tuple(offset), tag) for offset, tag in zip(offsets, tags, strict=True)
                  if offset[0] < offset[1]]
        for index, (offset, tag) in enumerate(tokens):
            context = min(index, len(tokens) - index - 1)
            if offset not in by_offset or context > by_offset[offset][0]:
                by_offset[offset] = context, tag
    return [(offset, by_offset[offset][1]) for offset in sorted(by_offset)]


class TransformerNER:
    """Load a local Hugging Face token classifier and run it on CPU."""

    def __init__(self, model_dir: Path, threads: int = 2):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
        self.model.float().to("cpu").eval()
        self.labels = self.model.config.id2label
        if set(self.labels.values()) != {"O", "B-MENTION", "I-MENTION"}:
            raise ValueError("Expected a Phase 3 BIO token classifier")

    def predict(self, text: str) -> list[Mention]:
        if not text.strip():
            return []
        encoded = self.tokenizer(
            text, truncation=True, max_length=256, stride=64, padding=True,
            return_overflowing_tokens=True, return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").tolist()
        encoded.pop("overflow_to_sample_mapping")
        tag_windows = []
        with self.torch.inference_mode():
            for start in range(0, len(offsets), 8):
                batch = {key: value[start:start + 8] for key, value in encoded.items()}
                labels = self.model(**batch).logits.argmax(dim=-1).tolist()
                tag_windows.extend([[self.labels[tag] for tag in window] for window in labels])
        return decode_bio(merge_window_tags(offsets, tag_windows))


def load_predictor(dataset: Dataset | None = None, model_dir: Path = NER_MODEL_DIR):
    if model_dir.exists():
        return TransformerNER(model_dir)
    return TrainingDictionaryNER(dataset if dataset is not None else load_dataset())


def benchmark(dataset, predictor):
    ids = sorted(dataset.splits["validation"])
    gold = [row for row in dataset.annotations if row.note_id in dataset.splits["validation"]]
    baseline = ExactStringLookup(dataset.terminology)
    baseline_rows = [hit for note_id in ids
                     for hit in baseline.predict(note_id, dataset.notes[note_id])]
    baseline_score = evaluate(gold, baseline_rows).mention_f1

    predictor.predict(dataset.notes[ids[0]])  # Warm up before measuring per-note inference.
    elapsed, predicted = [], []
    for note_id in ids:
        start = perf_counter()
        mentions = predictor.predict(dataset.notes[note_id])
        elapsed.append((perf_counter() - start) * 1000)
        predicted.extend(Annotation(note_id, hit.start, hit.end, "") for hit in mentions)
    score = evaluate(gold, predicted).mention_f1
    p95 = quantiles(elapsed, n=20, method="inclusive")[18] if len(elapsed) > 1 else elapsed[0]
    print(f"{type(predictor).__name__}: {len(ids)} validation notes, {len(predicted)} predicted mentions")
    print(f"CPU inference per note: {mean(elapsed):.3f} ms mean, {p95:.3f} ms p95")
    print(f"Mention F1: {score:.6f} (Phase 2 baseline: {baseline_score:.6f})")
    return score > baseline_score


def main():
    parser = argparse.ArgumentParser(description="Benchmark Phase 3 mention detection on CPU.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fallback", action="store_true", help="Use the training dictionary explicitly.")
    mode.add_argument("--model-dir", type=Path, help="Require an exported model at this path.")
    args = parser.parse_args()
    dataset = load_dataset()
    if args.model_dir is not None:
        predictor = TransformerNER(args.model_dir)
    elif args.fallback:
        predictor = TrainingDictionaryNER(dataset)
    else:
        predictor = load_predictor(dataset)
    if not benchmark(dataset, predictor):
        raise SystemExit("Mention F1 did not beat the Phase 2 baseline.")


if __name__ == "__main__":
    main()

