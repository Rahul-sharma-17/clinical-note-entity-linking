"""Read and validate the common CSV interface used by downstream stages."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from .config import DatasetPaths, SPLIT_SEED, VALIDATION_FRACTION

NOTE_FIELDS = ("note_id", "text")
ANNOTATION_FIELDS = ("note_id", "start", "end", "concept_id")
TERMINOLOGY_FIELDS = ("concept_id", "concept_name", "synonyms")
SPLIT_FIELDS = ("note_id", "split")
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class Annotation:
    note_id: str
    start: int
    end: int
    concept_id: str


@dataclass(frozen=True)
class Concept:
    name: str
    synonyms: tuple[str, ...]


@dataclass
class Dataset:
    notes: dict[str, str]
    annotations: list[Annotation]
    terminology: dict[str, Concept]
    splits: dict[str, set[str]]


def read_rows(path: Path, fields: tuple[str, ...]):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if tuple(next(reader, [])) != fields:
            raise ValueError(f"{path}: expected header {fields}")
        for row in reader:
            if len(row) != len(fields):
                raise ValueError(f"{path}:{reader.line_num}: invalid column count")
            yield row


def load_splits(path: Path | None, note_ids: set[str]) -> dict[str, set[str]]:
    splits = {name: set() for name in SPLIT_NAMES}
    if len(note_ids) < 2:
        raise ValueError("A train/validation split requires at least two notes")
    if path is None:
        ordered_ids = sorted(note_ids)
        random.Random(SPLIT_SEED).shuffle(ordered_ids)
        size = max(1, min(len(note_ids) - 1, round(len(note_ids) * VALIDATION_FRACTION)))
        splits["validation"].update(ordered_ids[:size])
        splits["train"].update(ordered_ids[size:])
        return splits

    assigned = set()
    for note_id, split in read_rows(path, SPLIT_FIELDS):
        if note_id not in note_ids or note_id in assigned or split not in splits:
            raise ValueError(f"{path}: invalid or duplicate split assignment for {note_id!r}")
        splits[split].add(note_id)
        assigned.add(note_id)
    if assigned != note_ids or not splits["train"] or not splits["validation"]:
        raise ValueError(f"{path}: expected complete coverage and nonempty train and validation splits")
    return splits


def load_dataset(paths: DatasetPaths = DatasetPaths()) -> Dataset:
    notes = {}
    for note_id, text in read_rows(paths.notes, NOTE_FIELDS):
        if not note_id.strip() or note_id in notes:
            raise ValueError(f"{paths.notes}: invalid or duplicate note ID {note_id!r}")
        notes[note_id] = text

    terminology = {}
    for concept_id, name, synonyms in read_rows(paths.terminology, TERMINOLOGY_FIELDS):
        if not concept_id.strip() or concept_id in terminology or not name.strip():
            raise ValueError(f"{paths.terminology}: invalid or duplicate concept {concept_id!r}")
        aliases = tuple(value.strip() for value in synonyms.split("|") if value.strip())
        terminology[concept_id] = Concept(name, aliases)

    annotations = []
    for note_id, start, end, concept_id in read_rows(paths.annotations, ANNOTATION_FIELDS):
        span = Annotation(note_id, int(start), int(end), concept_id)
        text = notes.get(note_id)
        if text is None or not (0 <= span.start < span.end <= len(text)):
            raise ValueError(f"{paths.annotations}: unknown note or invalid offsets in {span}")
        if not text[span.start:span.end].strip() or concept_id not in terminology:
            raise ValueError(f"{paths.annotations}: blank mention or unknown concept in {span}")
        annotations.append(span)

    return Dataset(notes, annotations, terminology, load_splits(paths.splits, set(notes)))

