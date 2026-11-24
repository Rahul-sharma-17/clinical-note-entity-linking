"""Import the challenge CSVs and an unpacked international RF2 snapshot."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from .config import DatasetPaths, ROOT
from .data_loader import (
    ANNOTATION_FIELDS, NOTE_FIELDS, SPLIT_FIELDS, TERMINOLOGY_FIELDS,
    load_dataset, load_splits,
)

ROOTS = {"404684003", "71388002", "123037004"}  # finding, procedure, body structure


CHALLENGE_FINDINGS = {
    "298430001": "Increased active range of cervical spine right lateral flexion",
    "298577009": "Observation of sensation of musculoskeletal structure of thoracic spine",
    "312087002": "Disorder following clinical procedure",
}


def rows(path, fields, delimiter=","):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter, strict=True)
        if not set(fields).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected columns {fields}")
        for row in reader:
            yield tuple(row[field] for field in fields)


def snapshot(root, pattern):
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} under {root}, found {len(matches)}")
    return matches[0]


def terminology(root):
    children = defaultdict(set)
    relations = snapshot(root, "sct2_Relationship_Snapshot_*.txt")
    for active, child, parent, kind in rows(relations, ("active", "sourceId", "destinationId", "typeId"), "\t"):
        if active == "1" and kind == "116680003":
            children[parent].add(child)
    scope = set()
    pending = list(ROOTS)
    while pending:
        code = pending.pop()
        if code not in scope:
            scope.add(code)
            pending.extend(children.get(code, ()))

    scope.update(CHALLENGE_FINDINGS)
    descriptions = snapshot(root, "sct2_Description_Snapshot-en_*.txt")
    names = {}
    aliases = defaultdict(set)
    for active, code, kind, term in rows(descriptions, ("active", "conceptId", "typeId", "term"), "\t"):
        if active != "1" or code not in scope:
            continue
        if kind == "900000000000003001":  # fully specified name
            names[code] = term
        elif kind == "900000000000013009":
            aliases[code].add(term)
    # Team SNOBERT explicitly includes these three annotated findings in the
    # challenge edition. Prefer the supplied release label when it is available.
    for code, fallback in CHALLENGE_FINDINGS.items():
        names.setdefault(code, fallback)
    for code in sorted(names):
        terms = sorted(aliases[code] - {names[code]})
        if any("|" in term for term in terms):
            raise ValueError(f"Alias for {code} contains the CSV alias delimiter")
        yield code, names[code], "|".join(terms)


def write_table(path, fields, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(records)


def convert(source, output):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        notes = list(rows(source / "mimic-iv_notes_training_set.csv", NOTE_FIELDS))
        splits = load_splits(None, {note_id for note_id, _ in notes})
        write_table(stage / "notes.csv", NOTE_FIELDS, notes)
        write_table(stage / "annotations.csv", ANNOTATION_FIELDS,
                    rows(source / "train_annotations.csv", ANNOTATION_FIELDS))
        write_table(stage / "terminology.csv", TERMINOLOGY_FIELDS, terminology(source))
        write_table(stage / "splits.csv", SPLIT_FIELDS,
                    ((note_id, split) for split, ids in splits.items() for note_id in sorted(ids)))
        dataset = load_dataset(DatasetPaths(*(stage / name for name in
            ("notes.csv", "annotations.csv", "terminology.csv", "splits.csv"))))
        if not dataset.annotations:
            raise ValueError("Challenge training annotations are empty")
        stage.rename(output)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory with notes, annotations and unpacked RF2 release")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "official" / "processed")
    args = parser.parse_args()
    dataset = convert(args.source, args.output)
    print(f"Imported {len(dataset.notes)} notes, {len(dataset.annotations)} links and "
          f"{len(dataset.terminology)} concepts into {args.output}")
