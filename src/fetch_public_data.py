"""Build public BC5CDR/MONDO CSVs with Python's standard library."""

import csv
import hashlib
import io
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

from .config import PROCESSED_DIR, RAW_DIR, DatasetPaths
from .data_loader import (
    ANNOTATION_FIELDS, NOTE_FIELDS, SPLIT_FIELDS, TERMINOLOGY_FIELDS, load_dataset,
)

BC5CDR_URL = (
    "https://huggingface.co/datasets/bigbio/bc5cdr/resolve/"
    "6ba1463320d003e8232dfa0c3ee9aaa2559998ec/CDR_Data.zip"
)
MONDO_URL = (
    "https://github.com/monarch-initiative/mondo/releases/download/"
    "v2026-11-03/mondo-base.obo"
)
BC5CDR_SHA256 = "0a359a7f038d283a7b05b084fa73de014e7410e1f5d7034bf3fd01f016fc2444"
MONDO_SHA256 = "0c4365c5641a35374d581c2fae06e6844f732e7a6e65463d8af335eea1f0a0c8"
ARCHIVE_DIR = "CDR_Data/CDR.Corpus.v010516"
SOURCE_SPLITS = {
    "train": "TrainingSet",
    "validation": "DevelopmentSet",
    "test": "TestSet",
}
EXACT_MATCH = re.compile(
    r"property_value: skos:exactMatch https?://identifiers\.org/(mesh|snomedct)/(\w+)$"
)
EXACT_SYNONYM = re.compile(r'^synonym: "((?:\\.|[^"\\])*)" EXACT\b')


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, target: Path, expected_hash: str) -> Path:
    if target.is_file() and digest(target) == expected_hash:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if digest(temporary) != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {url}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def term_blocks(lines):
    block = None
    for raw_line in lines:
        line = raw_line.rstrip()
        if line.startswith("["):
            if block:
                yield block
            block = [] if line == "[Term]" else None
        elif line and block is not None:
            block.append(line)
    if block:
        yield block


def load_mondo(path: Path):
    targets_by_mesh = defaultdict(set)
    labels_by_target = defaultdict(set)
    synonyms_by_target = defaultdict(set)
    with path.open(encoding="utf-8") as stream:
        for block in term_blocks(stream):
            if "is_obsolete: true" in block:
                continue
            name = next((line[6:] for line in block if line.startswith("name: ")), "")
            if not name:
                continue
            mesh_ids, snomed_ids, aliases = set(), set(), set()
            for line in block:
                if match := EXACT_MATCH.fullmatch(line):
                    vocabulary, code = match.groups()
                    if vocabulary == "snomedct" and code.isascii() and code.isdecimal():
                        snomed_ids.add(code)
                    elif vocabulary == "mesh":
                        mesh_ids.add(code)
                elif match := EXACT_SYNONYM.match(line):
                    alias = re.sub(r'\\(["\\])', r'\1', match[1])
                    if alias and "|" not in alias:
                        aliases.add(alias)
            for mesh_id in mesh_ids:
                targets_by_mesh[mesh_id].update(snomed_ids)
            if len(snomed_ids) == 1:
                target = next(iter(snomed_ids))
                labels_by_target[target].add(name)
                synonyms_by_target[target].update(aliases)

    mapping = {mesh: next(iter(codes)) for mesh, codes in targets_by_mesh.items()
               if len(codes) == 1}
    # Include the whole unambiguous public subset, independent of corpus labels.
    terminology = []
    for code, labels in sorted(labels_by_target.items()):
        name = min(labels)
        synonyms = (labels | synonyms_by_target[code]) - {name}
        terminology.append((code, name, "|".join(sorted(alias for alias in synonyms if "|" not in alias))))
    if not mapping or not terminology:
        raise ValueError(f"{path}: no exact MeSH/SNOMED CT mappings found")
    return mapping, terminology


def parse_pubtator(lines):
    title = abstract = None
    note_id = None
    mentions = []
    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        title_match = re.match(r"^(\d+)\|t\|(.*)$", line)
        abstract_match = re.match(r"^(\d+)\|a\|(.*)$", line)
        if title_match:
            if note_id is not None:
                if abstract is None:
                    raise ValueError(f"{note_id}: missing abstract")
                yield note_id, title + " " + abstract, mentions
            note_id, title = title_match.groups()
            abstract, mentions = None, []
        elif abstract_match:
            if abstract_match[1] != note_id or abstract is not None:
                raise ValueError(f"Unexpected abstract for {abstract_match[1]}")
            abstract = abstract_match[2]
        elif line:
            fields = line.split("\t")
            if fields[0] != note_id or abstract is None:
                raise ValueError(f"Unexpected PubTator annotation: {line!r}")
            if len(fields) == 4 and fields[1] == "CID":
                continue
            # Some composite mentions have a seventh column of component names.
            if len(fields) not in (6, 7):
                raise ValueError(f"Unexpected PubTator annotation: {line!r}")
            _, start, end, surface, kind, mesh = fields[:6]
            mentions.append((int(start), int(end), surface, kind, mesh))
    if note_id is not None:
        if abstract is None:
            raise ValueError(f"{note_id}: missing abstract")
        yield note_id, title + " " + abstract, mentions


def convert(corpus_path: Path, ontology_path: Path, output_dir: Path):
    mapping, terminology = load_mondo(ontology_path)
    notes, annotations, splits = [], [], []
    counts = {split: Counter() for split in SOURCE_SPLITS}
    with ZipFile(corpus_path) as archive:
        for split, source in SOURCE_SPLITS.items():
            member = f"{ARCHIVE_DIR}/CDR_{source}.PubTator.txt"
            with archive.open(member) as binary, io.TextIOWrapper(binary, encoding="utf-8") as stream:
                for note_id, text, mentions in parse_pubtator(stream):
                    notes.append((note_id, text))
                    splits.append((note_id, split))
                    counts[split]["notes"] += 1
                    for start, end, surface, kind, mesh in mentions:
                        if not (0 <= start < end <= len(text)) or not surface.strip() or text[start:end] != surface:
                            raise ValueError(f"{note_id}: offsets {start}:{end} do not match {surface!r}")
                        if kind != "Disease":
                            counts[split]["other_mentions"] += 1
                            continue
                        counts[split]["disease_mentions"] += 1
                        targets = {mapping.get(code) for code in mesh.split("|")}
                        if None in targets or len(targets) != 1:
                            continue
                        annotations.append((note_id, start, end, targets.pop()))
                        counts[split]["linked_mentions"] += 1
    annotations = list(dict.fromkeys(annotations))
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = (
        ("notes.csv", NOTE_FIELDS, notes),
        ("annotations.csv", ANNOTATION_FIELDS, annotations),
        ("terminology.csv", TERMINOLOGY_FIELDS, terminology),
        ("splits.csv", SPLIT_FIELDS, splits),
    )
    with TemporaryDirectory(dir=output_dir) as staging:
        staged = Path(staging)
        for name, header, rows in tables:
            with (staged / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)
        load_dataset(DatasetPaths(*(staged / name for name, _, _ in tables)))
        for name, _, _ in tables:
            (staged / name).replace(output_dir / name)
    return counts, len(annotations), len(terminology)


def main():
    corpus = download(BC5CDR_URL, RAW_DIR / "CDR_Data.zip", BC5CDR_SHA256)
    ontology = download(MONDO_URL, RAW_DIR / "mondo-base.obo", MONDO_SHA256)
    counts, annotations, concepts = convert(corpus, ontology, PROCESSED_DIR)
    print(f"Wrote {annotations:,} unique annotations and {concepts:,} concepts to {PROCESSED_DIR}")
    for split, count in counts.items():
        print(
            f"{split}: {count['notes']:,} notes, "
            f"{count['linked_mentions']:,}/{count['disease_mentions']:,} disease mentions mapped, "
            f"{count['other_mentions']:,} other mentions excluded"
        )


if __name__ == "__main__":
    main()

