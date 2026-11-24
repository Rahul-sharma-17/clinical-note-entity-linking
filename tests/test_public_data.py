"""Phase 1 checks against the downloaded source and generated CSVs."""

import csv
import re
import unittest
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from src.config import DatasetPaths, PROCESSED_DIR, RAW_DIR
from src.data_loader import load_dataset, load_splits
from src.fetch_public_data import (
    BC5CDR_SHA256, MONDO_SHA256, digest, load_mondo,
)


class PublicDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = DatasetPaths(
            PROCESSED_DIR / "notes.csv", PROCESSED_DIR / "annotations.csv",
            PROCESSED_DIR / "terminology.csv", PROCESSED_DIR / "splits.csv",
        )
        cls.dataset = load_dataset(cls.paths)

    def test_downloads_and_csv_schema(self):
        self.assertEqual(digest(RAW_DIR / "CDR_Data.zip"), BC5CDR_SHA256)
        self.assertEqual(digest(RAW_DIR / "mondo-base.obo"), MONDO_SHA256)
        for path, expected, size in [
            (self.paths.notes, ["note_id", "text"], 1500),
            (self.paths.annotations, ["note_id", "start", "end", "concept_id"], 7592),
            (self.paths.terminology, ["concept_id", "concept_name", "synonyms"], 8783),
        ]:
            with self.subTest(file=path.name), path.open(encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream)
                self.assertEqual(next(reader), expected)
                self.assertEqual(sum(1 for _ in reader), size)
        for code in self.dataset.terminology:
            self.assertRegex(code, r"^[0-9]+$")

    def test_every_output_span_matches_the_original_pubtator_surface(self):
        surfaces = {}
        corpus_splits = {}
        with ZipFile(RAW_DIR / "CDR_Data.zip") as archive:
            for name, source in [
                ("train", "Training"), ("validation", "Development"), ("test", "Test"),
            ]:
                member = f"CDR_Data/CDR.Corpus.v010516/CDR_{source}Set.PubTator.txt"
                raw = archive.read(member).decode("utf-8")
                corpus_splits[name] = set(re.findall(r"^(\d+)\|t\|", raw, re.MULTILINE))
                for row in raw.split("\n"):
                    fields = row.rstrip("\r").split("\t")
                    if len(fields) >= 6 and fields[4] == "Disease":
                        note_id, start, end, surface = fields[:4]
                        surfaces[note_id, int(start), int(end)] = surface

        self.assertEqual(self.dataset.splits, corpus_splits)
        for annotation in self.dataset.annotations:
            note_id, start, end = annotation.note_id, annotation.start, annotation.end
            mention = self.dataset.notes[note_id][start:end]
            self.assertTrue(mention.strip(), annotation)
            self.assertEqual(mention, surfaces.get((note_id, start, end)), annotation)
            self.assertIn(annotation.concept_id, self.dataset.terminology)

    def test_splits_are_disjoint_and_cover_all_notes(self):
        splits = self.dataset.splits
        self.assertEqual({name: len(ids) for name, ids in splits.items()},
                         {"train": 500, "validation": 500, "test": 500})
        for left, right in combinations(splits.values(), 2):
            self.assertFalse(left & right)
        self.assertEqual(set.union(*splits.values()), set(self.dataset.notes))


class LoaderTests(unittest.TestCase):
    def test_replacement_files_can_use_deterministic_splits(self):
        ids = {"001", "02", "3", "4", "5"}
        first = load_splits(None, ids)
        self.assertEqual(first, load_splits(None, set(reversed(sorted(ids)))))
        self.assertEqual(len(first["validation"]), 1)
        self.assertFalse(first["train"] & first["validation"])
        self.assertEqual(first["train"] | first["validation"], ids)

    def test_mondo_crosswalk_excludes_ambiguous_and_non_exact_mappings(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.obo"
            path.write_text(
                '[Term]\nname: Example disorder\nsynonym: "An alias" EXACT []\n'
                'property_value: skos:exactMatch http://identifiers.org/mesh/D01\n'
                'property_value: skos:exactMatch http://identifiers.org/mesh/D02\n'
                'property_value: skos:exactMatch http://identifiers.org/snomedct/123\n'
                'property_value: skos:closeMatch http://identifiers.org/mesh/D03\n'
                '[Term]\nname: Another disorder\n'
                'property_value: skos:exactMatch http://identifiers.org/mesh/D02\n'
                'property_value: skos:exactMatch http://identifiers.org/snomedct/456\n',
                encoding="utf-8",
            )
            mapping, terminology = load_mondo(path)
        self.assertEqual(mapping, {"D01": "123"})
        self.assertIn(("123", "Example disorder", "An alias"), terminology)


if __name__ == "__main__":
    unittest.main()

