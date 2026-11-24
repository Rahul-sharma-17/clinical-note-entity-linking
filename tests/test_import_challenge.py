import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.import_challenge import convert
from src.entity_linker import load_linker
from src.ner_predictor import TrainingDictionaryNER
from src.pipeline import Pipeline


class ChallengeImportTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = self.root / "processed"
        self.note = "Example\r\nrenal failure."
        with (self.root / "mimic-iv_notes_training_set.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("note_id", "text"))
            writer.writerows((identifier, self.note) for identifier in ("001", "002", "003"))
        (self.root / "train_annotations.csv").write_text(
            "note_id,start,end,concept_id\n001,9,22,100\n002,9,22,100\n003,9,22,100\n")
        (self.root / "sct2_Relationship_Snapshot_INT_test.txt").write_text(
            "active\tsourceId\tdestinationId\ttypeId\n1\t100\t404684003\t116680003\n"
            "0\t200\t404684003\t116680003\n")
        (self.root / "sct2_Description_Snapshot-en_INT_test.txt").write_text(
            "active\tconceptId\ttypeId\tterm\n1\t100\t900000000000003001\tRenal failure (disorder)\n"
            "1\t100\t900000000000013009\trenal failure\n1\t200\t900000000000003001\tOther\n"
            "0\t100\t900000000000013009\tOld alias\n")

    def test_import_and_predict(self):
        dataset = convert(self.root, self.output)
        self.assertEqual(dataset.notes["001"], self.note)
        self.assertIn("312087002", dataset.terminology)
        self.assertNotIn("200", dataset.terminology)
        self.assertEqual(dataset.terminology["100"].synonyms, ("renal failure",))
        self.assertFalse(dataset.splits["train"] & dataset.splits["validation"])
        model = Pipeline(TrainingDictionaryNER(dataset), load_linker(dataset, lexical=True))
        hit, = model.predict(dataset.notes[next(iter(dataset.splits["validation"]))])
        self.assertEqual((hit.start, hit.end, hit.text, hit.concept_id), (9, 22, "renal failure", "100"))
        with self.assertRaises(FileExistsError):
            convert(self.root, self.output)

    def test_invalid_annotation_leaves_no_partial_dataset(self):
        (self.root / "train_annotations.csv").write_text("note_id,start,end,concept_id\n001,9,99,100\n")
        with self.assertRaisesRegex(ValueError, "invalid offsets"):
            convert(self.root, self.output)
        self.assertFalse(self.output.exists())
