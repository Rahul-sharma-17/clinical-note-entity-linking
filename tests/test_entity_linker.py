import json
import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from notebooks.build_linker_colab import write_embeddings
from src.data_loader import Annotation, Concept, Dataset
from src.entity_linker import (
    EntityLinker, LexicalRetriever, alias_rows, evaluate_linking, fit_calibration,
    gold_mentions, load_linker,
)
from src.semantic_linker import DenseRetriever


def fixture():
    return Dataset(
        {"train": "Cardiac failure", "validation": "Tachycardia"},
        [Annotation("train", 0, 15, "10"), Annotation("validation", 0, 11, "30")],
        {
            "10": Concept("Heart failure", ("Cardiac failure", "HEART FAILURE")),
            "20": Concept("Renal failure", ()),
            "30": Concept("Tachycardia", ()),
        },
        {"train": {"train"}, "validation": {"validation"}},
    )


class EntityLinkerTests(unittest.TestCase):
    def test_synonyms_and_casefold_rank_exact_matches_first_and_deduplicate(self):
        dataset = fixture()
        rows = alias_rows(dataset.terminology)
        self.assertEqual(sum(row == ("10", "heart failure") for row in rows), 1)
        linker = load_linker(dataset, lexical=True)
        candidates = linker.link("CARDIAC  failure")
        self.assertEqual([item.concept_id for item in candidates], ["10", "20", "30"])
        self.assertEqual(candidates[0].concept_name, "Heart failure")
        self.assertEqual(linker.link("cardiac failure", top_k=1), candidates[:1])
        self.assertEqual(linker.link("  "), [])
        with self.assertRaises(ValueError):
            linker.link("heart failure", top_k=0)

    def test_confidence_comes_from_training_outcomes(self):
        dataset = fixture()
        retriever = LexicalRetriever(alias_rows(dataset.terminology))
        right = fit_calibration(retriever, gold_mentions(dataset, "train"))
        wrong = fit_calibration(retriever, [("Cardiac failure", {"20"})])
        self.assertEqual(list(gold_mentions(dataset, "train")), [("Cardiac failure", {"10"})])
        self.assertGreater(right.confidence(0, 1.0), wrong.confidence(0, 1.0))
        results = EntityLinker(dataset.terminology, retriever, right).link("Cardiac failure")
        for candidate in results:
            self.assertIs(type(candidate.confidence), float)
            self.assertGreaterEqual(candidate.confidence, 0.0)
            self.assertLessEqual(candidate.confidence, 1.0)

    def test_recall_scores_include_misses_and_abstentions(self):
        linker = load_linker(fixture(), lexical=True)
        report = evaluate_linking(linker, [
            ("Cardiac failure", {"10"}),
            ("Cardiac failure", {"20"}),
            ("zzzz", {"30"}),
        ])
        self.assertEqual(report.mentions, 3)
        self.assertEqual(report.recall_at_1, 1 / 3)
        self.assertEqual(report.recall_at_5, 2 / 3)
        self.assertEqual(report.coverage, 2 / 3)

    def test_semantic_index_rejects_stale_terminology_before_loading_weights(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text(
                json.dumps({"version": 1, "aliases": "old"}), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_linker(fixture(), path)


@unittest.skipUnless(find_spec("numpy"), "NumPy is required for the optional semantic index")
class DenseIndexTests(unittest.TestCase):
    def test_fp16_export_and_cpu_retrieval(self):
        import numpy as np

        vectors = {
            "cardiac failure": [1.0, 0.0],
            "heart failure": [1.0, 0.0],
            "renal failure": [0.0, 1.0],
            "tachycardia": [-1.0, 0.0],
        }

        class Encoder:
            model = SimpleNamespace(config=SimpleNamespace(hidden_size=2))

            def encode(self, texts):
                return np.asarray([vectors[text] for text in texts], dtype=np.float32)

        dataset = fixture()
        rows = alias_rows(dataset.terminology)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.npy"
            encoder = Encoder()
            write_embeddings(path, rows, encoder, batch_size=3)
            self.assertEqual(np.load(path, allow_pickle=False).dtype, np.float16)
            retriever = DenseRetriever(rows, path, encoder)
            self.assertEqual(retriever.rank("cardiac failure", 3),
                             (("10", 1.0), ("20", 0.0), ("30", -1.0)))


if __name__ == "__main__":
    unittest.main()

