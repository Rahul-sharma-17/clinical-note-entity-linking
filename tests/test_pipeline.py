import unittest

from src.data_loader import Annotation, Concept, Dataset
from src.entity_linker import load_linker
from src.metrics import Scores, evaluate
from src.ner_predictor import TrainingDictionaryNER
from src.pipeline import Pipeline, predict


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = Dataset(
            notes={"train": "CHF and heart failure."},
            annotations=[Annotation("train", 0, 3, "100"), Annotation("train", 8, 21, "100")],
            terminology={"100": Concept("Heart failure", ("CHF",))},
            splits={"train": {"train"}, "validation": set()},
        )
        cls.pipeline = Pipeline(TrainingDictionaryNER(dataset), load_linker(dataset, lexical=True))

    def test_empty_note(self):
        self.assertEqual(predict(""), [])
        self.assertEqual(self.pipeline.predict(""), [])
        self.assertEqual(self.pipeline.predict(" \r\n"), [])

    def test_nonmedical_note(self):
        self.assertEqual(self.pipeline.predict("The meeting starts after lunch."), [])

    def test_thousand_word_note_preserves_late_spans_and_links(self):
        prefix = "today " * 997
        note = prefix + "CHF; heart failure."
        self.assertEqual(len(note.split()), 1000)
        hits = self.pipeline.predict(note)
        self.assertEqual(
            [(hit.start, hit.end, hit.text, hit.concept_id, hit.concept_name) for hit in hits],
            [(len(prefix), len(prefix) + 3, "CHF", "100", "Heart failure"),
             (len(prefix) + 5, len(prefix) + 18, "heart failure", "100", "Heart failure")],
        )
        for hit in hits:
            self.assertEqual(note[hit.start:hit.end], hit.text)
            self.assertIs(type(hit.confidence), float)
            self.assertTrue(0.0 <= hit.confidence <= 1.0)

    def test_unlinked_spans_count_for_detection_and_linking_denominator(self):
        hit = Annotation("note", 0, 3, "100")
        gold = [hit, Annotation("note", 8, 21, "100")]
        predicted = [hit, Annotation("note", 8, 21, "")]
        self.assertEqual(evaluate(gold, predicted), Scores(1.0, 0.5, 2 / 3))


if __name__ == "__main__":
    unittest.main()

