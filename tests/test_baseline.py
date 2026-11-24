import unittest

from src.baseline import ExactStringLookup
from src.data_loader import Annotation, Concept
from src.metrics import Scores, evaluate


class EvaluationTests(unittest.TestCase):
    def test_exact_span_wrong_concept(self):
        gold = [Annotation("n", 0, 4, "10")]
        predicted = [Annotation("n", 0, 4, "20")]
        self.assertEqual(evaluate(gold, predicted), Scores(1.0, 0.0, 0.0))

    def test_partial_overlap(self):
        gold = [Annotation("n", 0, 4, "10")]
        predicted = [Annotation("n", 0, 3, "10")]
        self.assertEqual(evaluate(gold, predicted), Scores(0.0, 0.0, 0.0))

    def test_missed_entity(self):
        hit = Annotation("n", 0, 4, "10")
        missed = Annotation("n", 5, 10, "20")
        self.assertEqual(evaluate([hit, missed], [hit]), Scores(2 / 3, 1.0, 2 / 3))

    def test_extra_false_positive(self):
        hit = Annotation("n", 0, 4, "10")
        extra = Annotation("other-note", 0, 4, "10")
        self.assertEqual(evaluate([hit], [hit, extra]), Scores(2 / 3, 1.0, 2 / 3))

    def test_duplicate_and_alternative_gold_ids(self):
        hit = Annotation("n", 0, 4, "10")
        other_id = Annotation("n", 0, 4, "20")
        self.assertEqual(evaluate([hit, hit, other_id], [hit, hit]), Scores(1.0, 1.0, 2 / 3))

    def test_empty_inputs(self):
        self.assertEqual(evaluate([], []), Scores(0.0, 0.0, 0.0))


class BaselineTests(unittest.TestCase):
    def test_literal_offsets_boundaries_overlaps_and_synonyms(self):
        baseline = ExactStringLookup({
            "10": Concept("failure", ()),
            "20": Concept("heart failure", ("CHF",)),
            "30": Concept("type 2 diabetes", ()),
        })
        text = "HEART FAILURE, CHF; xCHF CHF2. Type  2 diabetes."
        results = baseline.predict("n", text)
        self.assertEqual(
            [(hit.note_id, hit.start, hit.end, text[hit.start:hit.end], hit.concept_id, hit.confidence)
             for hit in results],
            [("n", 0, 13, "HEART FAILURE", "20", 1.0), ("n", 15, 18, "CHF", "20", 1.0)],
        )

    def test_ambiguity_prefers_name_then_lexical_id(self):
        baseline = ExactStringLookup({
            "01": Concept("Hemorrhage", ("Bleeding",)),
            "20": Concept("Bleeding", ()),
            "10": Concept("bleeding", ()),
        })
        self.assertEqual(baseline.predict("n", "BLEEDING")[0].concept_id, "10")

    def test_unicode_casefold_preserves_original_offsets(self):
        baseline = ExactStringLookup({"10": Concept("STRASSE", ())})
        result, = baseline.predict("n", "(Straße)")
        self.assertEqual((result.start, result.end), (1, 7))
        self.assertEqual(baseline.predict("n", ""), [])


if __name__ == "__main__":
    unittest.main()

