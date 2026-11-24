import re
import unittest

from notebooks.train_ner_colab import aligned_labels, training_features
from src.data_loader import Annotation, Dataset
from src.ner_predictor import Mention, TrainingDictionaryNER, decode_bio, merge_window_tags


class MentionDetectionTests(unittest.TestCase):
    def test_training_dictionary_handles_punctuation_and_excludes_validation_labels(self):
        notes = {"train": "CHF; type 2 diabetes.", "validation": "Asthma."}
        dataset = Dataset(
            notes, [Annotation("train", 0, 3, "1"), Annotation("train", 5, 20, "2"),
                    Annotation("validation", 0, 6, "3")],
            {}, {"train": {"train"}, "validation": {"validation"}},
        )
        predictor = TrainingDictionaryNER(dataset)
        text = "(CHF), type 2 diabetes; xCHF, Asthma."
        hits = predictor.predict(text)
        self.assertEqual(hits, [Mention(1, 4), Mention(7, 22)])
        self.assertEqual([text[hit.start:hit.end] for hit in hits], ["CHF", "type 2 diabetes"])

    def test_exact_training_alignment_and_decoding(self):
        text = "(CHF), type 2 diabetes."
        offsets = [(0, 0)] + [match.span() for match in re.finditer(r"\w+|[^\w\s]", text)]
        spans = [(1, 4), (7, 22), (8, 10)]
        labels = aligned_labels(offsets, spans)
        self.assertNotIn((0, 0), labels)
        self.assertEqual(labels[(7, 11)], 1)
        self.assertEqual(labels[(12, 13)], 2)
        self.assertEqual(labels[(14, 22)], 2)
        tags = {0: "O", 1: "B-MENTION", 2: "I-MENTION"}
        self.assertEqual(
            decode_bio([(offset, tags[label]) for offset, label in labels.items()]),
            [Mention(1, 4), Mention(7, 22)],
        )
        self.assertEqual(aligned_labels([(0, 5)], [(1, 4)]), {(0, 5): -100})

    def test_window_overlap_merges_into_one_mention(self):
        offsets = [
            [(0, 0), (0, 4), (5, 10), (11, 12), (13, 15), (0, 0)],
            [(0, 0), (11, 12), (13, 15), (16, 18), (19, 20), (0, 0)],
        ]
        tags = [
            ["O", "O", "B-MENTION", "I-MENTION", "O", "O"],
            ["O", "O", "I-MENTION", "I-MENTION", "O", "O"],
        ]
        self.assertEqual(decode_bio(merge_window_tags(offsets, tags)), [Mention(5, 18)])

    def test_feature_builder_uses_train_notes_and_keeps_overlap_labels_consistent(self):
        dataset = Dataset(
            {"a": "CHF", "b": "Other"}, [Annotation("a", 0, 3, "1")],
            {}, {"train": {"a"}, "validation": {"b"}},
        )

        def tokenizer(text, **kwargs):
            self.assertEqual(text, "CHF")
            return {
                "input_ids": [[101, 1, 102], [101, 1, 102]],
                "offset_mapping": [[(0, 0), (0, 3), (0, 0)]] * 2,
                "overflow_to_sample_mapping": [0, 0],
            }

        features = list(training_features(tokenizer, dataset))
        self.assertEqual([row["labels"] for row in features], [[-100, 1, -100]] * 2)
        self.assertEqual(set(features[0]), {"input_ids", "labels"})


if __name__ == "__main__":
    unittest.main()

