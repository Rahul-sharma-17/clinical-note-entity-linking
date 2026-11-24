"""Exact string lookup over terminology labels and synonyms."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from .data_loader import Annotation, Concept, load_dataset
from .metrics import evaluate


@dataclass(frozen=True)
class Prediction(Annotation):
    confidence: float


@dataclass
class _TrieNode:
    children: dict[str, "_TrieNode"] = field(default_factory=dict)
    concept_id: str | None = None
    ends_in_word: bool = False


def _word_character(char: str) -> bool:
    return char.isalnum() or char == "_"


class ExactStringLookup:
    """Match case-insensitive literals with word boundaries and original offsets."""

    def __init__(self, terminology: Mapping[str, Concept]):
        self._root = _TrieNode()
        # Preferred names precede synonyms. Within either group, IDs sort lexically.
        ordered = sorted(terminology.items())
        for concept_id, concept in ordered:
            self._add(concept.name, concept_id)
        for concept_id, concept in ordered:
            for synonym in concept.synonyms:
                self._add(synonym, concept_id)

    def _add(self, term: str, concept_id: str) -> None:
        term = term.strip()
        if not term:
            return
        node = self._root
        for char in term.casefold():
            if char not in node.children:
                node.children[char] = _TrieNode()
            node = node.children[char]
        if node.concept_id is None:
            node.concept_id = concept_id
            node.ends_in_word = _word_character(term[-1])

    def _match(self, text: str, start: int) -> tuple[int, str] | None:
        if start and _word_character(text[start - 1]) and _word_character(text[start]):
            return None
        node = self._root
        longest = None
        for index in range(start, len(text)):
            for folded in text[index].casefold():
                node = node.children.get(folded)
                if node is None:
                    return longest
            end = index + 1
            if node.concept_id is not None and (
                not node.ends_in_word or end == len(text) or not _word_character(text[end])
            ):
                longest = end, node.concept_id
        return longest

    def predict(self, note_id: str, text: str) -> list[Prediction]:
        """Return leftmost, longest non-overlapping matches at confidence 1.0."""
        predictions = []
        start = 0
        while start < len(text):
            match = self._match(text, start)
            if match is None:
                start += 1
                continue
            end, concept_id = match
            predictions.append(Prediction(note_id, start, end, concept_id, 1.0))
            start = end
        return predictions


def main() -> None:
    dataset = load_dataset()
    note_ids = dataset.splits["validation"]
    baseline = ExactStringLookup(dataset.terminology)
    gold = [row for row in dataset.annotations if row.note_id in note_ids]
    predictions = [
        prediction
        for note_id in sorted(note_ids)
        for prediction in baseline.predict(note_id, dataset.notes[note_id])
    ]
    scores = evaluate(gold, predictions)
    print(f"Validation: {len(note_ids):,} notes, {len(gold):,} gold links, "
          f"{len(predictions):,} predictions")
    print(f"Mention F1:       {scores.mention_f1:.6f}")
    print(f"Linking Accuracy: {scores.linking_accuracy:.6f}")
    print(f"End-to-End F1:   {scores.end_to_end_f1:.6f}")


if __name__ == "__main__":
    main()

