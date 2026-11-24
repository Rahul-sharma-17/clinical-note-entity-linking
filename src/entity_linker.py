"""Retrieve and score unique SNOMED CT candidates for a known mention span."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from heapq import nsmallest
from pathlib import Path

from .config import LINKER_INDEX_DIR
from .data_loader import Dataset, load_dataset

CALIBRATION_K = 5


@dataclass(frozen=True)
class Candidate:
    concept_id: str
    concept_name: str
    confidence: float


def _normalize(text):
    return " ".join(text.casefold().split())


def alias_rows(terminology):
    return [
        (code, alias)
        for code, concept in sorted(terminology.items())
        for alias in sorted({_normalize(term) for term in (concept.name, *concept.synonyms)} - {""})
    ]


def alias_fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("utf-8")).hexdigest()


def _trigrams(text):
    padded = f"  {text}  "
    return {padded[index:index + 3] for index in range(len(padded) - 2)}


class LexicalRetriever:
    """Rank by maximum character trigram Jaccard similarity over aliases."""

    def __init__(self, rows):
        self.rows = rows
        self.postings = defaultdict(list)
        self.sizes = []
        for index, (_, text) in enumerate(rows):
            grams = _trigrams(text)
            self.sizes.append(len(grams))
            for gram in grams:
                self.postings[gram].append(index)
        self._rank = lru_cache(maxsize=4096)(self._rank_uncached)

    def rank(self, text, k):
        return self._rank(text)[:k]

    def _rank_uncached(self, text):
        grams = _trigrams(text)
        overlap = Counter(index for gram in grams for index in self.postings.get(gram, ()))
        scores = {}
        for index, shared in overlap.items():
            code, alias = self.rows[index]
            # Reserve perfect scores for literal matches, including rare trigram collisions.
            score = 1.0 if alias == text else 0.99 * shared / (len(grams) + self.sizes[index] - shared)
            if score > scores.get(code, 0.0):
                scores[code] = score
        return tuple(nsmallest(CALIBRATION_K, scores.items(), key=lambda item: (-item[1], item[0])))


def _score_bin(score):
    return min(9, max(0, int(score * 10)))


@dataclass(frozen=True)
class Calibration:
    """Estimate correctness by rank and score band with rank-level smoothing."""
    rates: list[list[float]]

    def __post_init__(self):
        if len(self.rates) != CALIBRATION_K or any(
            len(bands) != 10 or any(type(score) is not float or not 0.0 <= score <= 1.0 for score in bands)
            for bands in self.rates
        ):
            raise ValueError("Invalid calibration table")

    def confidence(self, rank, score):
        return self.rates[rank][_score_bin(score)]


def gold_mentions(dataset, split):
    """Yield each exact gold span with its accepted IDs, once per note."""
    ids = dataset.splits[split]
    grouped = defaultdict(set)
    for row in dataset.annotations:
        if row.note_id in ids:
            grouped[row.note_id, row.start, row.end].add(row.concept_id)
    for (note_id, start, end), codes in grouped.items():
        yield dataset.notes[note_id][start:end], codes


def fit_calibration(retriever, examples):
    counts = [Counter() for _ in range(CALIBRATION_K)]
    hits = [Counter() for _ in range(CALIBRATION_K)]
    for mention, codes in examples:
        for rank, (code, score) in enumerate(retriever.rank(_normalize(mention), CALIBRATION_K)):
            band = _score_bin(score)
            counts[rank][band] += 1
            hits[rank][band] += code in codes
    if not counts[0]:
        raise ValueError("Confidence calibration requires retrievable training mentions")
    rates = []
    for observed, correct in zip(counts, hits, strict=True):
        prior = (sum(correct.values()) + 1) / (sum(observed.values()) + 2)
        rates.append([(correct[band] + 5 * prior) / (observed[band] + 5) for band in range(10)])
    return Calibration(rates)


class EntityLinker:
    def __init__(self, terminology, retriever, calibration):
        self.terminology = terminology
        self.retriever = retriever
        self.calibration = calibration

    def link(self, mention: str, top_k: int = 5) -> list[Candidate]:
        if not 1 <= top_k <= CALIBRATION_K:
            raise ValueError(f"top_k must be between 1 and {CALIBRATION_K}")
        text = _normalize(mention)
        if not text:
            return []
        return [
            Candidate(code, self.terminology[code].name, self.calibration.confidence(rank, score))
            for rank, (code, score) in enumerate(self.retriever.rank(text, top_k))
        ]


def load_linker(dataset: Dataset | None = None, index_dir: Path = LINKER_INDEX_DIR,
                *, lexical: bool = False) -> EntityLinker:
    dataset = dataset if dataset is not None else load_dataset()
    rows = alias_rows(dataset.terminology)
    if lexical or not index_dir.exists():
        retriever = LexicalRetriever(rows)
        calibration = fit_calibration(retriever, gold_mentions(dataset, "train"))
    else:
        from .semantic_linker import DenseRetriever, SapBERTEncoder

        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["version"] != 1 or manifest["aliases"] != alias_fingerprint(rows):
            raise ValueError("Linker index does not match the configured terminology. Rebuild the index.")
        encoder = SapBERTEncoder(index_dir / "encoder", device="cpu", local_only=True)
        retriever = DenseRetriever(rows, index_dir / "embeddings.npy", encoder)
        calibration = Calibration(manifest["calibration"])
    return EntityLinker(dataset.terminology, retriever, calibration)


@dataclass(frozen=True)
class LinkingReport:
    mentions: int
    recall_at_1: float
    recall_at_5: float
    coverage: float
    top_candidate_brier: float


def evaluate_linking(linker, examples):
    total = hits_one = hits_five = covered = 0
    squared_error = 0.0
    for surface, gold_ids in examples:
        total += 1
        candidates = linker.link(surface)
        for candidate in candidates:
            if type(candidate.confidence) is not float or not 0.0 <= candidate.confidence <= 1.0:
                raise ValueError(f"Invalid confidence on {candidate}")
        if candidates:
            covered += 1
            correct = candidates[0].concept_id in gold_ids
            hits_one += correct
            squared_error += (candidates[0].confidence - correct) ** 2
            hits_five += any(item.concept_id in gold_ids for item in candidates)
    if total == 0:
        raise ValueError("This evaluation split contains no gold mentions")
    return LinkingReport(total, hits_one / total, hits_five / total, covered / total,
                         squared_error / covered if covered else 0.0)


def peak_ram_mib():
    """Return OS-reported process peak working set on Windows or peak RSS on Unix."""
    import sys

    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                    "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile",
                )
            ]

        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = (wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD)
        query.restype = wintypes.BOOL
        current_process = ctypes.WinDLL("kernel32").GetCurrentProcess
        current_process.restype = wintypes.HANDLE
        counters = MemoryCounters()
        counters.size = ctypes.sizeof(counters)
        if not query(current_process(), ctypes.byref(counters), counters.size):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.peak_working_set / 1024 ** 2

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 2 if sys.platform == "darwin" else 1024)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CPU linking on validation gold spans.")
    parser.add_argument("--lexical", action="store_true", help="Force the local lexical retriever.")
    parser.add_argument("--index-dir", type=Path, help="Require the exported semantic index at this path.")
    args = parser.parse_args()
    if args.index_dir is not None and (args.lexical or not args.index_dir.is_dir()):
        parser.error("--index-dir must exist and cannot be combined with --lexical")
    dataset = load_dataset()
    linker = load_linker(dataset, args.index_dir or LINKER_INDEX_DIR, lexical=args.lexical)
    report = evaluate_linking(linker, gold_mentions(dataset, "validation"))
    print(f"{type(linker.retriever).__name__}: {report.mentions:,} validation gold spans")
    print(f"Recall@1 / standalone Linking Accuracy: {report.recall_at_1:.6f}")
    print(f"Recall@5: {report.recall_at_5:.6f}")
    print(f"Candidate coverage: {report.coverage:.2%}")
    if report.coverage:
        print(f"Top candidate Brier score (returned candidates): {report.top_candidate_brier:.6f}")
    print(f"Peak process RAM: {peak_ram_mib():.1f} MiB")


if __name__ == "__main__":
    main()

