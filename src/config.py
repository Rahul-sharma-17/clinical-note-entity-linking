"""Dataset paths and deterministic split settings."""

from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DATASET = os.environ.get("CODER_REVIEW_DATASET", "public")
if DATASET not in {"public", "official"}:
    raise ValueError("CODER_REVIEW_DATASET must be public or official")
DATA_DIR = PROCESSED_DIR if DATASET == "public" else ROOT / "data" / "official" / "processed"
MODEL_DIR = ROOT / "models" if DATASET == "public" else ROOT / "models" / "official"
RESULTS_DIR = ROOT / "results" if DATASET == "public" else ROOT / "results" / "official"
NER_MODEL_DIR = MODEL_DIR / "ner_model"
LINKER_INDEX_DIR = MODEL_DIR / "linker_index"
BENCHMARK_METRICS_PATH = RESULTS_DIR / "benchmark_metrics.json"

NOTES_PATH = DATA_DIR / "notes.csv"
ANNOTATIONS_PATH = DATA_DIR / "annotations.csv"
TERMINOLOGY_PATH = DATA_DIR / "terminology.csv"
# Set to None for a deterministic split of a replacement dataset, or supply
# a CSV with note_id,split columns for predefined splits.
SPLITS_PATH: Path | None = DATA_DIR / "splits.csv"
VALIDATION_FRACTION = 0.2
SPLIT_SEED = 42


@dataclass(frozen=True)
class DatasetPaths:
    notes: Path = NOTES_PATH
    annotations: Path = ANNOTATIONS_PATH
    terminology: Path = TERMINOLOGY_PATH
    splits: Path | None = SPLITS_PATH

