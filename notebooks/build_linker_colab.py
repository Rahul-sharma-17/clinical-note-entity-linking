"""Build a local SapBERT linker index from the configured terminology on CUDA.

Run from the uploaded project root with NumPy, Transformers and CUDA PyTorch.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.config import LINKER_INDEX_DIR
from src.data_loader import load_dataset
from src.entity_linker import alias_fingerprint, alias_rows, fit_calibration, gold_mentions
from src.semantic_linker import DenseRetriever, SapBERTEncoder


def write_embeddings(path, rows, encoder, batch_size):
    from numpy.lib.format import open_memmap

    if not rows:
        raise ValueError("The terminology contains no aliases")
    dimension = encoder.model.config.hidden_size
    matrix = open_memmap(path, mode="w+", dtype="float16", shape=(len(rows), dimension))
    for start in range(0, len(rows), batch_size):
        terms = [term for _, term in rows[start:start + batch_size]]
        matrix[start:start + len(terms)] = encoder.encode(terms)
    matrix.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=LINKER_INDEX_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1 or args.output.exists():
        parser.error("Use a positive batch size and an output directory that does not exist.")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Enable a CUDA GPU in the Colab or Kaggle runtime.")
    dataset = load_dataset()
    rows = alias_rows(dataset.terminology)
    encoder = SapBERTEncoder(device="cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=args.output.parent) as temporary:
        stage = Path(temporary) / "linker_index"
        stage.mkdir()
        write_embeddings(stage / "embeddings.npy", rows, encoder, args.batch_size)
        print(f"Encoded {len(rows):,} terminology aliases. Fitting calibration on train spans.", flush=True)
        retriever = DenseRetriever(rows, stage / "embeddings.npy", encoder)
        calibration = fit_calibration(retriever, gold_mentions(dataset, "train"))
        encoder.save(stage / "encoder")
        manifest = {"version": 1, "aliases": alias_fingerprint(rows), "calibration": calibration.rates}
        (stage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        stage.rename(args.output)
    print(f"Exported the FP16 embedding matrix, encoder and calibration to {args.output}")


if __name__ == "__main__":
    main()

