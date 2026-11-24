"""Fine-tune DistilBERT on a Colab or Kaggle CUDA runtime.

From the uploaded project root:
    python -m pip install "transformers>=4.48,<5" safetensors
    python -m notebooks.train_ner_colab

The runtime must have CUDA-enabled PyTorch 2.4 or newer.
"""

import argparse
from bisect import bisect_left
from pathlib import Path

from src.config import NER_MODEL_DIR
from src.data_loader import DatasetPaths, load_dataset

LABELS = {0: "O", 1: "B-MENTION", 2: "I-MENTION"}
CHECKPOINT = "distilbert/distilbert-base-uncased"


def aligned_labels(offsets, spans):
    """Assign BIO labels to exactly representable, non-overlapping spans."""
    tokens = sorted({tuple(offset) for offset in offsets if offset[0] < offset[1]})
    starts = [start for start, _ in tokens]
    labels = dict.fromkeys(tokens, 0)
    occupied_until = 0
    for start, end in sorted(set(spans), key=lambda span: (span[0], -span[1])):
        left = bisect_left(starts, start)
        right = bisect_left(starts, end)
        if (
            left < right
            and tokens[left][0] == start
            and tokens[right - 1][1] == end
            and start >= occupied_until
        ):
            for index in range(left, right):
                labels[tokens[index]] = 1 if index == left else 2
            occupied_until = end
        else:
            # Preserve any valid positive labels and ignore unsupported boundaries.
            for token in tokens[max(0, left - 1):right]:
                if token[1] > start and labels[token] == 0:
                    labels[token] = -100
    return labels


def training_features(tokenizer, dataset):
    from collections import defaultdict

    allowed = dataset.splits["train"]
    spans = defaultdict(list)
    for row in dataset.annotations:
        if row.note_id in allowed:
            spans[row.note_id].append((row.start, row.end))

    for note_id in sorted(allowed):
        text = dataset.notes[note_id]
        if not text.strip():
            continue
        encoded = tokenizer(
            text, truncation=True, max_length=256, stride=64,
            return_overflowing_tokens=True, return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping")
        labels = aligned_labels((offset for window in offsets for offset in window), spans[note_id])
        for index, window in enumerate(offsets):
            feature = {key: values[index] for key, values in encoded.items()}
            feature["labels"] = [labels.get(tuple(offset), -100) for offset in window]
            if any(label != -100 for label in feature["labels"]):
                yield feature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Directory containing the four Phase 1 CSVs.")
    parser.add_argument("--output", type=Path, default=NER_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

    if not torch.cuda.is_available():
        raise SystemExit("Enable a CUDA GPU in the Colab or Kaggle runtime.")
    if args.data_dir is None:
        dataset = load_dataset()
    else:
        dataset = load_dataset(DatasetPaths(
            args.data_dir / "notes.csv", args.data_dir / "annotations.csv",
            args.data_dir / "terminology.csv", args.data_dir / "splits.csv",
        ))

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT, num_labels=len(LABELS), id2label=LABELS,
        label2id={name: index for index, name in LABELS.items()},
    ).to("cuda")
    features = list(training_features(tokenizer, dataset))
    if not any(1 in feature["labels"] for feature in features):
        raise SystemExit("Training data contains no mentions aligned to tokenizer boundaries.")
    loader = torch.utils.data.DataLoader(
        features, batch_size=args.batch_size, shuffle=True, pin_memory=True,
        collate_fn=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for batch in loader:
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach().item()
        print(f"Epoch {epoch + 1}: mean training loss {total / len(loader):.4f}", flush=True)

    model.eval().cpu().half().save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(f"Exported FP16 weights and tokenizer to {args.output}")


if __name__ == "__main__":
    main()

