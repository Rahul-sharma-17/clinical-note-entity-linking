"""SapBERT encoding and exact CPU cosine retrieval for Phase 4 indexes."""

from functools import lru_cache

CHECKPOINT = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


class SapBERTEncoder:
    def __init__(self, source=CHECKPOINT, *, device="cpu", local_only=False):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        if device == "cpu":
            torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only)
        self.model = AutoModel.from_pretrained(source, local_files_only=local_only).to(device).eval()
        if device == "cuda":
            self.model.half()
        else:
            self.model.float()

    def encode(self, texts):
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=64, return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            cls = self.model(**batch).last_hidden_state[:, 0].float()
            return self.torch.nn.functional.normalize(cls, dim=1).cpu().numpy()

    def save(self, path):
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)


class DenseRetriever:
    """Take the maximum cosine score over aliases of each concept."""

    def __init__(self, rows, path, encoder):
        import numpy as np

        self.np = np
        vectors = np.load(path, mmap_mode="r", allow_pickle=False)
        if vectors.ndim != 2 or len(vectors) != len(rows) or not rows or vectors.shape[1] == 0:
            raise ValueError("Invalid linker index dimensions")
        self.matrix = vectors.astype(np.float32)
        if not np.isfinite(self.matrix).all():
            raise ValueError("Non-finite embeddings in linker index")
        self.codes = []
        self.offsets = []
        for index, (code, _) in enumerate(rows):
            if not self.codes or self.codes[-1] != code:
                self.codes.append(code)
                self.offsets.append(index)
        self.encoder = encoder
        self._rank = lru_cache(maxsize=2048)(self._rank_uncached)

    def rank(self, text, k):
        return self._rank(text)[:k]

    def _rank_uncached(self, text):
        from .entity_linker import CALIBRATION_K

        vector = self.encoder.encode([text])[0]
        similarities = self.np.einsum("ij,j->i", self.matrix, vector)
        scores = self.np.maximum.reduceat(similarities, self.offsets)
        indices = self.np.argsort(-scores, kind="stable")[:CALIBRATION_K]
        return tuple((self.codes[index], float(scores[index])) for index in indices)

