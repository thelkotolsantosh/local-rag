"""Sentence-transformers wrapper that returns normalized float32 vectors."""
from typing import List

import numpy as np


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        # Normalized vectors make inner product equal to cosine similarity.
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")
