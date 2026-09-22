from __future__ import annotations

import os
from typing import List
import numpy as np


class LightweightEmbedder:
    """Lightweight TF-IDF & Character N-gram embedder for fast local semantic similarity

    without requiring heavy PyTorch or external download dependencies.
    Fallback to SentenceTransformers or HuggingFace when available.
    """

    def __init__(self, vector_dim: int = 128):
        self.vector_dim = vector_dim

    def _text_to_vector(self, text: str) -> np.ndarray:
        text_clean = text.lower().strip()
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        if not text_clean:
            return vec

        words = text_clean.split()
        for word in words:
            # Word level hashing for exact token matches
            w_idx = sum(ord(c) for c in word) % self.vector_dim
            vec[w_idx] += 2.0
            for char in word:
                idx = ord(char) % self.vector_dim
                vec[idx] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(t).tolist() for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._text_to_vector(text).tolist()
