from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
from food_intelligence.module4_knowledge_base import KnowledgeDocument
from .embeddings import LightweightEmbedder


class VectorStore:
    """In-memory Vector Store with Cosine Similarity Search for Knowledge Documents."""

    def __init__(self, embedder: LightweightEmbedder | None = None):
        self.embedder = embedder or LightweightEmbedder()
        self.documents: List[KnowledgeDocument] = []
        self.vectors: List[np.ndarray] = []

    def add_documents(self, docs: List[KnowledgeDocument]) -> None:
        if not docs:
            return
        texts = [doc.content for doc in docs]
        embeddings = self.embedder.embed_documents(texts)
        for doc, emb in zip(docs, embeddings):
            self.documents.append(doc)
            self.vectors.append(np.array(emb, dtype=np.float32))

    def similarity_search_with_score(
        self, query: str, k: int = 3, doc_type: str | None = None
    ) -> List[Tuple[KnowledgeDocument, float]]:
        if not self.documents or not self.vectors:
            return []

        q_vec = np.array(self.embedder.embed_query(query), dtype=np.float32)
        results = []

        for doc, v in zip(self.documents, self.vectors):
            if doc_type and doc.doc_type != doc_type:
                continue
            # Cosine similarity
            dot = np.dot(q_vec, v)
            denom = np.linalg.norm(q_vec) * np.linalg.norm(v)
            score = float(dot / denom) if denom > 0 else 0.0
            results.append((doc, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
