"""In-memory vector index with cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import HashingEmbedder
from .ingest import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    channel: str  # "vector" | "keyword" | "hybrid"


class VectorStore:
    def __init__(self, embedder: HashingEmbedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        vectors = self.embedder.embed_many([c.text for c in chunks])
        if self.matrix is None:
            self.matrix = vectors
        else:
            self.matrix = np.vstack([self.matrix, vectors])

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if not self.chunks or self.matrix is None:
            return []
        q = self.embedder.embed(query)
        scores = self.matrix @ q
        order = np.argsort(-scores)[:top_k]
        return [
            ScoredChunk(chunk=self.chunks[i], score=float(scores[i]), channel="vector")
            for i in order
            if scores[i] > 0
        ]
