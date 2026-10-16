"""Hybrid retrieval: vector cosine + BM25-style keyword, reciprocal-rank fused."""

from __future__ import annotations

import math
from collections import Counter

from .embeddings import tokenize
from .ingest import Chunk
from .vector_store import ScoredChunk, VectorStore


class KeywordIndex:
    """Lightweight BM25 over in-memory chunks (no external search service)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.doc_tokens: list[list[str]] = []
        self.df: Counter[str] = Counter()
        self.avgdl = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            toks = tokenize(chunk.text)
            self.chunks.append(chunk)
            self.doc_tokens.append(toks)
            for term in set(toks):
                self.df[term] += 1
        total = sum(len(t) for t in self.doc_tokens)
        self.avgdl = total / max(len(self.doc_tokens), 1)

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        q_terms = tokenize(query)
        n = len(self.chunks)
        scored: list[ScoredChunk] = []
        for i, toks in enumerate(self.doc_tokens):
            tf = Counter(toks)
            dl = len(toks) or 1
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                df = self.df[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scored.append(
                    ScoredChunk(chunk=self.chunks[i], score=score, channel="keyword")
                )
        scored.sort(key=lambda s: -s.score)
        return scored[:top_k]


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredChunk]],
    k: int = 60,
    top_k: int = 5,
) -> list[ScoredChunk]:
    """RRF merge of ranked lists keyed by chunk_id."""
    fused: dict[str, tuple[Chunk, float]] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            cid = item.chunk.chunk_id
            contrib = 1.0 / (k + rank)
            if cid in fused:
                chunk, score = fused[cid]
                fused[cid] = (chunk, score + contrib)
            else:
                fused[cid] = (item.chunk, contrib)
    ranked = sorted(fused.values(), key=lambda x: -x[1])
    return [
        ScoredChunk(chunk=c, score=s, channel="hybrid") for c, s in ranked[:top_k]
    ]


class HybridRetriever:
    def __init__(self) -> None:
        self.vector = VectorStore()
        self.keyword = KeywordIndex()

    def index(self, chunks: list[Chunk]) -> None:
        self.vector.add(chunks)
        self.keyword.add(chunks)

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        v = self.vector.search(query, top_k=top_k * 2)
        k = self.keyword.search(query, top_k=top_k * 2)
        return reciprocal_rank_fusion([v, k], top_k=top_k)
