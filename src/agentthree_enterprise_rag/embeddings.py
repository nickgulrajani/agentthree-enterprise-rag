"""Local hashing embeddings — no model download, no API keys, zero cost."""

from __future__ import annotations

import hashlib
import re

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9$]+", re.IGNORECASE)


def _stem(token: str) -> str:
    """Tiny stemmer so outages≈outage, bridged≈bridge, employees≈employee."""
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        # bridged → bridge (drop d when stem already ends with e)
        if token[:-1].endswith("e"):
            return token[:-1]
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t.lower()) for t in TOKEN_RE.findall(text)]


class HashingEmbedder:
    """Deterministic bag-of-tokens → fixed-dim unit vector via feature hashing."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _hash(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") % self.dim

    def _sign(self, token: str) -> float:
        digest = hashlib.sha256(("sign:" + token).encode("utf-8")).digest()
        return 1.0 if digest[0] % 2 == 0 else -1.0

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        tokens = tokenize(text)
        if not tokens:
            return vec
        for tok in tokens:
            idx = self._hash(tok)
            vec[idx] += self._sign(tok)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts])
