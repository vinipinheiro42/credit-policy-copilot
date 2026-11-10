"""Embedding backends behind one small interface."""

from __future__ import annotations

import hashlib
import itertools
import re
import unicodedata
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return _TOKEN.findall(text)


class HashEmbedder:
    """Deterministic feature-hashing embedder (unigrams + bigrams).

    No model download, identical output on every machine: used for unit tests and as an
    offline fallback. It captures lexical overlap only, not semantics.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = _tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in itertools.pairwise(toks)]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder:
    """Local CPU embedding model via sentence-transformers.

    E5 models expect "query: " / "passage: " prefixes; they are added automatically.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.name = model_name
        m = self.model
        get_dim = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.batch_size = batch_size
        self._e5 = "e5" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self._e5:
            texts = [f"passage: {t}" for t in texts]
        return np.asarray(
            self.model.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        q = f"query: {text}" if self._e5 else text
        return np.asarray(self.model.encode([q], normalize_embeddings=True)[0], dtype=np.float32)


def build_embedder(kind: str, model_name: str, dim: int = 384) -> Embedder:
    if kind == "hash":
        return HashEmbedder(dim)
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown embedder: {kind}")
