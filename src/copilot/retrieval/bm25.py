"""BM25 lexical retrieval over all chunks (Okapi BM25 via rank-bm25)."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi

from copilot.models import Chunk

_STOP = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "must",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "o",
        "a",
        "os",
        "as",
        "de",
        "da",
        "do",
        "das",
        "dos",
        "e",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "um",
        "uma",
        "para",
        "por",
        "que",
        "se",
        "ao",
        "com",
    ]
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return [t for t in _TOKEN.findall(text) if t not in _STOP]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        corpus = [tokenize(f"{c.doc_title} {c.section_path} {c.text}") for c in chunks]
        self.bm25 = BM25Okapi(corpus) if chunks else None
        self._max_idf = max(self.bm25.idf.values()) if self.bm25 and self.bm25.idf else 1.0

    def idf(self, term: str) -> float:
        """IDF of a term; unseen terms get the maximum IDF (they are the most informative)."""
        if not self.bm25:
            return 1.0
        return max(float(self.bm25.idf.get(term, self._max_idf)), 0.0)

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self.bm25:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        idx = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in idx if scores[i] > 0]
