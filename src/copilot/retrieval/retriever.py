"""BM25, vector and hybrid (RRF) retrieval with an optional reranking stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from copilot.embeddings import Embedder
from copilot.models import ScoredChunk
from copilot.observability import tracer
from copilot.retrieval.bm25 import BM25Index, tokenize
from copilot.retrieval.fusion import reciprocal_rank_fusion
from copilot.retrieval.reranker import CrossEncoderReranker
from copilot.store.base import VectorStore

Mode = Literal["bm25", "vector", "hybrid"]


@dataclass
class RetrievalResult:
    items: list[ScoredChunk]
    mode: str
    max_vector_sim: float  # best dense similarity of the query to any chunk: abstention signal
    reranked: bool
    top_bm25_norm: float | None = None  # top BM25 score / number of query terms (bm25, hybrid)
    top_rerank: float | None = None  # cross-encoder score of the top item (when reranked)
    coverage: float | None = None  # IDF-weighted share of query terms present in the top-3 chunks

    def signals(self) -> dict[str, float | None]:
        return {
            "dense": self.max_vector_sim,
            "rerank": self.top_rerank,
            "bm25_norm": self.top_bm25_norm,
            "coverage": self.coverage,
        }


COVERAGE_TOP = 3


def term_coverage(query_terms: list[str], texts: list[str], idf) -> float:
    """IDF-weighted fraction of distinct query terms that occur in any of ``texts``."""
    terms = list(dict.fromkeys(query_terms))
    if not terms:
        return 0.0
    present: set[str] = set()
    for t in texts:
        present.update(tokenize(t))
    weights = [idf(t) for t in terms]
    total = sum(weights)
    if total <= 0:
        return 0.0
    return sum(w for t, w in zip(terms, weights, strict=True) if t in present) / total


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        candidate_k: int = 30,
        rrf_k: int = 60,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.reranker = reranker
        self._bm25: BM25Index | None = None
        self._bm25_rev: str | None = None

    def bm25(self) -> BM25Index:
        rev = self.store.revision()
        if self._bm25 is None or rev != self._bm25_rev:
            self._bm25 = BM25Index(self.store.all_chunks())
            self._bm25_rev = rev
        return self._bm25

    def retrieve(
        self, query: str, k: int = 5, mode: Mode = "hybrid", rerank: bool | None = None
    ) -> RetrievalResult:
        use_rerank = self.reranker is not None if rerank is None else (rerank and self.reranker is not None)
        n_cand = max(self.candidate_k, k)

        # Dense search always runs: it feeds the abstention signal even in bm25 mode.
        with tracer().start_as_current_span("retrieve.dense") as sp:
            qv = self.embedder.embed_query(query)
            vec_hits = self.store.vector_search(qv, n_cand)
            sp.set_attribute("hits", len(vec_hits))
        max_sim = vec_hits[0][1] if vec_hits else 0.0
        vec_rank = {c.chunk_id: (r, s) for r, (c, s) in enumerate(vec_hits, start=1)}

        with tracer().start_as_current_span("retrieve.bm25") as sp:
            bm_hits = self.bm25().search(query, n_cand) if mode in ("bm25", "hybrid") else []
            sp.set_attribute("hits", len(bm_hits))
        bm_rank = {c.chunk_id: (r, s) for r, (c, s) in enumerate(bm_hits, start=1)}

        by_id = {c.chunk_id: c for c, _ in vec_hits} | {c.chunk_id: c for c, _ in bm_hits}

        if mode == "bm25":
            ordered = [(c.chunk_id, s) for c, s in bm_hits]
        elif mode == "vector":
            ordered = [(c.chunk_id, s) for c, s in vec_hits]
        else:
            ordered = reciprocal_rank_fusion(
                [[c.chunk_id for c, _ in bm_hits], [c.chunk_id for c, _ in vec_hits]], k=self.rrf_k
            )

        items: list[ScoredChunk] = []
        for cid, score in ordered:
            sc: dict[str, float] = {}
            if cid in bm_rank:
                sc["bm25_rank"], sc["bm25"] = bm_rank[cid][0], round(bm_rank[cid][1], 4)
            if cid in vec_rank:
                sc["vector_rank"], sc["vector_sim"] = vec_rank[cid][0], round(vec_rank[cid][1], 4)
            if mode == "hybrid":
                sc["rrf"] = round(score, 6)
            items.append(ScoredChunk(by_id[cid], float(score), sc))

        top_bm25_norm = None
        if mode in ("bm25", "hybrid"):
            n_terms = max(1, len(tokenize(query)))
            top_bm25_norm = float(bm_hits[0][1]) / n_terms if bm_hits else 0.0

        top_rerank = None
        if use_rerank:
            with tracer().start_as_current_span("rerank") as sp:
                sp.set_attribute("candidates", min(len(items), n_cand))
                sp.set_attribute("model", getattr(self.reranker, "name", ""))
                items = self.reranker.rerank(query, items[:n_cand], k)
                top_rerank = items[0].scores["rerank"] if items else None

        idx = self.bm25()
        coverage = term_coverage(
            tokenize(query),
            [f"{it.chunk.doc_title} {it.chunk.section_path} {it.chunk.text}" for it in items[:COVERAGE_TOP]],
            idx.idf,
        )
        return RetrievalResult(
            items[:k], mode, float(max_sim), use_rerank, top_bm25_norm, top_rerank, coverage
        )
