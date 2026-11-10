"""Optional cross-encoder reranker (enabled with COPILOT_RERANK=true)."""

from __future__ import annotations

from copilot.models import ScoredChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device="cpu")
        self.name = model_name

    def rerank(self, query: str, items: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not items:
            return []
        scores = self.model.predict([(query, it.chunk.text) for it in items])
        for it, s in zip(items, scores, strict=True):
            it.scores["rerank"] = float(s)
            it.score = float(s)
        return sorted(items, key=lambda it: -it.score)[:k]
