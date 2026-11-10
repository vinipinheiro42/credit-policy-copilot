from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: Literal["bm25", "vector", "hybrid"] | None = None
    rerank: bool | None = None


class Citation(BaseModel):
    ref: int
    doc_id: str
    doc_title: str
    section: str
    chunk_id: str
    snippet: str
    page_start: int | None = None
    source_url: str | None = None


class Context(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    scores: dict[str, float]
    metadata: dict[str, Any] = {}


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    support_score: float
    citations: list[Citation]
    contexts: list[Context]
    retrieval: dict[str, Any]
    generation: dict[str, Any]
    latency_ms: dict[str, float]
    abstain_reason: str | None = None
    guardrails: dict[str, Any] = {}
    request_id: str | None = None
    cost_usd: float | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    store: dict[str, Any]
    embedder: str
    llm_provider: str
    retrieval_mode: str
    rerank: bool
