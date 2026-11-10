"""Core data structures shared by ingestion, storage, retrieval and the API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Section:
    """A logical section of a document, identified by its heading path."""

    path: tuple[str, ...]
    text: str
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    source_path: str
    content_hash: str
    sections: list[Section]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section_path: str
    ordinal: int
    text: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)
