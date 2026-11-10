"""Vector store interface. Implementations: pgvector (Postgres) and SQLite (fallback)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from copilot.models import Chunk


class EmbedderMismatchError(RuntimeError):
    """The index was built with a different embedding model than the one configured."""


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    source_path: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    name: str

    def ensure_embedder(self, embedder_name: str, dim: int) -> None:
        """Record the embedder on first use; raise EmbedderMismatchError if it changed."""

    def get_document_hash(self, doc_id: str) -> str | None: ...

    def chunk_ids(self, doc_id: str) -> set[str]: ...

    def upsert_document(self, doc: DocumentRecord, chunks: list[Chunk]) -> tuple[int, int]:
        """Atomically make ``chunks`` the full chunk set of ``doc``.

        Chunks whose ID already exists are kept as-is (their embedding may be None in the
        argument); new IDs are inserted; IDs no longer present are deleted.
        Returns (added, removed).
        """

    def delete_documents_except(self, keep_doc_ids: set[str]) -> list[str]: ...

    def all_chunks(self) -> list[Chunk]: ...

    def vector_search(self, query: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks by cosine similarity (higher is better)."""

    def revision(self) -> str:
        """Opaque value that changes whenever indexed content changes (cache invalidation)."""

    def stats(self) -> dict[str, Any]: ...

    def reset(self) -> None: ...


def build_store(settings) -> VectorStore:
    if settings.store == "pgvector":
        from copilot.store.pgvector_store import PgVectorStore

        return PgVectorStore(settings.database_url, settings.embedding_dim)
    from copilot.store.sqlite_store import SQLiteStore

    return SQLiteStore(settings.sqlite_path)
