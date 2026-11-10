"""Postgres + pgvector store (production default)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from copilot.models import Chunk
from copilot.store.base import DocumentRecord, EmbedderMismatchError


class PgVectorStore:
    name = "pgvector"

    def __init__(self, dsn: str, dim: int = 384):
        self.dsn = dsn
        self.dim = dim
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self._create_schema()

    def _create_schema(self) -> None:
        with self.conn.transaction():
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS copilot_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS documents (
                     doc_id TEXT PRIMARY KEY, title TEXT, source_path TEXT,
                     content_hash TEXT NOT NULL, metadata JSONB,
                     indexed_at TIMESTAMPTZ NOT NULL DEFAULT now())"""
            )
            self.conn.execute(
                f"""CREATE TABLE IF NOT EXISTS chunks (
                     chunk_id TEXT PRIMARY KEY,
                     doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                     doc_title TEXT, section_path TEXT, ordinal INT, text TEXT NOT NULL,
                     content_hash TEXT, metadata JSONB, embedding vector({self.dim}) NOT NULL)"""
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks "
                "USING hnsw (embedding vector_cosine_ops)"
            )

    def ensure_embedder(self, embedder_name: str, dim: int) -> None:
        sig = f"{embedder_name}|{dim}"
        row = self.conn.execute("SELECT value FROM copilot_meta WHERE key='embedder'").fetchone()
        if row is None:
            self.conn.execute("INSERT INTO copilot_meta(key, value) VALUES ('embedder', %s)", (sig,))
        elif row[0] != sig:
            raise EmbedderMismatchError(
                f"Index built with {row[0]!r}, configured {sig!r}. Re-ingest with --rebuild."
            )

    def get_document_hash(self, doc_id: str) -> str | None:
        row = self.conn.execute("SELECT content_hash FROM documents WHERE doc_id=%s", (doc_id,)).fetchone()
        return row[0] if row else None

    def chunk_ids(self, doc_id: str) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT chunk_id FROM chunks WHERE doc_id=%s", (doc_id,))}

    def upsert_document(self, doc: DocumentRecord, chunks: list[Chunk]) -> tuple[int, int]:
        added = 0
        with self.conn.transaction():
            # lock the document row so concurrent ingests of the same doc serialize
            self.conn.execute(
                """INSERT INTO documents(doc_id, title, source_path, content_hash, metadata)
                   VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT (doc_id) DO UPDATE SET title=EXCLUDED.title,
                     source_path=EXCLUDED.source_path, content_hash=EXCLUDED.content_hash,
                     metadata=EXCLUDED.metadata, indexed_at=now()""",
                (doc.doc_id, doc.title, doc.source_path, doc.content_hash, Jsonb(doc.metadata)),
            )
            self.conn.execute("SELECT 1 FROM documents WHERE doc_id=%s FOR UPDATE", (doc.doc_id,))
            existing = self.chunk_ids(doc.doc_id)
            new_ids = [c.chunk_id for c in chunks]
            to_delete = existing - set(new_ids)
            if to_delete:
                self.conn.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (list(to_delete),))
            with self.conn.cursor() as cur:
                for c in chunks:
                    if c.chunk_id in existing:
                        cur.execute(
                            "UPDATE chunks SET ordinal=%s, doc_title=%s, metadata=%s WHERE chunk_id=%s",
                            (c.ordinal, c.doc_title, Jsonb(c.metadata), c.chunk_id),
                        )
                        continue
                    if c.embedding is None:
                        raise ValueError(f"New chunk {c.chunk_id} has no embedding")
                    cur.execute(
                        """INSERT INTO chunks(chunk_id, doc_id, doc_title, section_path, ordinal, text,
                                              content_hash, metadata, embedding)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                           ON CONFLICT (chunk_id) DO NOTHING""",
                        (
                            c.chunk_id,
                            c.doc_id,
                            c.doc_title,
                            c.section_path,
                            c.ordinal,
                            c.text,
                            c.content_hash,
                            Jsonb(c.metadata),
                            np.asarray(c.embedding, dtype=np.float32),
                        ),
                    )
                    added += cur.rowcount
        return added, len(to_delete)

    def delete_documents_except(self, keep_doc_ids: set[str]) -> list[str]:
        rows = self.conn.execute(
            "DELETE FROM documents WHERE NOT (doc_id = ANY(%s)) RETURNING doc_id", (list(keep_doc_ids),)
        ).fetchall()
        return sorted(r[0] for r in rows)

    def all_chunks(self) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT chunk_id, doc_id, doc_title, section_path, ordinal, text, content_hash, metadata "
            "FROM chunks ORDER BY doc_id, ordinal"
        ).fetchall()
        return [Chunk(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7] or {}) for r in rows]

    def vector_search(self, query: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        q = np.asarray(query, dtype=np.float32)
        rows = self.conn.execute(
            "SELECT chunk_id, doc_id, doc_title, section_path, ordinal, text, content_hash, metadata, "
            "1 - (embedding <=> %s) AS sim FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (q, q, k),
        ).fetchall()
        return [(Chunk(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7] or {}), float(r[8])) for r in rows]

    def revision(self) -> str:
        rows = self.conn.execute("SELECT doc_id, content_hash FROM documents ORDER BY doc_id").fetchall()
        n = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return hashlib.sha256(json.dumps([[list(r) for r in rows], n]).encode()).hexdigest()

    def stats(self) -> dict[str, Any]:
        docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        emb = self.conn.execute("SELECT value FROM copilot_meta WHERE key='embedder'").fetchone()
        return {"store": self.name, "documents": docs, "chunks": chunks, "embedder": emb[0] if emb else None}

    def reset(self) -> None:
        with self.conn.transaction():
            self.conn.execute("DROP TABLE IF EXISTS chunks")
            self.conn.execute("DROP TABLE IF EXISTS documents")
            self.conn.execute("DROP TABLE IF EXISTS copilot_meta")
        self._create_schema()
