"""SQLite-backed store with brute-force cosine search in NumPy.

Fallback for environments without Postgres (laptops, CI without services, unit tests).
Brute force is exact and fast enough for tens of thousands of chunks.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from copilot.models import Chunk
from copilot.store.base import DocumentRecord, EmbedderMismatchError

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY, title TEXT, source_path TEXT, content_hash TEXT NOT NULL,
  metadata TEXT, indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  doc_title TEXT, section_path TEXT, ordinal INTEGER, text TEXT NOT NULL,
  content_hash TEXT, metadata TEXT, embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


class SQLiteStore:
    name = "sqlite"

    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._cache: tuple[str, list[Chunk], np.ndarray] | None = None

    def ensure_embedder(self, embedder_name: str, dim: int) -> None:
        row = self.conn.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
        sig = f"{embedder_name}|{dim}"
        if row is None:
            with self.conn:
                self.conn.execute("INSERT INTO meta(key, value) VALUES ('embedder', ?)", (sig,))
        elif row[0] != sig:
            raise EmbedderMismatchError(
                f"Index built with {row[0]!r}, configured {sig!r}. Re-ingest with --rebuild."
            )

    def get_document_hash(self, doc_id: str) -> str | None:
        row = self.conn.execute("SELECT content_hash FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return row[0] if row else None

    def chunk_ids(self, doc_id: str) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,))}

    def upsert_document(self, doc: DocumentRecord, chunks: list[Chunk]) -> tuple[int, int]:
        existing = self.chunk_ids(doc.doc_id)
        new_ids = {c.chunk_id for c in chunks}
        to_delete = existing - new_ids
        added = 0
        with self.conn:
            self.conn.execute(
                """INSERT INTO documents(doc_id, title, source_path, content_hash, metadata)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(doc_id) DO UPDATE SET title=excluded.title,
                     source_path=excluded.source_path, content_hash=excluded.content_hash,
                     metadata=excluded.metadata, indexed_at=CURRENT_TIMESTAMP""",
                (
                    doc.doc_id,
                    doc.title,
                    doc.source_path,
                    doc.content_hash,
                    json.dumps(doc.metadata, default=str),
                ),
            )
            self.conn.executemany("DELETE FROM chunks WHERE chunk_id=?", [(i,) for i in to_delete])
            for c in chunks:
                if c.chunk_id in existing:
                    # text unchanged (content-addressed id); refresh position/metadata only
                    self.conn.execute(
                        "UPDATE chunks SET ordinal=?, doc_title=?, metadata=? WHERE chunk_id=?",
                        (c.ordinal, c.doc_title, json.dumps(c.metadata), c.chunk_id),
                    )
                    continue
                if c.embedding is None:
                    raise ValueError(f"New chunk {c.chunk_id} has no embedding")
                self.conn.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.doc_title,
                        c.section_path,
                        c.ordinal,
                        c.text,
                        c.content_hash,
                        json.dumps(c.metadata),
                        np.asarray(c.embedding, dtype=np.float32).tobytes(),
                    ),
                )
                added += 1
        self._cache = None
        return added, len(to_delete)

    def delete_documents_except(self, keep_doc_ids: set[str]) -> list[str]:
        all_ids = {r[0] for r in self.conn.execute("SELECT doc_id FROM documents")}
        gone = sorted(all_ids - keep_doc_ids)
        with self.conn:
            self.conn.executemany("DELETE FROM documents WHERE doc_id=?", [(d,) for d in gone])
        self._cache = None
        return gone

    def _load(self) -> tuple[list[Chunk], np.ndarray]:
        rev = self.revision()
        if self._cache and self._cache[0] == rev:
            return self._cache[1], self._cache[2]
        rows = self.conn.execute(
            "SELECT chunk_id, doc_id, doc_title, section_path, ordinal, text, content_hash, "
            "metadata, embedding FROM chunks ORDER BY doc_id, ordinal"
        ).fetchall()
        chunks = [Chunk(r[0], r[1], r[2], r[3], r[4], r[5], r[6], json.loads(r[7] or "{}")) for r in rows]
        mat = (
            np.stack([np.frombuffer(r[8], dtype=np.float32) for r in rows])
            if rows
            else np.zeros((0, 1), dtype=np.float32)
        )
        self._cache = (rev, chunks, mat)
        return chunks, mat

    def all_chunks(self) -> list[Chunk]:
        return list(self._load()[0])

    def vector_search(self, query: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        chunks, mat = self._load()
        if not chunks:
            return []
        sims = mat @ np.asarray(query, dtype=np.float32)
        idx = np.argsort(-sims)[:k]
        return [(chunks[i], float(sims[i])) for i in idx]

    def revision(self) -> str:
        rows = self.conn.execute("SELECT doc_id, content_hash FROM documents ORDER BY doc_id").fetchall()
        n = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return hashlib.sha256(json.dumps([rows, n]).encode()).hexdigest()

    def stats(self) -> dict[str, Any]:
        docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        emb = self.conn.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
        return {"store": self.name, "documents": docs, "chunks": chunks, "embedder": emb[0] if emb else None}

    def reset(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM chunks")
            self.conn.execute("DELETE FROM documents")
            self.conn.execute("DELETE FROM meta")
        self._cache = None
