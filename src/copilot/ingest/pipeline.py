"""Idempotent ingestion: parse -> chunk -> embed only new chunks -> transactional upsert."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from copilot.embeddings import Embedder
from copilot.ingest.chunker import CHUNKER_VERSION, chunk_document, embedding_text
from copilot.ingest.manifest import ManifestEntry
from copilot.ingest.parsers import PARSER_VERSION, parse_file, sha256_bytes
from copilot.store.base import DocumentRecord, VectorStore
from copilot.text import EMBED_TEXT_VERSION

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    documents_seen: int = 0
    documents_indexed: int = 0
    documents_unchanged: int = 0
    documents_pruned: list[str] = field(default_factory=list)
    chunks_added: int = 0
    chunks_removed: int = 0
    chunks_reused: int = 0
    hash_mismatches: list[str] = field(default_factory=list)
    seconds: float = 0.0


def index_fingerprint(file_sha: str, target_chars: int, max_chars: int, embedder_name: str) -> str:
    """Anything that changes the produced chunks/vectors must change this fingerprint."""
    raw = (
        f"{file_sha}|p{PARSER_VERSION}|c{CHUNKER_VERSION}|e{EMBED_TEXT_VERSION}|"
        f"{target_chars}|{max_chars}|{embedder_name}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def ingest(
    entries: list[ManifestEntry],
    corpus_dir: str | Path,
    store: VectorStore,
    embedder: Embedder,
    target_chars: int = 1200,
    max_chars: int = 1800,
    prune: bool = True,
) -> IngestStats:
    t0 = time.perf_counter()
    stats = IngestStats()
    store.ensure_embedder(f"{embedder.name}|et{EMBED_TEXT_VERSION}", embedder.dim)
    corpus_dir = Path(corpus_dir)

    for e in entries:
        stats.documents_seen += 1
        path = corpus_dir / e.path
        file_sha = sha256_bytes(path.read_bytes())
        if e.sha256 and e.sha256 != file_sha:
            log.warning(
                "sha256 mismatch for %s (manifest %s, file %s)", e.doc_id, e.sha256[:12], file_sha[:12]
            )
            stats.hash_mismatches.append(e.doc_id)
        fp = index_fingerprint(file_sha, target_chars, max_chars, embedder.name)
        if store.get_document_hash(e.doc_id) == fp:
            stats.documents_unchanged += 1
            continue

        doc = parse_file(path, doc_id=e.doc_id, extra_meta=e.meta())
        chunks = chunk_document(doc, target_chars, max_chars)
        existing = store.chunk_ids(e.doc_id)
        new = [c for c in chunks if c.chunk_id not in existing]
        if new:
            vecs = embedder.embed_documents([embedding_text(c) for c in new])
            for c, v in zip(new, vecs, strict=True):
                c.embedding = v.tolist()
        added, removed = store.upsert_document(
            DocumentRecord(e.doc_id, doc.title, e.path, fp, {**e.meta(), "file_sha256": file_sha}),
            chunks,
        )
        stats.documents_indexed += 1
        stats.chunks_added += added
        stats.chunks_removed += removed
        stats.chunks_reused += len(chunks) - len(new)
        log.info("indexed %s: %d chunks (+%d / -%d)", e.doc_id, len(chunks), added, removed)

    if prune:
        stats.documents_pruned = store.delete_documents_except({e.doc_id for e in entries})
    stats.seconds = round(time.perf_counter() - t0, 2)
    return stats
