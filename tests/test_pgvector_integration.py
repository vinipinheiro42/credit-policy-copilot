"""Integration tests against a real Postgres + pgvector.

Uses COPILOT_TEST_DATABASE_URL when set (e.g. the docker-compose `db` service); otherwise
starts an embedded Postgres via `pgserver` if installed; otherwise skips.
"""

from __future__ import annotations

import os

import pytest

from copilot.ingest.manifest import load_manifest
from copilot.ingest.pipeline import ingest
from copilot.retrieval.retriever import Retriever

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pg_dsn(tmp_path_factory):
    dsn = os.environ.get("COPILOT_TEST_DATABASE_URL")
    if dsn:
        yield dsn
        return
    pgserver = pytest.importorskip(
        "pgserver", reason="no COPILOT_TEST_DATABASE_URL and pgserver not installed"
    )
    srv = pgserver.get_server(str(tmp_path_factory.mktemp("pgdata")), cleanup_mode="stop")
    srv.psql("CREATE EXTENSION IF NOT EXISTS vector;")
    yield srv.get_uri()
    srv.cleanup()


@pytest.fixture
def pg_store(pg_dsn):
    from copilot.store.pgvector_store import PgVectorStore

    s = PgVectorStore(pg_dsn, dim=384)
    s.reset()
    yield s
    s.reset()
    s.conn.close()


def test_pgvector_ingest_is_idempotent(pg_store, corpus_dir, embedder):
    entries = load_manifest(corpus_dir / "MANIFEST.yaml")
    first = ingest(entries, corpus_dir, pg_store, embedder)
    n = pg_store.stats()["chunks"]
    assert first.chunks_added == n > 0
    second = ingest(entries, corpus_dir, pg_store, embedder)
    assert second.documents_unchanged == 2 and second.chunks_added == 0
    assert pg_store.stats()["chunks"] == n


def test_pgvector_changed_doc_and_prune(pg_store, corpus_dir, embedder):
    entries = load_manifest(corpus_dir / "MANIFEST.yaml")
    ingest(entries, corpus_dir, pg_store, embedder)
    f = corpus_dir / "POL-A.md"
    f.write_text(f.read_text().replace("180 days", "150 days"))
    s = ingest(entries, corpus_dir, pg_store, embedder)
    assert (s.chunks_added, s.chunks_removed) == (1, 1)
    s2 = ingest([e for e in entries if e.doc_id == "POL-A"], corpus_dir, pg_store, embedder)
    assert s2.documents_pruned == ["POL-B"]
    assert {c.doc_id for c in pg_store.all_chunks()} == {"POL-A"}


def test_pgvector_search_matches_sqlite_ranking(pg_store, indexed_store, corpus_dir, embedder):
    ingest(load_manifest(corpus_dir / "MANIFEST.yaml"), corpus_dir, pg_store, embedder)
    q = embedder.embed_query("sanctions screening at onboarding")
    pg = [(c.chunk_id, round(s, 4)) for c, s in pg_store.vector_search(q, 3)]
    lite = [(c.chunk_id, round(s, 4)) for c, s in indexed_store.vector_search(q, 3)]
    assert pg == lite
    res = Retriever(pg_store, embedder).retrieve("sanctions screening at onboarding", k=2)
    assert res.items[0].chunk.doc_id == "POL-B"
