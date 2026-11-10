import pytest

from copilot.embeddings import HashEmbedder
from copilot.ingest.manifest import load_manifest
from copilot.ingest.pipeline import ingest
from copilot.store.base import EmbedderMismatchError


def _run(corpus_dir, store, embedder, **kw):
    return ingest(load_manifest(corpus_dir / "MANIFEST.yaml"), corpus_dir, store, embedder, **kw)


def test_first_ingest_indexes_everything(corpus_dir, store, embedder):
    s = _run(corpus_dir, store, embedder)
    assert s.documents_indexed == 2 and s.documents_unchanged == 0
    assert s.chunks_added == store.stats()["chunks"] > 0
    assert s.chunks_removed == 0


def test_reingest_is_a_noop(corpus_dir, store, embedder):
    _run(corpus_dir, store, embedder)
    before = store.stats()["chunks"]
    ids_before = {c.chunk_id for c in store.all_chunks()}
    s = _run(corpus_dir, store, embedder)
    assert s.documents_indexed == 0 and s.documents_unchanged == 2
    assert s.chunks_added == 0 and s.chunks_removed == 0
    assert store.stats()["chunks"] == before
    assert {c.chunk_id for c in store.all_chunks()} == ids_before


def test_changed_document_reindexes_only_changed_chunks(corpus_dir, store, embedder):
    _run(corpus_dir, store, embedder)
    old_ids = {c.chunk_id for c in store.all_chunks() if c.doc_id == "POL-A"}
    f = corpus_dir / "POL-A.md"
    f.write_text(f.read_text().replace("180 days past due", "150 days past due"))

    s = _run(corpus_dir, store, embedder)
    assert s.documents_indexed == 1 and s.documents_unchanged == 1
    assert s.chunks_added == 1 and s.chunks_removed == 1
    assert s.chunks_reused == len(old_ids) - 1
    texts = " ".join(c.text for c in store.all_chunks())
    assert "150 days" in texts and "180 days" not in texts


def test_document_removed_from_manifest_is_pruned(corpus_dir, store, embedder):
    _run(corpus_dir, store, embedder)
    m = corpus_dir / "MANIFEST.yaml"
    lines = m.read_text().split("  - doc_id: POL-B")[0]
    m.write_text(lines)
    s = _run(corpus_dir, store, embedder)
    assert s.documents_pruned == ["POL-B"]
    assert {c.doc_id for c in store.all_chunks()} == {"POL-A"}


def test_manifest_hash_mismatch_is_reported(corpus_dir, store, embedder):
    m = corpus_dir / "MANIFEST.yaml"
    m.write_text(m.read_text().replace("    path: POL-A.md\n", "    path: POL-A.md\n    sha256: deadbeef\n"))
    s = _run(corpus_dir, store, embedder)
    assert s.hash_mismatches == ["POL-A"]


def test_switching_embedder_requires_rebuild(corpus_dir, store, embedder):
    _run(corpus_dir, store, embedder)
    with pytest.raises(EmbedderMismatchError):
        _run(corpus_dir, store, HashEmbedder(128))
