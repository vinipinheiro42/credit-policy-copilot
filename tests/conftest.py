from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from copilot.embeddings import HashEmbedder
from copilot.ingest.manifest import load_manifest
from copilot.ingest.pipeline import ingest
from copilot.store.sqlite_store import SQLiteStore

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """A writable copy of the fixture corpus (tests may edit documents)."""
    dst = tmp_path / "corpus"
    shutil.copytree(FIXTURES / "corpus", dst)
    shutil.copy(FIXTURES / "manifest.yaml", dst / "MANIFEST.yaml")
    return dst


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(384)


@pytest.fixture
def store() -> SQLiteStore:
    return SQLiteStore(":memory:")


@pytest.fixture
def indexed_store(corpus_dir, store, embedder) -> SQLiteStore:
    ingest(load_manifest(corpus_dir / "MANIFEST.yaml"), corpus_dir, store, embedder)
    return store
