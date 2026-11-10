from copilot.retrieval.retriever import Retriever


def test_bm25_vector_and_hybrid_find_the_right_section(indexed_store, embedder):
    r = Retriever(indexed_store, embedder, candidate_k=10)
    for mode in ("bm25", "vector", "hybrid"):
        res = r.retrieve("When are credit card balances charged off?", k=3, mode=mode)
        assert res.items, mode
        assert res.items[0].chunk.section_path == "2. Charge-off", mode


def test_hybrid_scores_expose_both_rankers(indexed_store, embedder):
    res = Retriever(indexed_store, embedder).retrieve("sanctions screening daily", k=2, mode="hybrid")
    top = res.items[0]
    assert top.chunk.doc_id == "POL-B"
    assert {"rrf", "bm25_rank", "vector_rank"} <= set(top.scores)


def test_bm25_index_refreshes_after_reingest(indexed_store, embedder, corpus_dir):
    from copilot.ingest.manifest import load_manifest
    from copilot.ingest.pipeline import ingest

    r = Retriever(indexed_store, embedder)
    assert not r.retrieve("zebra", k=3, mode="bm25").items
    f = corpus_dir / "POL-B.md"
    f.write_text(f.read_text() + "\n## 3. Zebra\n\nZebra accounts are reviewed weekly.\n")
    ingest(load_manifest(corpus_dir / "MANIFEST.yaml"), corpus_dir, indexed_store, embedder)
    assert r.retrieve("zebra", k=3, mode="bm25").items[0].chunk.section_path == "3. Zebra"
