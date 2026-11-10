import numpy as np
import pytest

from copilot.embeddings import HashEmbedder


def test_hash_embedder_is_deterministic_and_normalized():
    e = HashEmbedder(384)
    a, b = e.embed_documents(["Grade A cap is 25,000"]), e.embed_documents(["Grade A cap is 25,000"])
    assert np.array_equal(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)
    assert e.embed_documents([]).shape == (0, 384)


def test_hash_embedder_ranks_lexical_overlap_higher():
    e = HashEmbedder(384)
    q = e.embed_query("credit card charge off days past due")
    docs = e.embed_documents(["Credit card balances are charged off at 180 days past due.", "Pizza recipe."])
    sims = docs @ q
    assert sims[0] > sims[1]


@pytest.mark.slow
def test_sentence_transformer_embedder_loads_on_cpu():
    pytest.importorskip("sentence_transformers")
    from copilot.embeddings import SentenceTransformerEmbedder

    try:
        e = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
    except OSError as exc:  # no network / model not cached
        pytest.skip(f"model unavailable: {exc}")
    q = e.embed_query("maximum loan-to-value for a used vehicle")
    d = e.embed_documents(["Used vehicle: maximum LTV 80%.", "Sanctions screening runs daily."])
    assert q.shape == (e.dim,) == (384,)
    assert (d @ q)[0] > (d @ q)[1]
