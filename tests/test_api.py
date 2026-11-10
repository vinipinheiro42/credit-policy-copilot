import pytest
from fastapi.testclient import TestClient

from copilot.api.app import create_app
from copilot.config import Settings
from copilot.llm.extractive import ExtractiveGenerator
from copilot.retrieval.retriever import Retriever
from copilot.service import CopilotService


@pytest.fixture
def client(indexed_store, embedder):
    settings = Settings(
        store="sqlite", embedder="hash", abstain_threshold=0.15, abstain_rule_path="", top_k=3
    )
    svc = CopilotService(
        settings, indexed_store, embedder, ExtractiveGenerator(embedder), Retriever(indexed_store, embedder)
    )
    with TestClient(create_app(svc)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["store"]["chunks"] > 0
    assert body["llm_provider"] == "extractive"


def test_ask_contract_returns_answer_citations_and_contexts(client):
    r = client.post("/ask", json={"question": "When are credit card balances charged off?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "answer",
        "abstained",
        "support_score",
        "citations",
        "contexts",
        "retrieval",
        "generation",
        "latency_ms",
        "abstain_reason",
        "guardrails",
        "request_id",
        "cost_usd",
    }
    assert body["abstained"] is False
    assert r.headers["X-Request-ID"] == body["request_id"]
    assert body["cost_usd"] == 0.0
    assert "180 days" in body["answer"]
    cit = body["citations"][0]
    assert set(cit) >= {"doc_id", "section", "chunk_id", "snippet"}
    assert cit["doc_id"] == "POL-A" and cit["chunk_id"].startswith("POL-A::")
    assert {c["chunk_id"] for c in body["citations"]} <= {c["chunk_id"] for c in body["contexts"]}
    assert body["contexts"][0]["rank"] == 1 and "scores" in body["contexts"][0]


def test_ask_abstains_when_nothing_relevant(client):
    body = client.post("/ask", json={"question": "Best pizza recipe with mozzarella?"}).json()
    assert body["abstained"] is True
    assert body["citations"] == []
    assert body["support_score"] < 0.15


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_ask_accepts_retrieval_mode(client, mode):
    body = client.post("/ask", json={"question": "suspicious activity report deadline", "mode": mode}).json()
    assert body["retrieval"]["mode"] == mode
    assert body["contexts"][0]["doc_id"] == "POL-B"


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "ok?", "top_k": 99}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "mode": "magic"}).status_code == 422


def test_rerank_flag_without_model_is_a_client_error(client):
    assert client.post("/ask", json={"question": "charge off", "rerank": True}).status_code == 400


def test_rerank_rule_degrades_gracefully_without_reranker(indexed_store):
    from copilot.abstention import AbstentionRule
    from copilot.embeddings import HashEmbedder
    from copilot.llm.extractive import ExtractiveGenerator
    from copilot.retrieval.retriever import Retriever
    from copilot.service import CopilotService

    s = Settings(store="sqlite", embedder="hash", abstain_rule_path="", top_k=3)
    emb = HashEmbedder(s.embedding_dim)
    rule = AbstentionRule({"dense": 0.0, "rerank": 100.0}, "any")
    svc = CopilotService(s, indexed_store, emb, ExtractiveGenerator(emb), Retriever(indexed_store, emb), rule)
    r = svc.ask("credit limit")
    assert r.abstained is False  # rerank criterion skipped, dense criterion passes
    assert "abstain_rule_degraded" in r.retrieval
    assert r.retrieval["signals"]["rerank"] is None
