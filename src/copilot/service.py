"""Question answering service.

input guardrails -> retrieval (+rerank) -> abstention gate -> document guardrails -> generation
-> output checks -> citations. Every step is an OpenTelemetry span; every request emits one JSON log
line (PII-masked question, signals, decision, latency, tokens, cost).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from copilot.abstention import AbstentionRule, resolve_rule
from copilot.config import Settings
from copilot.embeddings import Embedder, build_embedder
from copilot.guardrails import check_output, guard_documents, guard_input
from copilot.llm.base import SYSTEM_PROMPT, AnswerGenerator
from copilot.llm.extractive import ExtractiveGenerator
from copilot.observability import get_request_id, new_request_id, tracer
from copilot.retrieval.retriever import Retriever
from copilot.store import VectorStore, build_store

log = logging.getLogger("copilot.service")

ABSTAIN_MESSAGE = (
    "I could not find this in the indexed policies. Please rephrase, or check with Retail Credit Risk."
)
BLOCK_MESSAGE = (
    "This request was blocked by the copilot's safety checks (it asks to override instructions or to "
    "disclose data). Please ask a credit policy question."
)
WITHHELD_MESSAGE = (
    "An answer was generated but withheld because it failed the citation checks. Please consult the "
    "retrieved policy sections directly."
)


@dataclass
class AskResult:
    answer: str
    abstained: bool
    support_score: float
    citations: list[dict[str, Any]]
    contexts: list[dict[str, Any]]
    retrieval: dict[str, Any]
    generation: dict[str, Any]
    latency_ms: dict[str, float] = field(default_factory=dict)
    abstain_reason: str | None = None  # out_of_scope | not_found | blocked:<why> | output_check_failed
    guardrails: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    cost_usd: float | None = 0.0


def _snippet(text: str, n: int = 280) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1].rsplit(" ", 1)[0] + "…"


def _ms(t0: float, t1: float) -> float:
    return round((t1 - t0) * 1000, 1)


class CopilotService:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        embedder: Embedder,
        generator: AnswerGenerator,
        retriever: Retriever,
        abstention: AbstentionRule | None = None,
    ):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.generator = generator
        self.retriever = retriever
        self.abstention = abstention or resolve_rule(settings.abstain_rule_path, settings.abstain_threshold)

    @classmethod
    def from_settings(cls, s: Settings) -> CopilotService:
        store = build_store(s)
        embedder = build_embedder(s.embedder, s.embedding_model, s.embedding_dim)
        reranker = None
        if s.rerank:
            from copilot.retrieval.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(s.rerank_model)
        retriever = Retriever(store, embedder, s.candidate_k, s.rrf_k, reranker)
        generator: AnswerGenerator
        if s.llm_provider == "anthropic":
            from copilot.llm.providers import AnthropicGenerator

            generator = AnthropicGenerator(s.llm_model, s.llm_max_tokens, s.llm_timeout_s)
        elif s.llm_provider == "openai":
            from copilot.llm.providers import OpenAIGenerator

            generator = OpenAIGenerator(s.llm_model, s.llm_max_tokens, s.llm_timeout_s)
        else:
            generator = ExtractiveGenerator(embedder)
        return cls(s, store, embedder, generator, retriever)

    # ------------------------------------------------------------------ helpers
    def _abstain_reason(self, signals: dict[str, float | None]) -> str:
        dense_min = self.abstention.thresholds.get("dense", self.settings.abstain_threshold)
        dense = signals.get("dense")
        return "out_of_scope" if dense is not None and dense < dense_min else "not_found"

    def _cost(self, usage: dict[str, int], provider: str) -> float | None:
        if provider in ("extractive", "abstain", "blocked"):
            return 0.0
        pi, po = self.settings.llm_price_input_per_mtok, self.settings.llm_price_output_per_mtok
        if pi is None or po is None:
            return None
        return usage.get("input_tokens", 0) / 1e6 * pi + usage.get("output_tokens", 0) / 1e6 * po

    def _log(self, r: AskResult, question_masked: str) -> None:
        log.info(
            "ask",
            extra={
                "request_id": r.request_id,
                "fields": {
                    "event": "ask",
                    "question": question_masked,  # PII-masked; raw input is never logged
                    "abstained": r.abstained,
                    "abstain_reason": r.abstain_reason,
                    "guardrails": r.guardrails,
                    "signals": r.retrieval.get("signals"),
                    "mode": r.retrieval.get("mode"),
                    "reranked": r.retrieval.get("reranked"),
                    "citations": [f"{c['doc_id']}:{c['section']}" for c in r.citations],
                    "provider": r.generation.get("provider"),
                    "model": r.generation.get("model"),
                    "tokens": r.generation.get("usage") or {},
                    "cost_usd": r.cost_usd,
                    "latency_ms": r.latency_ms,
                },
            },
        )

    # ------------------------------------------------------------------ main entry
    def ask(
        self, question: str, top_k: int | None = None, mode: str | None = None, rerank: bool | None = None
    ) -> AskResult:
        rid = get_request_id() or new_request_id()
        k = top_k or self.settings.top_k
        mode = mode or self.settings.retrieval_mode
        lat: dict[str, float] = {}
        t_start = time.perf_counter()
        with tracer().start_as_current_span("copilot.ask") as root:
            root.set_attribute("request_id", rid)
            root.set_attribute("retrieval.mode", mode)
            result = self._ask(question, k, mode, rerank, rid, lat, root)
            lat["total"] = _ms(t_start, time.perf_counter())
            result.latency_ms = lat
            root.set_attribute("abstained", result.abstained)
            root.set_attribute("abstain_reason", result.abstain_reason or "")
        self._log(result, result.guardrails.get("input", {}).get("masked_question", ""))
        result.guardrails.get("input", {}).pop("masked_question", None)
        return result

    def _ask(self, question, k, mode, rerank, rid, lat, root) -> AskResult:
        guards: dict[str, Any] = {}
        # 1. input guardrails ---------------------------------------------------------------
        t0 = time.perf_counter()
        with tracer().start_as_current_span("guardrails.input") as sp:
            if self.settings.guardrails:
                decision = guard_input(question)
                query, masked = decision.question, decision.masked_question
                guards["input"] = decision.as_dict() | {"masked_question": masked}
                sp.set_attribute("action", decision.action)
                sp.set_attribute("pii", ",".join(decision.pii))
            else:
                query, masked = question, question
                guards["input"] = {"action": "disabled", "masked_question": question}
        lat["guardrails_input"] = _ms(t0, time.perf_counter())
        if guards["input"].get("action") == "block":
            return AskResult(
                BLOCK_MESSAGE,
                True,
                0.0,
                [],
                [],
                {"mode": mode, "signals": {}},
                {"provider": "blocked"},
                abstain_reason=f"blocked:{guards['input']['reason']}",
                guardrails=guards,
                request_id=rid,
            )

        # 2. retrieval (dense + bm25 + fusion + rerank spans are inside) ---------------------
        t0 = time.perf_counter()
        with tracer().start_as_current_span("retrieve") as sp:
            res = self.retriever.retrieve(query, k=k, mode=mode, rerank=rerank)  # type: ignore[arg-type]
            sp.set_attribute("items", len(res.items))
            sp.set_attribute("reranked", res.reranked)
        lat["retrieval"] = _ms(t0, time.perf_counter())
        signals = res.signals()
        contexts = [
            {
                "rank": i,
                "chunk_id": it.chunk.chunk_id,
                "doc_id": it.chunk.doc_id,
                "doc_title": it.chunk.doc_title,
                "section": it.chunk.section_path,
                "text": it.chunk.text,
                "scores": it.scores,
                "metadata": it.chunk.metadata,
            }
            for i, it in enumerate(res.items, start=1)
        ]
        retrieval_info = {
            "mode": res.mode,
            "reranked": res.reranked,
            "top_k": k,
            "max_vector_sim": round(res.max_vector_sim, 4),
            "signals": {n: (None if v is None else round(v, 4)) for n, v in signals.items()},
            "abstain_rule": self.abstention.describe(),
        }
        if self.abstention.needs_rerank and not res.reranked:
            retrieval_info["abstain_rule_degraded"] = "rerank signal unavailable (reranker not enabled)"

        # 3. abstention gate ----------------------------------------------------------------
        with tracer().start_as_current_span("abstain") as sp:
            abstain = (not res.items) or self.abstention.decide(signals)
            sp.set_attribute("abstain", abstain)
            for n, v in signals.items():
                if v is not None:
                    sp.set_attribute(f"signal.{n}", float(v))
        if abstain:
            return AskResult(
                ABSTAIN_MESSAGE,
                True,
                round(res.max_vector_sim, 4),
                [],
                contexts,
                retrieval_info,
                {"provider": "abstain"},
                abstain_reason=self._abstain_reason(signals),
                guardrails=guards,
                request_id=rid,
            )

        # 4. document guardrails ------------------------------------------------------------
        items = res.items
        if self.settings.guardrails:
            with tracer().start_as_current_span("guardrails.documents") as sp:
                dg = guard_documents(res.items)
                items = dg.contexts
                guards["documents"] = {"flagged_chunks": dg.flagged}
                sp.set_attribute("flagged", len(dg.flagged))

        # 5. generation ---------------------------------------------------------------------
        t0 = time.perf_counter()
        with tracer().start_as_current_span("generate") as sp:
            gen = self.generator.generate(query, items)
            sp.set_attribute("provider", gen.provider)
            sp.set_attribute("model", gen.model or "")
            for kk, vv in (gen.usage or {}).items():
                sp.set_attribute(f"tokens.{kk}", int(vv))
        lat["generation"] = _ms(t0, time.perf_counter())
        cost = self._cost(gen.usage or {}, gen.provider)
        root.set_attribute("cost_usd", -1.0 if cost is None else cost)
        generation = {"provider": gen.provider, "model": gen.model, "usage": gen.usage}

        # 6. output checks ------------------------------------------------------------------
        if self.settings.guardrails:
            with tracer().start_as_current_span("guardrails.output") as sp:
                oc = check_output(gen.answer, items, SYSTEM_PROMPT)
                guards["output"] = {"ok": oc.ok, "claims": oc.claims, "violations": oc.violations}
                sp.set_attribute("ok", oc.ok)
            if not oc.ok:
                log.warning("output check failed", extra={"fields": {"violations": oc.violations}})
                return AskResult(
                    WITHHELD_MESSAGE,
                    True,
                    round(res.max_vector_sim, 4),
                    [],
                    contexts,
                    retrieval_info,
                    generation,
                    abstain_reason="output_check_failed",
                    guardrails=guards,
                    request_id=rid,
                    cost_usd=cost,
                )

        citations = []
        for idx in gen.cited:
            c = res.items[idx - 1].chunk
            citations.append(
                {
                    "ref": idx,
                    "doc_id": c.doc_id,
                    "doc_title": c.doc_title,
                    "section": c.section_path,
                    "chunk_id": c.chunk_id,
                    "snippet": _snippet(c.text),
                    "page_start": c.metadata.get("page_start"),
                    "source_url": c.metadata.get("source_url"),
                }
            )
        return AskResult(
            gen.answer,
            False,
            round(res.max_vector_sim, 4),
            citations,
            contexts,
            retrieval_info,
            generation,
            guardrails=guards,
            request_id=rid,
            cost_usd=cost,
        )
