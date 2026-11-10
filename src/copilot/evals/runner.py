"""Evaluation runner: per-question records for every retriever + the end-to-end system, then aggregates."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from copilot.evals.golden import GoldItem, requirement_matches
from copilot.evals.metrics import (
    abstention_metrics,
    canary_violations,
    citation_metrics,
    cost_usd,
    key_fact_recall,
    mean,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)

K_EVAL = 10


def chunk_hits(item: GoldItem, doc_id: str, section: str) -> set[int]:
    return {i for i, req in enumerate(item.requirements) if requirement_matches(req, doc_id, section)}


@dataclass
class RetrieverSpec:
    name: str
    mode: str
    retriever: Any  # copilot.retrieval.retriever.Retriever
    rerank: bool = False


def run_retrievers(item: GoldItem, specs: list[RetrieverSpec]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for spec in specs:
        t0 = time.perf_counter()
        res = spec.retriever.retrieve(item.question, k=K_EVAL, mode=spec.mode, rerank=spec.rerank)
        ms = (time.perf_counter() - t0) * 1000
        top = [
            {"doc_id": it.chunk.doc_id, "section": it.chunk.section_path, "chunk_id": it.chunk.chunk_id}
            for it in res.items
        ]
        out[spec.name] = {
            "top": top,
            "hits": [sorted(chunk_hits(item, t["doc_id"], t["section"])) for t in top],
            "latency_ms": round(ms, 2),
            "signals": res.signals(),
        }
    return out


def run_system(
    item: GoldItem,
    service: Any,
    judge: Any | None,
    price_in: float | None,
    price_out: float | None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = service.ask(item.question)
    ms = (time.perf_counter() - t0) * 1000
    cited = [{"doc_id": c["doc_id"], "section": c["section"]} for c in r.citations]
    rec: dict[str, Any] = {
        "abstained": r.abstained,
        "answer": r.answer,
        "citations": cited,
        "signals": r.retrieval.get("signals", {}),
        "latency_ms": round(ms, 2),
        "provider": r.generation.get("provider"),
        "abstain_reason": getattr(r, "abstain_reason", None),
        "guard_input": (getattr(r, "guardrails", {}) or {}).get("input", {}).get("action"),
        "pii": (getattr(r, "guardrails", {}) or {}).get("input", {}).get("pii", []),
        "flagged_chunks": len(
            ((getattr(r, "guardrails", {}) or {}).get("documents") or {}).get("flagged_chunks", {})
        ),
        "output_ok": ((getattr(r, "guardrails", {}) or {}).get("output") or {}).get("ok"),
    }
    usage = r.generation.get("usage") or {}
    tin, tout = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    rec["tokens"] = {"input": tin, "output": tout}
    if r.generation.get("provider") in ("extractive", "abstain", "blocked"):
        rec["cost_usd"] = 0.0
    else:
        rec["cost_usd"] = cost_usd(tin, tout, price_in, price_out)
    rec["canary_violations"] = canary_violations(r.answer, item.must_not_contain)
    if item.answerable and not r.abstained:
        hits = [chunk_hits(item, c["doc_id"], c["section"]) for c in cited]
        rec["citation"] = citation_metrics(hits, len(item.requirements))
        rec["key_fact_recall"] = key_fact_recall(r.answer, item.key_facts) if item.key_facts else None
        if judge is not None:
            rec["faithfulness"] = judge.faithfulness(r.answer, [c["text"] for c in r.contexts])
            rec["correctness"] = judge.correctness(item.question, r.answer, item.gold_answer)
    return rec


def _retrieval_agg(records: list[dict], name: str) -> dict[str, Any]:
    ans = [r for r in records if r["answerable"]]
    lat = [r["retrievers"][name]["latency_ms"] for r in records]
    out: dict[str, Any] = {"n": len(ans)}
    if ans:
        hits = [[set(h) for h in r["retrievers"][name]["hits"]] for r in ans]
        nreq = [r["n_required"] for r in ans]
        for k in (1, 5, 10):
            out[f"recall@{k}"] = mean(recall_at_k(h, n, k) for h, n in zip(hits, nreq, strict=True))
        out["mrr"] = mean(reciprocal_rank(h, K_EVAL) for h in hits)
        out["ndcg@10"] = mean(ndcg_at_k(h, n, 10) for h, n in zip(hits, nreq, strict=True))
    if lat:
        out["latency_p50_ms"] = percentile(lat, 50)
        out["latency_p95_ms"] = percentile(lat, 95)
    return out


def _system_agg(records: list[dict]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    sys_ = [r["system"] for r in records]
    out: dict[str, Any] = {"n": len(records)}
    out["abstention"] = abstention_metrics(
        [not r["answerable"] for r in records], [s["abstained"] for s in sys_]
    )
    answered = [r for r in records if r["answerable"] and not r["system"]["abstained"]]
    out["n_answered_answerable"] = len(answered)
    for m in ("hit", "precision", "recall"):
        out[f"citation_{m}"] = mean(r["system"]["citation"][m] for r in answered)
    kf = [r["system"]["key_fact_recall"] for r in answered if r["system"].get("key_fact_recall") is not None]
    out["key_fact_recall_answered"] = mean(kf)
    # end-to-end "answered correctly" proxy: answerable, not abstained, all key facts present, cites gold
    ok = [
        r
        for r in answered
        if r["system"].get("key_fact_recall") == 1.0 and r["system"]["citation"]["hit"] == 1.0
    ]
    n_ans = sum(1 for r in records if r["answerable"])
    out["answerable_fully_supported_rate"] = len(ok) / n_ans if n_ans else None
    canary_items = [r for r in records if r["must_not_contain"]]
    out["canary_items"] = len(canary_items)
    out["canary_violation_rate"] = (
        mean(1.0 if r["system"]["canary_violations"] else 0.0 for r in canary_items) if canary_items else None
    )
    reasons: dict[str, int] = {}
    for s in sys_:
        if s["abstained"]:
            key = s.get("abstain_reason") or "unknown"
            reasons[key] = reasons.get(key, 0) + 1
    out["abstain_reasons"] = dict(sorted(reasons.items()))
    out["input_guard_actions"] = {
        a: sum(1 for s in sys_ if s.get("guard_input") == a) for a in ("allow", "sanitized", "block")
    }
    out["output_check_failures"] = sum(1 for s in sys_ if s.get("output_ok") is False)
    lat = [s["latency_ms"] for s in sys_]
    out["latency_p50_ms"] = percentile(lat, 50)
    out["latency_p95_ms"] = percentile(lat, 95)
    costs = [s["cost_usd"] for s in sys_]
    out["cost_per_query_usd"] = None if any(c is None for c in costs) else mean(costs)
    out["tokens_per_query"] = mean(s["tokens"]["input"] + s["tokens"]["output"] for s in sys_)
    for m in ("faithfulness", "correctness"):
        vals = [r["system"][m] for r in answered if r["system"].get(m) is not None]
        out[m] = mean(vals) if vals else None
    return out


def aggregate(
    records: list[dict], retriever_names: Iterable[str], group: Callable[[dict], str] | None = None
) -> dict[str, Any]:
    names = list(retriever_names)
    if group is None:
        return {
            "retrieval": {n: _retrieval_agg(records, n) for n in names},
            "system": _system_agg(records),
        }
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(group(r), []).append(r)
    return {g: aggregate(rs, names) for g, rs in sorted(groups.items())}
