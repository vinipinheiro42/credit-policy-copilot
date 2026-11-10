"""Pure metric functions. Inputs are plain lists so every metric is unit-testable on tiny examples.

Retrieval relevance is defined per *requirement* (a fact the answer needs, satisfiable by any of its
gold locations). ``hits[i]`` is the set of requirement indexes satisfied by the chunk at rank i+1.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(hits: Sequence[set[int]], n_required: int, k: int) -> float:
    """Fraction of requirements satisfied by at least one chunk in the top k."""
    if n_required == 0:
        raise ValueError("recall undefined without requirements")
    covered: set[int] = set()
    for h in hits[:k]:
        covered |= h
    return len(covered) / n_required


def reciprocal_rank(hits: Sequence[set[int]], k: int | None = None) -> float:
    """1/rank of the first chunk that satisfies any requirement (0 if none in the top k)."""
    for rank, h in enumerate(hits[:k] if k else hits, start=1):
        if h:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(hits: Sequence[set[int]], n_required: int, k: int = 10) -> float:
    """Binary-gain nDCG where only the FIRST chunk satisfying a requirement earns gain.

    Duplicated chunks for an already-satisfied requirement earn nothing, so the ideal ranking places
    one chunk per requirement at the top: IDCG = sum_{i=1..min(n,k)} 1/log2(i+1).
    """
    if n_required == 0:
        raise ValueError("nDCG undefined without requirements")
    seen: set[int] = set()
    dcg = 0.0
    for rank, h in enumerate(hits[:k], start=1):
        new = h - seen
        if new:
            dcg += 1.0 / math.log2(rank + 1)
            seen |= new
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_required, k) + 1))
    return dcg / idcg


def abstention_metrics(should_abstain: Sequence[bool], abstained: Sequence[bool]) -> dict[str, float | int]:
    """Positive class = "should abstain". false_refusal_rate = abstained on answerable / answerable."""
    if len(should_abstain) != len(abstained):
        raise ValueError("length mismatch")
    tp = sum(1 for y, p in zip(should_abstain, abstained, strict=True) if y and p)
    fp = sum(1 for y, p in zip(should_abstain, abstained, strict=True) if not y and p)
    fn = sum(1 for y, p in zip(should_abstain, abstained, strict=True) if y and not p)
    n_answerable = sum(1 for y in should_abstain if not y)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_refusal_rate": fp / n_answerable if n_answerable else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_should_abstain": tp + fn,
        "n_answerable": n_answerable,
    }


def citation_metrics(cited_hits: Sequence[set[int]], n_required: int) -> dict[str, float]:
    """For one answered question: cited_hits[i] = requirements satisfied by the i-th cited chunk.

    - hit: at least one citation points to a gold section.
    - precision: fraction of citations that point to a gold section.
    - recall: fraction of requirements covered by the citations.
    No citations -> all zero (an answer without sources is a citation failure).
    """
    if not cited_hits:
        return {"hit": 0.0, "precision": 0.0, "recall": 0.0}
    good = sum(1 for h in cited_hits if h)
    covered: set[int] = set().union(*cited_hits)
    return {
        "hit": 1.0 if good else 0.0,
        "precision": good / len(cited_hits),
        "recall": len(covered) / n_required if n_required else 0.0,
    }


def percentile(values: Iterable[float], q: float) -> float:
    """Linear-interpolated percentile (same as numpy's default), q in [0, 100]."""
    v = sorted(values)
    if not v:
        raise ValueError("percentile of empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    pos = (len(v) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def mean(values: Iterable[float]) -> float | None:
    v = list(values)
    return sum(v) / len(v) if v else None


def key_fact_recall(answer: str, key_facts: Sequence[str]) -> float:
    """Lexical proxy for correctness: fraction of key facts present (case/space-insensitive)."""
    from copilot.evals.golden import normalize_answer

    if not key_facts:
        raise ValueError("no key facts")
    a = normalize_answer(answer)
    return sum(1 for kf in key_facts if normalize_answer(kf) in a) / len(key_facts)


def canary_violations(answer: str, must_not_contain: Sequence[str]) -> list[str]:
    """Case-sensitive on purpose: canaries are distinctive tokens (PWNED, system-prompt sentences)."""
    return [m for m in must_not_contain if m in answer]


def cost_usd(
    input_tokens: int, output_tokens: int, price_in_per_mtok: float | None, price_out_per_mtok: float | None
) -> float | None:
    """None when prices are not configured (reported as unknown, never guessed)."""
    if price_in_per_mtok is None or price_out_per_mtok is None:
        return None
    return input_tokens / 1e6 * price_in_per_mtok + output_tokens / 1e6 * price_out_per_mtok
