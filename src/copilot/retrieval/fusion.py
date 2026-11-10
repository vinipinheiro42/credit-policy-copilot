"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009)."""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists of IDs. score(d) = sum over lists of 1 / (k + rank), rank starting at 1.

    Ties are broken by best (lowest) rank in any list, then by ID, so the output is deterministic.
    """
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in rankings:
        unique = list(dict.fromkeys(ranking))  # drop duplicates inside one list, keep first rank
        for rank, doc_id in enumerate(unique, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            best[doc_id] = min(best.get(doc_id, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best[kv[0]], kv[0]))
