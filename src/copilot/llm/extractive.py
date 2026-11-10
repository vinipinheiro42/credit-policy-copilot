"""No-LLM answer mode: return the passages that best support the question, verbatim.

Runs fully offline. It never paraphrases, so it cannot hallucinate numbers, but it also cannot
synthesize across passages; it is the baseline every LLM provider must beat in the evals.
"""

from __future__ import annotations

import re

import numpy as np

from copilot.embeddings import Embedder
from copilot.guardrails.core import strip_quoted
from copilot.llm.base import Generation
from copilot.models import ScoredChunk
from copilot.retrieval.bm25 import tokenize
from copilot.text import table_rows

_SENT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9§(\"“])")


def segment(text: str) -> list[str]:
    """Split a chunk into answerable units: sentences, bullet items, or table rows with headers."""
    units: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if block.startswith("|"):
            units.extend(table_rows(block))
            continue
        lines = block.splitlines()
        if all(ln.strip().startswith(("- ", "* ")) for ln in lines[1:]) and len(lines) > 1:
            lead = lines[0].strip()
            units.append(lead)
            units.extend(
                f"{lead} {ln.strip()[2:]}" if lead.endswith(":") else ln.strip()[2:] for ln in lines[1:]
            )
            continue
        units.extend(s.strip() for s in _SENT.split(" ".join(ln.strip() for ln in lines)) if s.strip())
    return [u for u in units if len(u) > 15]


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, embedder: Embedder, max_units: int = 3, from_top: int = 3):
        self.embedder = embedder
        self.max_units = max_units
        self.from_top = from_top

    def generate(self, question: str, contexts: list[ScoredChunk]) -> Generation:
        cands: list[tuple[int, int, str]] = []  # (context index, unit position, text)
        for ci, it in enumerate(contexts[: self.from_top]):
            for ui, u in enumerate(segment(it.chunk.text)):
                # instruction-like document text is quoted data: never part of an answer
                u = strip_quoted(u)
                if len(u) >= 8:
                    cands.append((ci, ui, u))
        if not cands:
            return Generation("No supporting passage found.", [], self.name)

        qv = self.embedder.embed_query(question)
        mat = self.embedder.embed_documents([u for _, _, u in cands])
        dense = mat @ qv
        q_terms = set(tokenize(question))
        lex = np.array([len(q_terms & set(tokenize(u))) / max(1, len(q_terms)) for _, _, u in cands])
        rank_prior = np.array([1.0 / (1 + ci) for ci, _, _ in cands]) * 0.05
        score = dense + 0.3 * lex + rank_prior

        chosen = sorted(np.argsort(-score)[: self.max_units], key=lambda i: (cands[i][0], cands[i][1]))
        lines, cited = [], []
        for i in chosen:
            ci, _, u = cands[i]
            lines.append(f"- {u} [{ci + 1}]")
            if ci + 1 not in cited:
                cited.append(ci + 1)
        answer = "Relevant policy text (extractive mode, no LLM):\n" + "\n".join(lines)
        return Generation(answer, sorted(cited), self.name, "extractive-v1")
