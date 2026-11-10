"""Guardrail policies used by the service.

Input:  PII is masked first (the masked text is what retrieval, the LLM and the logs see). Sentences that
        match an injection pattern are removed; if a legitimate question remains it is answered, otherwise
        the request is blocked. Data-exfiltration and secret-extraction requests are always blocked.
Docs:   retrieved chunks are untrusted. Instruction-like sentences inside them are replaced by a quoted,
        inert marker so neither the extractive generator nor an LLM treats them as instructions.
Output: every claim must cite a retrieved passage, and every number in a claim must appear in the
        passages it cites; violations withhold the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from copilot.guardrails.injection import (
    InjectionHit,
    find_injection,
    split_sentences,
    strip_injected_sentences,
)
from copilot.guardrails.pii import mask_pii
from copilot.models import ScoredChunk

ALWAYS_BLOCK = {"exfiltration"}
MIN_CONTENT_TOKENS = 3
QUOTE_OPEN, QUOTE_CLOSE = "[[quoted-untrusted-text: ", "]]"


@dataclass
class InputDecision:
    action: str  # "allow" | "sanitized" | "block"
    question: str  # text passed downstream (PII-masked, injection-stripped)
    masked_question: str  # PII-masked original (safe to log)
    pii: list[str] = field(default_factory=list)  # kinds found, e.g. ["cpf", "card"]
    injection: list[str] = field(default_factory=list)  # pattern kinds found
    reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "pii": self.pii,
            "injection": self.injection,
            "reason": self.reason,
        }


def _content_tokens(text: str) -> int:
    from copilot.retrieval.bm25 import tokenize

    return len(tokenize(text))


def guard_input(question: str) -> InputDecision:
    masked, pii = mask_pii(question)
    kinds = sorted({m.kind for m in pii})
    hits = find_injection(masked)
    if not hits:
        return InputDecision("allow", masked, masked, kinds)
    hit_kinds = sorted({h.kind for h in hits})
    if ALWAYS_BLOCK & set(hit_kinds):
        return InputDecision("block", "", masked, kinds, hit_kinds, "data_exfiltration")
    remainder, _ = strip_injected_sentences(masked)
    if remainder and not find_injection(remainder) and _content_tokens(remainder) >= MIN_CONTENT_TOKENS:
        return InputDecision("sanitized", remainder, masked, kinds, hit_kinds, "prompt_injection_removed")
    return InputDecision("block", "", masked, kinds, hit_kinds, "prompt_injection")


@dataclass
class DocumentGuardResult:
    contexts: list[ScoredChunk]
    flagged: dict[str, list[str]]  # chunk_id -> injection kinds


def neutralize(text: str) -> tuple[str, list[InjectionHit]]:
    out, hits = [], []
    for s in split_sentences(text):
        h = find_injection(s)
        if h:
            hits.extend(h)
            inert = s.strip().replace("[[", "[").replace("]]", "]")
            out.append(f"{QUOTE_OPEN}{inert}{QUOTE_CLOSE}")
        else:
            out.append(s)
    return " ".join(out) if hits else text, hits


def guard_documents(contexts: list[ScoredChunk]) -> DocumentGuardResult:
    safe, flagged = [], {}
    for it in contexts:
        text, hits = neutralize(it.chunk.text)
        if hits:
            flagged[it.chunk.chunk_id] = sorted({h.kind for h in hits})
            safe.append(ScoredChunk(replace(it.chunk, text=text), it.score, dict(it.scores)))
        else:
            safe.append(it)
    return DocumentGuardResult(safe, flagged)


_CITE = re.compile(r"\[(\d+)\]")
_NUM = re.compile(r"(?<![\w.])\d[\d,.]*\d|(?<![\w.])\d")
_HEADER = re.compile(r"^(relevant policy text|i could not find|no supporting passage found)", re.I)


_QUOTED_RE = re.compile(re.escape(QUOTE_OPEN) + r"(.*?)" + re.escape(QUOTE_CLOSE), re.S)


def strip_quoted(text: str) -> str:
    """Remove quoted-untrusted spans (added by `neutralize`) from a piece of text."""
    return re.sub(r"\s{2,}", " ", _QUOTED_RE.sub("", text)).strip()


def _norm_num(n: str) -> str:
    n = n.rstrip(".,")
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", n):  # 10,000 / 1,000.50
        n = n.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", n):  # 10.000 / 1.000,50 (pt-BR)
        n = n.replace(".", "").replace(",", ".")
    return n


def numbers_in(text: str) -> set[str]:
    return {_norm_num(m) for m in _NUM.findall(text)}


@dataclass
class OutputCheck:
    ok: bool
    violations: list[str]
    claims: int


def check_output(answer: str, contexts: list[ScoredChunk], system_prompt: str | None = None) -> OutputCheck:
    """Claims = answer lines/sentences (headers and the refusal string excluded)."""
    violations: list[str] = []
    claims = 0
    for raw in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", answer):
        line = raw.strip().lstrip("-• ").strip()
        if not line or _HEADER.match(line):
            continue
        claims += 1
        refs = [int(r) for r in _CITE.findall(line)]
        if not refs:
            violations.append(f"uncited claim: {line[:80]!r}")
            continue
        bad = [r for r in refs if not 1 <= r <= len(contexts)]
        if bad:
            violations.append(f"citation to non-retrieved passage {bad}: {line[:60]!r}")
            continue
        cited_text = " ".join(contexts[r - 1].chunk.text for r in refs)
        missing = numbers_in(_CITE.sub("", line)) - numbers_in(cited_text)
        if missing:
            violations.append(f"numbers not in cited passages {sorted(missing)}: {line[:60]!r}")
    # an answer must never reproduce document text that was quarantined as instruction-like
    for c in contexts:
        for q in _QUOTED_RE.findall(c.chunk.text):
            if len(q) >= 20 and q.strip() in answer:
                violations.append(f"echoed quarantined document text: {q[:50]!r}")
    if system_prompt:
        for sent in split_sentences(system_prompt):
            if len(sent) > 30 and sent.strip() in answer:
                violations.append("system prompt leaked")
                break
    return OutputCheck(not violations, violations, claims)
