"""LLM-as-judge for faithfulness and answer correctness. Runs ONLY when an API key is present.

Without ANTHROPIC_API_KEY / OPENAI_API_KEY the harness reports these metrics as "not run" instead of
estimating them. The completion function is injectable so the parsing/aggregation logic is tested
offline with a fake judge (tests/test_judge.py).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

Complete = Callable[[str, str], tuple[str, dict[str, int]]]  # (system, user) -> (text, usage)

FAITHFULNESS_SYSTEM = """You are a strict evaluator of a retrieval-augmented assistant for bank credit policy.
Given numbered CONTEXT passages and an ANSWER, list each factual claim in the ANSWER and decide whether it is
fully supported by the CONTEXT (no outside knowledge). Ignore citation markers like [1].
Reply with JSON only: {"claims": [{"claim": str, "supported": true|false}], "faithfulness": float in [0,1]}
where faithfulness = supported claims / total claims (1.0 if the answer is a refusal with no claims)."""

CORRECTNESS_SYSTEM = """You grade answers to bank credit-policy questions against a reference answer.
Judge only whether the ANSWER states the same facts as the REFERENCE (numbers, thresholds, approvers,
conditions). Extra correct detail is fine; any contradiction or missing key condition lowers the grade.
Reply with JSON only: {"verdict": "correct"|"partial"|"incorrect", "score": 1.0|0.5|0.0, "reason": str}"""

_JSON = re.compile(r"\{.*\}", re.S)


def detect_provider() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def _anthropic_complete(model: str, timeout: float) -> Complete:
    client = httpx.Client(timeout=timeout)
    key = os.environ["ANTHROPIC_API_KEY"]

    def complete(system: str, user: str) -> tuple[str, dict[str, int]]:
        r = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={
                "model": model,
                "max_tokens": 800,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        r.raise_for_status()
        d = r.json()
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage", {})
        return text, {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)}

    return complete


def _openai_complete(model: str, timeout: float) -> Complete:
    client = httpx.Client(timeout=timeout)
    key = os.environ["OPENAI_API_KEY"]
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def complete(system: str, user: str) -> tuple[str, dict[str, int]]:
        r = client.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "max_tokens": 800,
                "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {})
        return d["choices"][0]["message"]["content"] or "", {
            "input_tokens": u.get("prompt_tokens", 0),
            "output_tokens": u.get("completion_tokens", 0),
        }

    return complete


def parse_json(text: str) -> dict:
    m = _JSON.search(text)
    if not m:
        raise ValueError(f"judge returned no JSON: {text[:120]!r}")
    return json.loads(m.group(0))


@dataclass
class JudgeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    errors: list[str] = field(default_factory=list)


class LLMJudge:
    def __init__(self, complete: Complete, provider: str, model: str):
        self.complete = complete
        self.provider = provider
        self.model = model
        self.usage = JudgeUsage()

    @classmethod
    def from_env(cls, model: str = "", timeout: float = 60.0) -> LLMJudge | None:
        provider = detect_provider()
        if provider == "anthropic":
            m = model or os.environ.get("COPILOT_JUDGE_MODEL", "claude-sonnet-4-5")
            return cls(_anthropic_complete(m, timeout), provider, m)
        if provider == "openai":
            m = model or os.environ.get("COPILOT_JUDGE_MODEL", "gpt-4o-mini")
            return cls(_openai_complete(m, timeout), provider, m)
        return None

    def _call(self, system: str, user: str) -> dict | None:
        try:
            text, u = self.complete(system, user)
            self.usage.calls += 1
            self.usage.input_tokens += u.get("input_tokens", 0)
            self.usage.output_tokens += u.get("output_tokens", 0)
            return parse_json(text)
        except Exception as e:  # a judge failure must not abort the whole eval run
            self.usage.errors.append(f"{type(e).__name__}: {e}"[:200])
            return None

    def faithfulness(self, answer: str, contexts: list[str]) -> float | None:
        ctx = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
        d = self._call(FAITHFULNESS_SYSTEM, f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}")
        if d is None:
            return None
        claims = d.get("claims") or []
        if claims:
            return sum(1 for c in claims if c.get("supported")) / len(claims)
        return float(d.get("faithfulness", 1.0))

    def correctness(self, question: str, answer: str, reference: str) -> float | None:
        d = self._call(CORRECTNESS_SYSTEM, f"QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {answer}")
        if d is None:
            return None
        verdict = str(d.get("verdict", "")).lower()
        return {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(verdict, float(d.get("score", 0.0)))
