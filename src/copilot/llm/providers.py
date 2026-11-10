"""Hosted LLM providers over plain HTTP (no vendor SDKs). Selected with COPILOT_LLM_PROVIDER."""

from __future__ import annotations

import os
import re

import httpx

from copilot.llm.base import SYSTEM_PROMPT, Generation, user_prompt
from copilot.models import ScoredChunk

_CITE = re.compile(r"\[(\d+)\]")


def _cited(text: str, n: int) -> list[int]:
    return sorted({int(m) for m in _CITE.findall(text) if 1 <= int(m) <= n})


class AnthropicGenerator:
    name = "anthropic"

    def __init__(self, model: str = "", max_tokens: int = 700, timeout: float = 60.0):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self.max_tokens = max_tokens
        self.client = httpx.Client(timeout=timeout)

    def generate(self, question: str, contexts: list[ScoredChunk]) -> Generation:
        r = self.client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt(question, contexts)}],
            },
        )
        r.raise_for_status()
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage", {})
        return Generation(
            text,
            _cited(text, len(contexts)),
            self.name,
            self.model,
            {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)},
        )


class OpenAIGenerator:
    name = "openai"

    def __init__(self, model: str = "", max_tokens: int = 700, timeout: float = 60.0):
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.max_tokens = max_tokens
        self.client = httpx.Client(timeout=timeout)

    def generate(self, question: str, contexts: list[ScoredChunk]) -> Generation:
        r = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt(question, contexts)},
                ],
            },
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        u = data.get("usage", {})
        return Generation(
            text,
            _cited(text, len(contexts)),
            self.name,
            self.model,
            {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)},
        )
