# ruff: noqa: E501  (regex table)
"""Prompt-injection heuristics for user input and for retrieved document text.

Written from generic attack patterns (instruction override, role play, system-prompt extraction,
delimiter smuggling, secret/data exfiltration), in English and Portuguese. Deliberately conservative:
a heuristic layer in front of an LLM that is itself instructed to treat documents as data. It is
not a complete defence (see docs/SECURITY.md).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def _fold(text: str) -> str:
    """Lowercase, strip accents, collapse spaces: 'Ignore   as instruções' -> 'ignore as instrucoes'."""
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t)


# (name, pattern over folded text)
_PATTERNS: list[tuple[str, str]] = [
    # --- instruction override (EN). Verbs/objects kept narrow: "can a manager override the DTI rule?" is legit.
    (
        "override",
        r"\b(ignore|disregard|forget|override)\b[^.?!\n]{0,40}\b(previous|prior|above|earlier|all|your|these|those)\b[^.?!\n]{0,20}\b(instructions?|rules|prompts?|guidelines|constraints|context)\b",
    ),
    ("override", r"\b(new|updated|real|actual)\s+(instructions?|rules?|system prompt)\s*[:\-]"),
    ("override", r"\bsystem\s+(override|message|instruction|update)\b"),
    ("override", r"\bfrom now on\b[^.?!\n]{0,40}\b(you|answer|respond|reply|say)\b"),
    # --- instruction override (PT, accents folded)
    (
        "override",
        r"\b(ignore|ignora|desconsidere|desconsidera|esqueca|despreze)\b[^.?!\n]{0,40}\b(instrucoes|regras|ordens|orientacoes|prompt|contexto)\b",
    ),
    ("override", r"\b(novas?)\s+(instrucoes|instrucao|regras?)\s*[:\-]"),
    ("override", r"\ba partir de agora\b[^.?!\n]{0,40}\b(voce|responda|diga)\b"),
    # --- role play / jailbreak
    ("roleplay", r"\byou are (now|no longer)\b"),
    (
        "roleplay",
        r"\b(act|behave|pretend|roleplay)\s+(as|like|to be)\b[^.?!\n]{0,40}\b(ai|assistant|model|dan|unrestricted|without (rules|restrictions|filters))\b",
    ),
    ("roleplay", r"\b(dan|developer mode|jailbreak|jailbroken)\b"),
    (
        "roleplay",
        r"\bvoce (agora )?e (um|uma)\b[^.?!\n]{0,30}\b(ia|assistente|modelo)\b[^.?!\n]{0,30}\b(sem|livre)\b",
    ),
    ("roleplay", r"\b(finja|aja como|faca de conta)\b"),
    ("roleplay", r"\bvoce (agora e|nao e mais)\b"),  # PT parity with "you are now / no longer" (added in M3)
    # --- prompt / secret extraction (needs a disclosure verb: "what is the password policy?" is legit)
    (
        "extraction",
        r"\b(print|reveal|show|repeat|output|display|leak|dump|tell me)\b[^.?!\n]{0,30}\b(system prompt|your prompt|your instructions|hidden instructions|initial prompt)\b",
    ),
    ("extraction", r"\b(revele|mostre|imprima|repita)\b[^.?!\n]{0,30}\b(prompt|instrucoes)\b"),
    (
        "extraction",
        r"\b(reveal|print|show|give|tell|list|send|share|leak|dump|output|expose|revele|mostre|imprima|envie|informe)\b[^.?!\n]{0,40}\b(api[_ ]?keys?|secret keys?|access tokens?|passwords?|environment variables|env vars|credentials|chaves? de api|senhas?|variaveis de ambiente|tokens? de acesso)\b",
    ),
    # --- forced output / authority claims
    (
        "forced_output",
        r"\b(reply|respond|answer|say|output|write)\s+(only|just|exactly)?\s*(with\s+)?(the word|the phrase)\b",
    ),
    ("forced_output", r"\b(responda|diga|escreva)\s+(apenas|somente|so)\b"),
    ("forced_output", r"\bappend\b[^.?!\n]{0,20}\b(the word|the phrase|the text)\b"),
    ("authority", r"\bi (authorize|order|instruct|command) you\b"),
    ("authority", r"\b(eu )?(autorizo|ordeno) (voce|que voce)\b"),
    # --- delimiter smuggling
    ("delimiter", r"</?\s*(context|system|document|instructions?|passages?)\s*>"),
    ("delimiter", r"\[/?(inst|system)\]|<\|im_(start|end)\|>|###\s*(system|instruction)"),
    # --- bulk data exfiltration
    (
        "exfiltration",
        r"\b(list|give|show|export|dump|send)\b[^.?!\n]{0,40}\b(customers?|clients?|cardholders?|applicants?)\b[^.?!\n]{0,40}\b(names?|card numbers?|cpfs?|account numbers?|ssns?|phone numbers?)\b",
    ),
    (
        "exfiltration",
        r"\b(list|give|show|export|dump|send)\b[^.?!\n]{0,40}\b(names?|card numbers?|cpfs?|account numbers?|ssns?|phone numbers?|emails?|addresses)\b[^.?!\n]{0,40}\b(of|for|from)\b[^.?!\n]{0,20}\b(customers?|clients?|cardholders?|applicants?|borrowers?)\b",
    ),
    (
        "exfiltration",
        r"\b(liste|lista|exporte|mostre|envie)\b[^.?!\n]{0,40}\b(nomes?|cartoes|cpfs?|contas?|telefones?|enderecos)\b[^.?!\n]{0,40}\b(dos?|das?)\b[^.?!\n]{0,20}\b(clientes?|portadores?|tomadores?)\b",
    ),
    (
        "exfiltration",
        r"\b(liste|lista|exporte|mostre|envie)\b[^.?!\n]{0,40}\b(clientes?|portadores?)\b[^.?!\n]{0,40}\b(nomes?|cartoes|cpfs?|numeros de conta|telefones?)\b",
    ),
]
_COMPILED = [(n, re.compile(p)) for n, p in _PATTERNS]
_SENT_SPLIT = re.compile(r"(?<=[.!?\n])\s+|(?=</?\s*(?:context|system)\s*>)")


@dataclass(frozen=True)
class InjectionHit:
    kind: str
    fragment: str


def find_injection(text: str) -> list[InjectionHit]:
    folded = _fold(text)
    hits = []
    for name, rx in _COMPILED:
        m = rx.search(folded)
        if m:
            hits.append(InjectionHit(name, m.group(0)[:80]))
    return hits


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text) if s and s.strip()]


def strip_injected_sentences(text: str) -> tuple[str, list[InjectionHit]]:
    """Drop sentences that contain an injection pattern; keep the rest of the question."""
    kept, hits = [], []
    for s in split_sentences(text):
        h = find_injection(s)
        if h:
            hits.extend(h)
        else:
            kept.append(s.strip())
    return " ".join(kept).strip(), hits
