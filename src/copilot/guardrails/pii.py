"""PII detection and masking for user input and logs (runs before retrieval, the LLM and logging).

Entities: CPF and CNPJ (validated with their check digits, or any CPF/CNPJ-shaped number right after
the keyword, so a mistyped CPF is still masked), payment card numbers (13-19 digits, Luhn),
email, phone numbers (BR and NANP formats), and bank account numbers (digits next to an account
keyword in EN/PT, incl. "agência/conta"). Check-digit validation keeps policy numbers such as
"USD 10,000" or "Art. 49" from being masked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PIIMatch:
    kind: str
    start: int
    end: int
    text: str


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def luhn_ok(number: str) -> bool:
    d = _digits(number)
    if not 13 <= len(d) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def cpf_ok(value: str) -> bool:
    d = _digits(value)
    if len(d) != 11 or d == d[0] * 11:
        return False
    for n in (9, 10):
        s = sum(int(d[i]) * (n + 1 - i) for i in range(n))
        if (s * 10 % 11) % 10 != int(d[n]):
            return False
    return True


def cnpj_ok(value: str) -> bool:
    d = _digits(value)
    if len(d) != 14 or d == d[0] * 14:
        return False
    w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    w2 = [6, *w1]
    for w, n in ((w1, 12), (w2, 13)):
        r = sum(int(d[i]) * w[i] for i in range(n)) % 11
        if (0 if r < 2 else 11 - r) != int(d[n]):
            return False
    return True


_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CNPJ = re.compile(r"(?<!\d)\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}(?!\d)")
_CPF = re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_PHONE = re.compile(
    r"(?<![\w.])(?:\+?55[\s.-]?)?\(?\d{2}\)?[\s.-]?9?\d{4}[\s.-]?\d{4}(?!\d)"  # BR: (11) 91234-5678
    r"|(?<![\w.])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"  # NANP: (555) 123-4567
)
_ACCOUNT = re.compile(
    r"(?i)\b(?:account(?:\s+(?:number|no\.?|#))?|acct\.?|a/c|conta(?:\s+corrente)?|ag[eê]ncia(?:\s*/\s*conta)?|iban)"
    r"\s*(?:n[ºo°.]*\s*)?[:#]?\s*((?:\d[\d\s./-]{3,}\d)|[A-Z]{2}\d{2}[A-Z0-9]{8,30})"
)

PLACEHOLDER = {
    "cpf": "[CPF]",
    "cnpj": "[CNPJ]",
    "card": "[CARD]",
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "account": "[ACCOUNT]",
}


def detect_pii(text: str) -> list[PIIMatch]:
    """Non-overlapping matches, highest-confidence kinds first (validated numbers beat phone shapes)."""
    found: list[PIIMatch] = []

    def add(kind: str, start: int, end: int) -> None:
        if any(not (end <= m.start or start >= m.end) for m in found):
            return
        found.append(PIIMatch(kind, start, end, text[start:end]))

    for m in _EMAIL.finditer(text):
        add("email", m.start(), m.end())

    def keyword_before(start: int, kw: str) -> bool:
        return kw in text[max(0, start - 12) : start].lower()

    for m in _CNPJ.finditer(text):
        if cnpj_ok(m.group()) or keyword_before(m.start(), "cnpj"):
            add("cnpj", m.start(), m.end())
    for m in _CPF.finditer(text):
        if cpf_ok(m.group()) or keyword_before(m.start(), "cpf"):
            add("cpf", m.start(), m.end())
    for m in _CARD.finditer(text):
        if luhn_ok(m.group()):
            add("card", m.start(), m.end())
    for m in _ACCOUNT.finditer(text):
        add("account", m.start(1), m.end(1))
    for m in _PHONE.finditer(text):
        if len(_digits(m.group())) >= 10:
            add("phone", m.start(), m.end())
    return sorted(found, key=lambda m: m.start)


def mask_pii(text: str) -> tuple[str, list[PIIMatch]]:
    matches = detect_pii(text)
    out, last = [], 0
    for m in matches:
        out.append(text[last : m.start])
        out.append(PLACEHOLDER[m.kind])
        last = m.end
    out.append(text[last:])
    return "".join(out), matches
