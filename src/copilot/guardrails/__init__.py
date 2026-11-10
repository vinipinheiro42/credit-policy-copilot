"""Guardrails: input (PII masking, prompt injection), retrieved documents (untrusted text), output checks."""

from copilot.guardrails.core import (
    DocumentGuardResult,
    InputDecision,
    OutputCheck,
    check_output,
    guard_documents,
    guard_input,
)
from copilot.guardrails.pii import detect_pii, mask_pii

__all__ = [
    "DocumentGuardResult",
    "InputDecision",
    "OutputCheck",
    "check_output",
    "detect_pii",
    "guard_documents",
    "guard_input",
    "mask_pii",
]
