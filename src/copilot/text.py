"""Text helpers shared by chunking (embedding text) and extractive answering."""

from __future__ import annotations

import re

# Bump when embedding_text() output changes: stored indexes must then be rebuilt.
EMBED_TEXT_VERSION = "2"


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_rows(block: str) -> list[str]:
    """Turn a Markdown table into self-describing rows: "Header: value; Header: value."."""
    lines = [ln.strip() for ln in block.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return [block]
    header = _cells(lines[0])
    return [
        "; ".join(f"{h}: {v}" for h, v in zip(header, _cells(ln), strict=False)) + "." for ln in lines[2:]
    ]


def linearize_tables(text: str) -> str:
    """Replace Markdown tables with linearized rows (dense encoders handle prose better than pipes)."""
    out = []
    for block in re.split(r"\n\s*\n", text):
        out.append("\n".join(table_rows(block)) if block.lstrip().startswith("|") else block)
    return "\n\n".join(out)
