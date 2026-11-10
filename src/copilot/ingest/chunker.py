"""Section-aware chunking with stable, content-addressed chunk IDs.

Chunk ID format: ``{doc_id}::{section-slug}::{sha256(text)[:12]}``. The same text in the same
section of the same document always gets the same ID, so re-ingestion is idempotent and only
chunks whose text actually changed are re-embedded.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from copilot.models import Chunk, ParsedDocument, Section
from copilot.text import linearize_tables

CHUNKER_VERSION = "1"
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9§(\"“])")


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9.]+", "-", text.lower()).strip("-")
    return text[:60] or "section"


def section_slug(path: tuple[str, ...]) -> str:
    return "/".join(slugify(p) for p in path)


def make_chunk_id(doc_id: str, path: tuple[str, ...], text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}::{section_slug(path)}::{digest}"


def _blocks(text: str) -> list[str]:
    """Split into paragraphs; keep Markdown tables and bullet lists as single blocks."""
    blocks: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para:
            blocks.append(para)
    return blocks


def _split_long(block: str, max_chars: int) -> list[str]:
    if len(block) <= max_chars:
        return [block]
    if block.lstrip().startswith("|"):
        lines = block.splitlines()
        header, rows = lines[:2], lines[2:]
        out, cur = [], list(header)
        for row in rows:
            if len("\n".join([*cur, row])) > max_chars and len(cur) > 2:
                out.append("\n".join(cur))
                cur = list(header)
            cur.append(row)
        out.append("\n".join(cur))
        return out
    pieces = _SENT_SPLIT.split(block)
    out, cur = [], ""
    for s in pieces:
        while len(s) > max_chars:  # pathological sentence without breaks: hard split
            if cur:
                out.append(cur)
                cur = ""
            out.append(s[:max_chars])
            s = s[max_chars:]
        if cur and len(cur) + 1 + len(s) > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_section(section: Section, target_chars: int, max_chars: int) -> list[str]:
    units: list[str] = []
    for b in _blocks(section.text):
        units.extend(_split_long(b, max_chars))
    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + 2 + len(u) > target_chars:
            chunks.append(cur)
            cur = u
        else:
            cur = f"{cur}\n\n{u}" if cur else u
    if cur:
        chunks.append(cur)
    return chunks


def chunk_document(doc: ParsedDocument, target_chars: int = 1200, max_chars: int = 1800) -> list[Chunk]:
    out: list[Chunk] = []
    seen: dict[str, int] = {}
    ordinal = 0
    for section in doc.sections:
        for text in chunk_section(section, target_chars, max_chars):
            cid = make_chunk_id(doc.doc_id, section.path, text)
            if cid in seen:  # identical text twice in one section: disambiguate deterministically
                seen[cid] += 1
                cid = f"{cid}-{seen[cid]}"
            else:
                seen[cid] = 1
            meta = {"section_path": list(section.path)}
            if section.page_start is not None:
                meta["page_start"], meta["page_end"] = section.page_start, section.page_end
            for k in ("synthetic", "license", "language", "source_url"):
                if k in doc.metadata:
                    meta[k] = doc.metadata[k]
            out.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    section_path=" > ".join(section.path),
                    ordinal=ordinal,
                    text=text,
                    content_hash=hashlib.sha256(text.encode()).hexdigest(),
                    metadata=meta,
                )
            )
            ordinal += 1
    return out


def embedding_text(chunk: Chunk) -> str:
    """Text actually embedded: a contextual header (title + section) plus the chunk body."""
    return f"{chunk.doc_title} | {chunk.section_path}\n{linearize_tables(chunk.text)}"
