"""Parse source files (Markdown, plain text, PDF) into documents made of heading-aware sections."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from copilot.models import ParsedDocument, Section

PARSER_VERSION = "1"

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_file(
    path: Path, doc_id: str | None = None, extra_meta: dict[str, Any] | None = None
) -> ParsedDocument:
    suffix = path.suffix.lower()
    raw = path.read_bytes()
    if suffix in {".md", ".markdown"}:
        doc = parse_markdown(raw.decode("utf-8"), source_path=str(path))
    elif suffix == ".txt":
        doc = parse_text(raw.decode("utf-8"), source_path=str(path))
    elif suffix == ".pdf":
        doc = parse_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    doc.content_hash = sha256_bytes(raw)
    if extra_meta:
        doc.metadata.update(extra_meta)
        doc.title = extra_meta.get("title", doc.title)
    if doc_id:
        doc.doc_id = doc_id
    if not doc.doc_id:
        doc.doc_id = path.stem
    return doc


def parse_markdown(text: str, source_path: str = "<memory>") -> ParsedDocument:
    meta: dict[str, Any] = {}
    m = _FRONT_MATTER.match(text)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        text = text[m.end() :]

    title = str(meta.get("title", ""))
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading)
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        path = tuple(h for _, h in stack) or ("Preamble",)
        sections.append(Section(path=path, text=body))

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        hm = None if in_fence else _MD_HEADING.match(line)
        if hm:
            level, heading = len(hm.group(1)), hm.group(2).strip()
            if level == 1 and not title:
                title = heading
            flush()
            if level == 1:
                stack.clear()
                continue  # the H1 is the document title, not part of the section path
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            buf.append(line)
    flush()

    return ParsedDocument(
        doc_id=str(meta.get("doc_id", "")),
        title=title or Path(source_path).stem,
        source_path=source_path,
        content_hash="",
        sections=sections,
        metadata={k: v for k, v in meta.items() if k not in {"doc_id", "title"}},
    )


def parse_text(text: str, source_path: str = "<memory>") -> ParsedDocument:
    """Plain text: a single section; blank lines separate paragraphs."""
    title = next((ln.strip() for ln in text.splitlines() if ln.strip()), Path(source_path).stem)
    return ParsedDocument(
        doc_id="",
        title=title[:200],
        source_path=source_path,
        content_hash="",
        sections=[Section(path=("Body",), text=text.strip())],
    )


def _clean_pdf_pages(pages: list[str]) -> list[str]:
    """Drop running headers/footers: short lines that repeat on many pages (digits normalized)."""
    norm = lambda s: re.sub(r"\d+", "#", s.strip())  # noqa: E731
    counts: Counter[str] = Counter()
    for p in pages:
        counts.update({norm(ln) for ln in p.splitlines() if ln.strip() and len(ln.strip()) < 90})
    threshold = max(3, int(len(pages) * 0.3))
    boiler = {k for k, c in counts.items() if c >= threshold}
    cleaned = []
    for p in pages:
        lines = [ln for ln in p.splitlines() if norm(ln) not in boiler]
        txt = "\n".join(lines)
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = re.sub(r"-\n(?=[a-z])", "", txt)  # de-hyphenate line breaks
        # join wrapped lines inside paragraphs, keep blank lines and bullets as breaks
        txt = re.sub(r"(?<![.:;!?\n])\n(?!\n|\s*[•\-–]\s|\s*\d+\.\s)", " ", txt)
        txt = re.sub(r" {2,}", " ", txt)
        cleaned.append(txt.strip())
    return cleaned


def parse_pdf(path: Path) -> ParsedDocument:
    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # font-encoding warnings are noise here
    reader = PdfReader(str(path))
    pages = [(pg.extract_text() or "") for pg in reader.pages]
    pages = _clean_pdf_pages(pages)

    # Map each page to the deepest outline (bookmark) entry that starts on or before it.
    entries: list[tuple[int, tuple[str, ...]]] = []

    def walk(items: list, parents: tuple[str, ...]) -> None:
        last: tuple[str, ...] | None = None
        for it in items:
            if isinstance(it, list):
                if last is not None:
                    walk(it, last)
                continue
            try:
                page_no = reader.get_destination_page_number(it)
            except Exception:
                continue
            title = re.sub(r"\s+", " ", str(it.title)).strip()
            last = (*parents, title)
            entries.append((page_no, last))

    try:
        walk(reader.outline, ())
    except Exception:
        entries = []
    # The single root bookmark is usually the document title; drop it from paths.
    roots = {p[0] for _, p in entries}
    if len(roots) == 1:
        entries = [(pg, p[1:]) for pg, p in entries if len(p) > 1]
    entries.sort(key=lambda e: e[0])

    title = (reader.metadata.title if reader.metadata and reader.metadata.title else "") or path.stem

    sections: list[Section] = []
    cur_path: tuple[str, ...] = ("Front matter",)
    cur_pages: list[tuple[int, str]] = []
    ei = 0

    def flush() -> None:
        if cur_pages and not (cur_path and cur_path[-1].lower() in {"contents", "table of contents"}):
            body = "\n\n".join(t for _, t in cur_pages if t)
            if body.strip():
                sections.append(Section(cur_path, body, cur_pages[0][0] + 1, cur_pages[-1][0] + 1))
        cur_pages.clear()

    for i, text in enumerate(pages):
        new_path = cur_path
        while ei < len(entries) and entries[ei][0] <= i:
            new_path = entries[ei][1]
            ei += 1
        if new_path != cur_path:
            flush()
            cur_path = new_path
        cur_pages.append((i, text))
    flush()

    return ParsedDocument(
        doc_id="",
        title=str(title),
        source_path=str(path),
        content_hash="",
        sections=sections,
        metadata={"pages": len(pages)},
    )
