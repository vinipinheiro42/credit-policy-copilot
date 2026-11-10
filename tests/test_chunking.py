from copilot.ingest.chunker import chunk_document, make_chunk_id, section_slug
from copilot.ingest.parsers import parse_markdown

MD = """---
doc_id: D1
title: Doc One
---

# Doc One

> preamble note

## 1. Scope

Applies to cards.

## 2. Limits

### 2.1 Caps

Grade A is capped at USD 25,000.

| Grade | Cap |
|---|---|
| A | 25,000 |
"""


def test_markdown_sections_follow_heading_hierarchy():
    doc = parse_markdown(MD)
    assert doc.doc_id == "D1" and doc.title == "Doc One"
    paths = [s.path for s in doc.sections]
    assert paths == [("Preamble",), ("1. Scope",), ("2. Limits", "2.1 Caps")]


def test_chunk_ids_are_stable_and_contain_doc_and_section():
    a = chunk_document(parse_markdown(MD))
    b = chunk_document(parse_markdown(MD))
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    cap = next(c for c in a if "Caps" in c.section_path)
    assert cap.chunk_id.startswith("D1::2.-limits/2.1-caps::")
    assert len(cap.chunk_id.rsplit("::", 1)[1]) == 12
    assert cap.section_path == "2. Limits > 2.1 Caps"


def test_editing_one_section_changes_only_its_chunk_ids():
    before = {c.section_path: c.chunk_id for c in chunk_document(parse_markdown(MD))}
    after = {
        c.section_path: c.chunk_id
        for c in chunk_document(
            parse_markdown(MD.replace("Applies to cards.", "Applies to cards and loans."))
        )
    }
    assert before["1. Scope"] != after["1. Scope"]
    assert before["2. Limits > 2.1 Caps"] == after["2. Limits > 2.1 Caps"]


def test_tables_are_not_split_from_their_section():
    chunks = chunk_document(parse_markdown(MD))
    cap = next(c for c in chunks if "Caps" in c.section_path)
    assert "| A | 25,000 |" in cap.text


def test_long_sections_respect_max_chars_and_keep_order():
    body = " ".join(f"Sentence number {i} explains a rule." for i in range(400))
    md = f"---\ndoc_id: L\n---\n# L\n\n## Big\n\n{body}\n"
    chunks = chunk_document(parse_markdown(md), target_chars=500, max_chars=800)
    assert len(chunks) > 5
    assert all(len(c.text) <= 800 for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_duplicate_text_in_same_section_gets_distinct_ids():
    md = "---\ndoc_id: X\n---\n# X\n\n## S\n\n" + ("Same paragraph. " * 60 + "\n\n") * 2
    chunks = chunk_document(parse_markdown(md), target_chars=200, max_chars=2000)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_slug_is_ascii_and_deterministic():
    assert section_slug(("CAPÍTULO I — DO OBJETO", "Art. 1º")) == "capitulo-i-do-objeto/art.-1o"
    assert make_chunk_id("D", ("A",), "t") == make_chunk_id("D", ("A",), "t")
