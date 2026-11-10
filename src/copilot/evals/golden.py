"""Golden set: schema, loading, section resolution and evidence verification against the corpus.

Gold labels are ``doc_id`` + section path. A label section matches a parsed section path when it is
equal to it or is a ``" > "``-delimited suffix of it (PDF bookmark paths are long, so labels use the
meaningful tail). ``alt`` lists equally valid locations of the same fact (e.g. the glossary/FAQ).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CATEGORIES = (
    "single_hop",
    "multi_hop",
    "numeric",
    "portuguese",
    "occ",
    "paraphrase",
    "unanswerable",
    "out_of_scope",
    "adversarial",
)
SPLITS = ("tune", "heldout")


@dataclass(frozen=True)
class GoldLocation:
    doc_id: str
    section: str


@dataclass(frozen=True)
class GoldRequirement:
    """One piece of evidence the answer needs; satisfied by any of its locations."""

    locations: tuple[GoldLocation, ...]


@dataclass
class GoldItem:
    id: str
    question: str
    lang: str
    category: str
    answerable: bool
    split: str
    requirements: list[GoldRequirement] = field(default_factory=list)
    gold_answer: str = ""
    key_facts: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    notes: str = ""


def section_matches(label: str, section_path: str) -> bool:
    return section_path == label or section_path.endswith(" > " + label)


def location_matches(loc: GoldLocation, doc_id: str, section_path: str) -> bool:
    return loc.doc_id == doc_id and section_matches(loc.section, section_path)


def requirement_matches(req: GoldRequirement, doc_id: str, section_path: str) -> bool:
    return any(location_matches(loc, doc_id, section_path) for loc in req.locations)


def normalize_for_match(text: str) -> str:
    """Aggressive normalization for verbatim checks: PDF extraction breaks spaces ("b efore")."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", "", text)


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()


def _parse_requirement(raw: dict[str, Any]) -> GoldRequirement:
    locs = [GoldLocation(raw["doc_id"], raw["section"])]
    for alt in raw.get("alt", []) or []:
        locs.append(GoldLocation(alt["doc_id"], alt["section"]))
    return GoldRequirement(tuple(locs))


def load_golden(path: str | Path) -> list[GoldItem]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = []
    for raw in data["items"]:
        items.append(
            GoldItem(
                id=raw["id"],
                question=raw["question"],
                lang=raw["lang"],
                category=raw["category"],
                answerable=bool(raw["answerable"]),
                split=raw.get("split", ""),
                requirements=[_parse_requirement(r) for r in raw.get("gold_sections", []) or []],
                gold_answer=str(raw.get("gold_answer", "")),
                key_facts=[str(k) for k in raw.get("key_facts", []) or []],
                evidence=[str(e) for e in raw.get("evidence", []) or []],
                must_not_contain=[str(m) for m in raw.get("must_not_contain", []) or []],
                notes=str(raw.get("notes", "") or ""),
            )
        )
    return items


def load_corpus_sections(corpus_dir: str | Path, manifest_path: str | Path) -> dict[str, dict[str, str]]:
    """doc_id -> {section_path: section text} using the production parser."""
    from copilot.ingest.manifest import load_manifest
    from copilot.ingest.parsers import parse_file

    out: dict[str, dict[str, str]] = {}
    for e in load_manifest(manifest_path):
        doc = parse_file(Path(corpus_dir) / e.path, doc_id=e.doc_id, extra_meta=e.meta())
        secs: dict[str, list[str]] = {}
        for s in doc.sections:
            secs.setdefault(" > ".join(s.path), []).append(s.text)
        out[e.doc_id] = {k: "\n".join(v) for k, v in secs.items()}
    return out


def verify_item(item: GoldItem, corpus: dict[str, dict[str, str]]) -> list[str]:
    """Return a list of problems (empty == verified)."""
    problems: list[str] = []
    if item.category not in CATEGORIES:
        problems.append(f"unknown category {item.category!r}")
    if item.split and item.split not in SPLITS:
        problems.append(f"unknown split {item.split!r}")
    if item.answerable and not item.requirements:
        problems.append("answerable item without gold_sections")
    if item.answerable and not item.evidence:
        problems.append("answerable item without evidence")
    if not item.answerable and item.requirements:
        problems.append("unanswerable item with gold_sections")
    texts: list[str] = []
    for req in item.requirements:
        for loc in req.locations:
            doc = corpus.get(loc.doc_id)
            if doc is None:
                problems.append(f"unknown doc {loc.doc_id}")
                continue
            hits = [p for p in doc if section_matches(loc.section, p)]
            if len(hits) != 1:
                problems.append(f"section {loc.doc_id}:{loc.section!r} resolves to {len(hits)} paths")
            texts.extend(doc[p] for p in hits)
    blob = normalize_for_match("\n".join(texts))
    for ev in item.evidence:
        if normalize_for_match(ev) not in blob:
            problems.append(f"evidence not found in gold sections: {ev[:70]!r}")
    for kf in item.key_facts:
        if (
            item.answerable
            and normalize_answer(kf) not in normalize_answer(item.gold_answer)
            and (normalize_for_match(kf) not in blob)
        ):
            problems.append(f"key fact neither in gold_answer nor evidence text: {kf!r}")
    return problems
