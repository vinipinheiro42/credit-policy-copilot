"""Corpus manifest: provenance, license and integrity hash of every source document."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ManifestEntry:
    doc_id: str
    path: str
    title: str
    synthetic: bool
    language: str
    license: str
    sha256: str | None
    source_url: str | None = None
    retrieved: str | None = None
    publisher: str | None = None
    notes: str | None = None

    def meta(self) -> dict[str, Any]:
        return {
            "synthetic": self.synthetic,
            "language": self.language,
            "license": self.license,
            "source_url": self.source_url,
            "publisher": self.publisher,
            "title": self.title,
        }


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = []
    for d in data.get("documents", []):
        entries.append(
            ManifestEntry(
                doc_id=d["doc_id"],
                path=d["path"],
                title=d["title"],
                synthetic=bool(d.get("synthetic", False)),
                language=d.get("language", "en"),
                license=d.get("license", ""),
                sha256=d.get("sha256"),
                source_url=d.get("source_url"),
                retrieved=str(d["retrieved"]) if d.get("retrieved") else None,
                publisher=d.get("publisher"),
                notes=d.get("notes"),
            )
        )
    ids = [e.doc_id for e in entries]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate doc_id in manifest: {sorted(dupes)}")
    return entries
