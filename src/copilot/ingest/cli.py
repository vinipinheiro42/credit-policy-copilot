"""`copilot-ingest` / `python -m copilot.ingest.cli`: index the corpus listed in MANIFEST.yaml."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict

from copilot.config import get_settings
from copilot.embeddings import build_embedder
from copilot.ingest.manifest import load_manifest
from copilot.ingest.pipeline import ingest
from copilot.store import build_store


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="drop the index before ingesting")
    ap.add_argument("--only", nargs="*", help="restrict to these doc_ids (disables pruning)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    s = get_settings()
    store = build_store(s)
    if args.rebuild:
        store.reset()
    embedder = build_embedder(s.embedder, s.embedding_model, s.embedding_dim)
    entries = load_manifest(s.manifest_path)
    if args.only:
        entries = [e for e in entries if e.doc_id in set(args.only)]
    stats = ingest(
        entries, s.corpus_dir, store, embedder, s.chunk_target_chars, s.chunk_max_chars, prune=not args.only
    )
    print(json.dumps({"stats": asdict(stats), "store": store.stats()}, indent=2))


if __name__ == "__main__":
    main()
