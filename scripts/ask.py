"""Tiny CLI client: python scripts/ask.py "question" [--url http://localhost:8000] [--out file.json]"""

from __future__ import annotations

import argparse
import json

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--mode", choices=["bm25", "vector", "hybrid"])
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    payload = {"question": a.question, "mode": a.mode, "rerank": a.rerank or None}
    body = httpx.post(f"{a.url}/ask", json=payload, timeout=120).json()
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"request": payload, "response": body}, f, indent=2, ensure_ascii=False)
    print(f"Q: {a.question}")
    print(f"abstained={body['abstained']} support={body['support_score']} latency_ms={body['latency_ms']}")
    print(body["answer"])
    for c in body["citations"]:
        print(f"  [{c['ref']}] {c['chunk_id']}  ({c['section']})")
    print("  top-3 retrieved:", [c["chunk_id"] for c in body["contexts"][:3]])


if __name__ == "__main__":
    main()
