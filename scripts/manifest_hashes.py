"""Check or write sha256 hashes in data/corpus/MANIFEST.yaml.

python scripts/manifest_hashes.py --check   # exit 1 if any file is missing or differs
python scripts/manifest_hashes.py --write   # fill in / update sha256 fields in place
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "corpus" / "MANIFEST.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args()

    text = MANIFEST.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    bad = 0
    for d in data["documents"]:
        f = MANIFEST.parent / d["path"]
        if not f.exists():
            print(f"MISSING {d['doc_id']}: {f}")
            bad += 1
            continue
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        if args.check:
            if d.get("sha256") != sha:
                print(f"MISMATCH {d['doc_id']}: manifest={d.get('sha256')} file={sha}")
                bad += 1
            continue
        # textual in-place edit keeps comments and anchors intact
        block = re.compile(rf"- doc_id: {re.escape(d['doc_id'])}\n(?: {{4}}.*\n)*", re.M)
        m = block.search(text)
        if not m:
            raise SystemExit(f"cannot locate entry {d['doc_id']}")
        entry = re.sub(r"    sha256: \S+\n", "", m.group(0))
        # insert sha256 right after the path line
        entry = re.sub(r"(    path: .*\n)", rf"\g<1>    sha256: {sha}\n", entry, count=1)
        text = text[: m.start()] + entry + text[m.end() :]
    if args.write:
        MANIFEST.write_text(text, encoding="utf-8")
        print(f"updated {MANIFEST}")
    if args.check:
        print("OK" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
