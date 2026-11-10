"""Print the dense support score for in-scope vs out-of-scope probe questions.

Used to pick COPILOT_ABSTAIN_THRESHOLD for a given embedder (see docs/DECISIONS.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from copilot.config import get_settings
from copilot.service import CopilotService

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    probe = yaml.safe_load((ROOT / "evals" / "abstention_probe.yaml").read_text())
    svc = CopilotService.from_settings(get_settings())
    rows = []
    for label in ("in_scope", "out_of_scope"):
        for q in probe[label]:
            r = svc.retriever.retrieve(q, k=1, mode="vector")
            rows.append((label, r.max_vector_sim, q, r.items[0].chunk.doc_id if r.items else "-"))
    for label, s, q, d in sorted(rows, key=lambda x: x[1]):
        print(f"{s:.4f}  {label:12s}  {d:20s}  {q}")
    ins = [s for lab, s, *_ in rows if lab == "in_scope"]
    outs = [s for lab, s, *_ in rows if lab == "out_of_scope"]
    print(
        f"\nin_scope  min={min(ins):.4f}  out_of_scope max={max(outs):.4f}  "
        f"separable={'yes' if min(ins) > max(outs) else 'no'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
