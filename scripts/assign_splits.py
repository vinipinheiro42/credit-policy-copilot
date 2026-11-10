"""One-off, deterministic split assignment for evals/golden_set.yaml.

Within each category (file order), round(25% of n) items at evenly spaced positions go to ``heldout``;
the rest go to ``tune``. Stratified by category, deterministic, independent of any model output.
Existing ``split`` values are kept unless --force is passed; new items get their own per-category
25% quota (evenly spaced among the new items), so
adding new items later does not reshuffle the held-out set.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "evals" / "golden_set.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    lines = PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    # find items: "- id:" line, then the "  category:" line and optional "  split:" line within the item
    starts = [i for i, ln in enumerate(lines) if ln.startswith("- id: ")]
    bounds = list(zip(starts, [*starts[1:], len(lines)], strict=True))
    cats_in_order: dict[str, list[int]] = defaultdict(list)
    for n, (a, b) in enumerate(bounds):
        cat = next(ln for ln in lines[a:b] if ln.startswith("  category: ")).split(":", 1)[1].strip()
        has_split = any(ln.startswith("  split: ") for ln in lines[a:b])
        if args.force or not has_split:  # quotas are computed among the items being (re)assigned
            cats_in_order[cat].append(n)
    heldout_items: set[int] = set()
    for members in cats_in_order.values():
        n_h = int(len(members) * 0.25 + 0.5)
        heldout_items.update(members[int((k + 0.5) * len(members) / n_h)] for k in range(n_h))
    edits: dict[int, str] = {}
    removals: set[int] = set()
    counts: Counter = Counter()
    for n, (a, b) in enumerate(bounds):
        block = lines[a:b]
        cat_idx = next(a + j for j, ln in enumerate(block) if ln.startswith("  category: "))
        cat = lines[cat_idx].split(":", 1)[1].strip()
        split_idx = next((a + j for j, ln in enumerate(block) if ln.startswith("  split: ")), None)
        new = "heldout" if n in heldout_items else "tune"
        if split_idx is not None and not args.force:
            counts[(cat, lines[split_idx].split(":", 1)[1].strip())] += 1
            continue
        if split_idx is not None:
            removals.add(split_idx)
        edits[cat_idx] = f"  split: {new}\n"
        counts[(cat, new)] += 1
    out = []
    for i, ln in enumerate(lines):
        if i in removals:
            continue
        out.append(ln)
        if i in edits:
            out.append(edits[i])
    PATH.write_text("".join(out), encoding="utf-8")
    for cat in sorted({c for c, _ in counts}):
        print(f"{cat:14s} tune={counts[(cat, 'tune')]:3d} heldout={counts[(cat, 'heldout')]:3d}")
    tot_h = sum(v for (c, s), v in counts.items() if s == "heldout")
    print(f"total={sum(counts.values())} heldout={tot_h} ({tot_h / sum(counts.values()):.1%})")
    assert not re.search(r"^  split: (?!tune|heldout)", PATH.read_text(encoding="utf-8"), re.M)


if __name__ == "__main__":
    main()
