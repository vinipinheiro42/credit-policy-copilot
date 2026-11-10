"""Score alternative abstention rules on tune vs held-out from the signals stored in an eval run.

Every rule here was fitted on the tuning split only (or is the hand-set M1 threshold); the held-out
column is reported, never used for selection.

    python scripts/ablate_abstention.py [evals/results/<run>.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from copilot.abstention import AbstentionRule
from copilot.evals.calibrate import calibrate
from copilot.evals.metrics import abstention_metrics

ROOT = Path(__file__).resolve().parents[1]

RULES: list[tuple[str, str, AbstentionRule | Path | str]] = [
    ("M1 hand-set (dense only)", "hybrid", AbstentionRule({"dense": 0.83})),
    ("dense only, tuned", "hybrid", "fit-dense"),  # refit on the tune records of this run
    ("dense AND bm25_norm, tuned (no reranker)", "hybrid", ROOT / "evals" / "abstention_rule.no_rerank.json"),
    (
        "dense AND rerank[ms-marco], tuned",
        "hybrid+rerank[ms-marco-MiniLM-L-6-v2]",
        ROOT / "evals" / "abstention_rule.msmarco.json",
    ),
    (
        "dense AND rerank[mmarco], tuned (SHIPPED)",
        "hybrid+rerank[mmarco-mMiniLMv2-L12-H384-v1]",
        ROOT / "evals" / "abstention_rule.json",
    ),
]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "evals" / "results").glob("*.json"))[-1]
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    print(f"run: {path.name}\n")
    print("| rule | split | P | R | F1 | false refusal |")
    print("|---|---|---|---|---|---|")
    for label, retr, rule in RULES:
        if retr not in records[0]["retrievers"]:
            continue
        if rule == "fit-dense":
            tune = [x for x in records if x["split"] == "tune"]
            r = calibrate(
                [x["retrievers"][retr]["signals"] for x in tune],
                [not x["answerable"] for x in tune],
                available=["dense"],
            ).rule
        else:
            r = AbstentionRule.load(rule) if isinstance(rule, Path) else rule
        for split in ("tune", "heldout"):
            rs = [x for x in records if x["split"] == split]
            m = abstention_metrics(
                [not x["answerable"] for x in rs], [r.decide(x["retrievers"][retr]["signals"]) for x in rs]
            )
            print(
                f"| {label}: `{r.describe()}` | {split} | {m['precision']:.3f} | {m['recall']:.3f} | "
                f"{m['f1']:.3f} | {m['false_refusal_rate']:.1%} |"
            )


if __name__ == "__main__":
    main()
