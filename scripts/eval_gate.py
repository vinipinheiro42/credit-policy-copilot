#!/usr/bin/env python
"""Eval regression gate on a FAST, model-free config (hash embedder + SQLite + no reranker).

Steps: ingest the corpus into a throwaway SQLite index, run the eval harness with the extractive generator and
the fast abstention rule (fitted on the tune split with this same config), then compare headline metrics to
the committed baseline `evals/baselines/fast.json`. Fails (exit 1) when a gated metric drops by more than
the tolerance. Absolute numbers are NOT comparable to the production config (real embedder + reranker);
the gate only detects regressions in code paths (chunking, BM25, fusion, gate, guardrails, citations).

  python scripts/eval_gate.py                    # gate
  python scripts/eval_gate.py --update-baseline  # rewrite the baseline (do this deliberately, in a PR)
  python scripts/eval_gate.py --calibrate        # re-fit the fast rule on tune (CV), then gate
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".data" / "ci-fast"
BASELINE = ROOT / "evals" / "baselines" / "fast.json"
FAST_RULE = ROOT / "evals" / "abstention_rule.fast.json"

# metric -> direction (+1 higher is better, -1 lower is better)
GATED = {
    "recall@5_hybrid_all": +1,
    "recall@5_bm25_all": +1,
    "abstention_f1_all": +1,
    "citation_hit_all": +1,
    "pii_recall": +1,
    "injection_block_rate": +1,
    "output_detection_rate": +1,
    "false_block_rate_benign": -1,
    "poisoned_canary_leak_on": -1,
}


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "COPILOT_STORE": "sqlite",
            "COPILOT_SQLITE_PATH": str(WORK / "index.sqlite"),
            "COPILOT_EMBEDDER": "hash",
            "COPILOT_RERANK": "false",
            "COPILOT_ABSTAIN_RULE_PATH": str(FAST_RULE),
            "COPILOT_LOG_JSON": "false",
            "COPILOT_OTEL_EXPORTER": "none",
        }
    )
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):  # the gate is deterministic: never call an LLM
        env.pop(k, None)
    return env


def _sh(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=_env(), check=True)


def headline(res: dict) -> dict[str, float | None]:
    ov = res["overall"]
    out = {
        "recall@5_hybrid_all": ov["retrieval"]["hybrid"].get("recall@5"),
        "recall@5_bm25_all": ov["retrieval"]["bm25"].get("recall@5"),
        "abstention_f1_all": ov["system"]["abstention"]["f1"],
        "citation_hit_all": ov["system"].get("citation_hit"),
    }
    for split in ("tune", "heldout"):
        sm = res["by_split"].get(split, {})
        out[f"recall@5_hybrid_{split}"] = sm.get("retrieval", {}).get("hybrid", {}).get("recall@5")
        out[f"abstention_f1_{split}"] = sm.get("system", {}).get("abstention", {}).get("f1")
    out.update(res.get("guardrails", {}).get("summary", {}))
    return {k: (None if v is None else round(float(v), 4)) for k, v in out.items()}


def compare(cur: dict, base: dict, tol: float) -> list[str]:
    problems = []
    for k, d in GATED.items():
        b, c = base.get(k), cur.get(k)
        if b is None:
            continue
        if c is None:
            problems.append(f"{k}: missing (baseline {b})")
        elif d * (c - b) < -tol:
            problems.append(f"{k}: {c:.4f} vs baseline {b:.4f} (tolerance {tol})")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "index.sqlite").unlink(missing_ok=True)
    _sh(sys.executable, "-m", "copilot.ingest.cli")
    if args.calibrate or not FAST_RULE.exists():
        _sh(
            sys.executable,
            "-m",
            "copilot.evals",
            "calibrate",
            "--rerank-model",
            "",
            "--signals",
            "dense,bm25_norm,coverage",
            "--out",
            str(FAST_RULE),
        )
    out_dir = WORK / "eval"
    _sh(
        sys.executable,
        "-m",
        "copilot.evals",
        "run",
        "--generator",
        "extractive",
        "--no-judge",
        "--out-dir",
        str(out_dir),
    )
    latest = max(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    cur = headline(json.loads(latest.read_text(encoding="utf-8")))
    print(json.dumps(cur, indent=1))
    if args.update_baseline:
        BASELINE.write_text(
            json.dumps(
                {"config": "hash+sqlite+no-rerank, extractive", "tolerance": args.tolerance, "metrics": cur},
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print("baseline updated:", BASELINE)
        return 0
    base = json.loads(BASELINE.read_text(encoding="utf-8"))["metrics"]
    problems = compare(cur, base, args.tolerance)
    print(f"gate finished in {time.perf_counter() - t0:.0f}s")
    if problems:
        print("EVAL REGRESSION:\n  " + "\n  ".join(problems))
        return 1
    print(f"eval gate OK ({len(GATED)} gated metrics within {args.tolerance} of baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
