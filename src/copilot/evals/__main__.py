"""python -m copilot.evals {verify,calibrate,run}

verify     check every golden item against the parsed corpus (no index/model needed)
calibrate  fit the abstention rule on the TUNING split only -> evals/abstention_rule.json
run        evaluate all retrievers + the end-to-end system -> evals/results/<timestamp>.{json,md}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from copilot.config import REPO_ROOT, Settings

GOLDEN = REPO_ROOT / "evals" / "golden_set.yaml"
RESULTS = REPO_ROOT / "evals" / "results"
log = logging.getLogger("copilot.evals")


def _verify(args: argparse.Namespace, s: Settings) -> int:
    from copilot.evals.golden import load_corpus_sections, load_golden, verify_item

    items = load_golden(args.golden)
    corpus = load_corpus_sections(s.corpus_dir, s.manifest_path)
    bad = {it.id: p for it in items if (p := verify_item(it, corpus))}
    for k, v in bad.items():
        print(k, v)
    print(f"{len(items)} items, {len(bad)} with problems")
    return 1 if bad else 0


def _build(s: Settings):
    from copilot.embeddings import build_embedder
    from copilot.store import build_store
    from copilot.text import EMBED_TEXT_VERSION

    store = build_store(s)
    embedder = build_embedder(s.embedder, s.embedding_model, s.embedding_dim)
    store.ensure_embedder(f"{embedder.name}|et{EMBED_TEXT_VERSION}", embedder.dim)
    return store, embedder


def _calibrate(args: argparse.Namespace, s: Settings) -> int:
    from copilot.abstention import SIGNALS
    from copilot.evals.calibrate import cross_validate
    from copilot.evals.golden import load_golden
    from copilot.guardrails import guard_input
    from copilot.retrieval.reranker import CrossEncoderReranker
    from copilot.retrieval.retriever import Retriever

    items = [it for it in load_golden(args.golden) if it.split == "tune"]  # never the held-out split
    store, embedder = _build(s)
    reranker = CrossEncoderReranker(args.rerank_model) if args.rerank_model else None
    retr = Retriever(store, embedder, s.candidate_k, s.rrf_k, reranker)
    signals, labels, rows, blocked = [], [], [], []
    for it in items:
        # mirror the service: blocked inputs never reach the gate; others are retrieved sanitized
        d = guard_input(it.question) if s.guardrails else None
        if d is not None and d.action == "block":
            blocked.append({"id": it.id, "answerable": it.answerable, "reason": d.reason})
            continue
        res = retr.retrieve(
            d.question if d else it.question, k=s.top_k, mode=args.mode, rerank=reranker is not None
        )
        sig = res.signals()
        signals.append(sig)
        labels.append(not it.answerable)
        rows.append(
            {"id": it.id, "category": it.category, "should_abstain": not it.answerable, "signals": sig}
        )
    available = [x for x in SIGNALS if x in args.signals.split(",")]
    cv = cross_validate(
        signals, labels, max_frr=args.max_frr, k=args.folds, repeats=args.repeats, available=available
    )
    rule = cv.rule
    rule.fitted_on = (
        f"tune split of {Path(args.golden).name}: {len(signals)} items reaching the gate "
        f"({len(blocked)} blocked by input guardrails excluded); family chosen by {args.repeats}x"
        f"{args.folds}-fold stratified CV, thresholds re-fitted on all tune items; max_frr={args.max_frr}"
    )
    rule.pipeline = {
        "mode": args.mode,
        "rerank": reranker is not None,
        "rerank_model": args.rerank_model or None,
    }
    Path(args.out).write_text(rule.to_json() + "\n", encoding="utf-8")
    report = {
        "rule": rule.describe(),
        "family": [*cv.family[0], cv.family[1]],
        "cv": {"folds": cv.folds, "repeats": cv.repeats, "table": cv.table},
        "n_tune_items_at_gate": len(signals),
        "n_should_abstain_at_gate": int(sum(labels)),
        "blocked_by_input_guardrails": blocked,
        "signals": rows,
    }
    rep_path = Path(args.out).with_name(Path(args.out).stem + ".cv.json")
    rep_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(
        f"{len(signals)} tune items at the gate ({int(sum(labels))} should abstain); {len(blocked)} blocked"
    )
    print("| family | CV F1 mean ± sd | CV false refusal |")
    for r in cv.table[:12]:
        print(f"| {r['family']} | {r['cv_f1_mean']:.3f} ± {r['cv_f1_std']:.3f} | {r['cv_frr_mean']:.1%} |")
    print("chosen:", rule.describe(), json.dumps({k: round(v, 4) for k, v in rule.tune_metrics.items()}))
    print("wrote", args.out, "and", rep_path)
    return 0


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def _run(args: argparse.Namespace, s: Settings) -> int:
    from copilot.abstention import resolve_rule
    from copilot.evals.golden import load_golden
    from copilot.evals.judge import LLMJudge, detect_provider
    from copilot.evals.report import render_markdown
    from copilot.evals.runner import RetrieverSpec, aggregate, run_retrievers, run_system
    from copilot.llm.extractive import ExtractiveGenerator
    from copilot.retrieval.reranker import CrossEncoderReranker
    from copilot.retrieval.retriever import Retriever
    from copilot.service import CopilotService

    items = load_golden(args.golden)
    if args.split != "all":
        items = [it for it in items if it.split == args.split]
    if args.limit:
        items = items[: args.limit]
    store, embedder = _build(s)
    rule = resolve_rule(s.abstain_rule_path, s.abstain_threshold)
    pipe = {"mode": s.retrieval_mode, "rerank": s.rerank, "rerank_model": s.rerank_model}
    if rule.pipeline and not args.ignore_rule_pipeline:
        pipe.update({k: v for k, v in rule.pipeline.items() if v is not None or k == "rerank"})

    rerank_models = [m for m in (args.rerank_models or "").split(",") if m]
    if pipe["rerank"] and pipe["rerank_model"] and pipe["rerank_model"] not in rerank_models:
        rerank_models.append(pipe["rerank_model"])
    rerankers = {m: CrossEncoderReranker(m) for m in rerank_models}

    base = Retriever(store, embedder, s.candidate_k, s.rrf_k, None)
    specs = [RetrieverSpec(n, n, base) for n in ("bm25", "vector", "hybrid")]
    for m, rr in rerankers.items():
        specs.append(
            RetrieverSpec(
                f"hybrid+rerank[{m.split('/')[-1]}]",
                "hybrid",
                Retriever(store, embedder, s.candidate_k, s.rrf_k, rr),
                True,
            )
        )

    # Generator: LLM if a key is present (or forced), else extractive. Judge only with a key.
    provider = args.generator if args.generator != "auto" else (detect_provider() or "extractive")
    sys_settings = s.model_copy(update={"retrieval_mode": pipe["mode"], "rerank": bool(pipe["rerank"])})
    sys_retr = Retriever(
        store,
        embedder,
        s.candidate_k,
        s.rrf_k,
        rerankers.get(pipe["rerank_model"]) if pipe["rerank"] else None,
    )
    if provider == "extractive":
        generator = ExtractiveGenerator(embedder)
    else:
        from copilot.llm.providers import AnthropicGenerator, OpenAIGenerator

        cls = AnthropicGenerator if provider == "anthropic" else OpenAIGenerator
        generator = cls(s.llm_model, s.llm_max_tokens, s.llm_timeout_s)
    service = CopilotService(sys_settings, store, embedder, generator, sys_retr, abstention=rule)
    judge = None if args.no_judge else LLMJudge.from_env(args.judge_model, s.llm_timeout_s)

    for spec in specs[:1]:  # warm-up (model load, BM25 build) so latency is steady-state
        spec.retriever.retrieve("warm up", k=3, mode="hybrid", rerank=spec.rerank)
    for spec in specs[3:]:
        spec.retriever.retrieve("warm up", k=3, mode="hybrid", rerank=True)

    records = []
    for it in items:
        records.append(
            {
                "id": it.id,
                "split": it.split,
                "category": it.category,
                "lang": it.lang,
                "answerable": it.answerable,
                "n_required": len(it.requirements),
                "must_not_contain": it.must_not_contain,
                "question": it.question,
                "gold_answer": it.gold_answer,
                "gold_sections": [
                    [f"{loc.doc_id}:{loc.section}" for loc in req.locations] for req in it.requirements
                ],
                "retrievers": {},
            }
        )
    # One pass per retriever (then one for the system) so latencies are not distorted by interleaving
    # cheap retrievers with cross-encoder inference on the same CPU.
    t0 = time.perf_counter()
    for spec in specs:
        for it, rec in zip(items, records, strict=True):
            rec["retrievers"].update(run_retrievers(it, [spec]))
        log.info("retriever %s done (%.0fs)", spec.name, time.perf_counter() - t0)
    for n, (it, rec) in enumerate(zip(items, records, strict=True), start=1):
        rec["system"] = run_system(
            it, service, judge, s.llm_price_input_per_mtok, s.llm_price_output_per_mtok
        )
        if n % 40 == 0:
            log.info("system %d/%d (%.0fs)", n, len(items), time.perf_counter() - t0)
    names = [sp.name for sp in specs]
    from copilot.evals.guardrail_eval import run_guardrail_eval, summary

    guard_res = run_guardrail_eval(golden_path=args.golden)
    llm_reason = None
    if judge is None:
        llm_reason = (
            "not run: no ANTHROPIC_API_KEY or OPENAI_API_KEY" if not detect_provider() else "disabled"
        )
    result = {
        "meta": {
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_rev": _git_rev() or "uncommitted working tree (no commits yet)",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "golden_set": str(Path(args.golden).relative_to(REPO_ROOT))
            if Path(args.golden).is_relative_to(REPO_ROOT)
            else args.golden,
            "n_items": len(items),
            "store": s.store,
            "embedder": f"{s.embedder}:{s.embedding_model}",
            "index": store.stats(),
            "retrievers": names,
            "candidate_k": s.candidate_k,
            "k_eval": 10,
            "system": {
                "pipeline": pipe,
                "top_k": s.top_k,
                "generator": provider,
                "abstention_rule": rule.describe(),
                "abstention_rule_fitted_on": rule.fitted_on,
            },
            "llm_metrics": {
                "status": "run" if judge else llm_reason,
                "judge": None if judge is None else {"provider": judge.provider, "model": judge.model},
                "judge_usage": None if judge is None else judge.usage.__dict__,
            },
            "prices_usd_per_mtok": {
                "input": s.llm_price_input_per_mtok,
                "output": s.llm_price_output_per_mtok,
            },
            "wall_time_s": round(time.perf_counter() - t0, 1),
        },
        "overall": aggregate(records, names),
        "by_split": aggregate(records, names, lambda r: r["split"]),
        "by_category": aggregate(records, names, lambda r: r["category"]),
        "by_lang": aggregate(records, names, lambda r: r["lang"]),
        "by_split_category": aggregate(records, names, lambda r: f"{r['split']}/{r['category']}"),
        "guardrails": {"summary": summary(guard_res), "details": guard_res},
        "records": records,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out_dir) / f"{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(render_markdown(result), encoding="utf-8")
    print(f"wrote {out}\nwrote {md}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "httpx2", "sentence_transformers", "transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser(prog="python -m copilot.evals")
    ap.add_argument("--golden", default=str(GOLDEN))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    c = sub.add_parser("calibrate")
    c.add_argument("--mode", default="hybrid", choices=["bm25", "vector", "hybrid"])
    c.add_argument(
        "--rerank-model",
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        help="'' to calibrate without a reranker",
    )
    c.add_argument("--signals", default="dense,rerank,bm25_norm,coverage")
    c.add_argument("--max-frr", type=float, default=0.10)
    c.add_argument("--folds", type=int, default=5)
    c.add_argument("--repeats", type=int, default=5)
    c.add_argument("--out", default=str(REPO_ROOT / "evals" / "abstention_rule.json"))
    r = sub.add_parser("run")
    r.add_argument("--split", default="all", choices=["all", "tune", "heldout"])
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--rerank-models", default="", help="comma-separated extra rerankers to compare")
    r.add_argument("--generator", default="auto", choices=["auto", "extractive", "anthropic", "openai"])
    r.add_argument("--judge-model", default="")
    r.add_argument("--no-judge", action="store_true")
    r.add_argument("--ignore-rule-pipeline", action="store_true")
    r.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args(argv)
    s = Settings()
    return {"verify": _verify, "calibrate": _calibrate, "run": _run}[args.cmd](args, s)


if __name__ == "__main__":
    raise SystemExit(main())
