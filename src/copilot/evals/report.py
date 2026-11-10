"""Markdown summary of an eval run (the JSON next to it has every per-question record)."""

from __future__ import annotations

from typing import Any


def _f(x: Any, pct: bool = False, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{x * 100:.1f}%" if pct else f"{x:.{nd}f}"


def _ms(x: Any) -> str:
    return "n/a" if x is None else f"{x:.0f}"


def retrieval_table(block: dict[str, Any]) -> list[str]:
    rows = [
        "| retriever | n | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in block.items():
        rows.append(
            f"| {name} | {m['n']} | {_f(m.get('recall@1'))} | {_f(m.get('recall@5'))} | "
            f"{_f(m.get('recall@10'))} | {_f(m.get('mrr'))} | {_f(m.get('ndcg@10'))} | "
            f"{_ms(m.get('latency_p50_ms'))} | {_ms(m.get('latency_p95_ms'))} |"
        )
    return rows


def system_row(label: str, m: dict[str, Any], llm: str) -> str:
    a = m["abstention"]
    cost = m["cost_per_query_usd"]
    return (
        f"| {label} | {m['n']} | {_f(a['precision'])} | {_f(a['recall'])} | {_f(a['f1'])} | "
        f"{_f(a['false_refusal_rate'], True)} | {_f(m['citation_hit'])} | {_f(m['citation_precision'])} | "
        f"{_f(m['key_fact_recall_answered'])} | {_f(m['answerable_fully_supported_rate'], True)} | "
        f"{_f(m['canary_violation_rate'], True)} | {_ms(m['latency_p50_ms'])} | {_ms(m['latency_p95_ms'])} | "
        f"{'unknown (set prices)' if cost is None else f'${cost:.5f}'} | "
        f"{_f(m['faithfulness']) if m['faithfulness'] is not None else llm} | "
        f"{_f(m['correctness']) if m['correctness'] is not None else llm} |"
    )


SYSTEM_HEADER = [
    "| slice | n | abst. P | abst. R | abst. F1 | false refusal | cite hit | cite prec. | key-fact recall"
    " | fully supported | canary viol. | p50 ms | p95 ms | cost/query | faithfulness | correctness |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]


ORDER = ("false refusal", "answered unanswerable", "canary leaked", "wrong citation", "key fact missing")


def _failures(result: dict[str, Any], system_retriever: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for r in result["records"]:
        s = r["system"]
        kind, detail = None, ""
        if r["answerable"] and s["abstained"]:
            kind, detail = "false refusal", f"signals={_sig(s['signals'])}"
        elif not r["answerable"] and not s["abstained"]:
            kind, detail = "answered unanswerable", f"signals={_sig(s['signals'])}"
        elif r["answerable"] and s.get("citation", {}).get("hit") == 0.0:
            got = s["citations"][0] if s["citations"] else {}
            kind = "wrong citation"
            detail = f"cited {got.get('doc_id')}:{_tail(got.get('section'))}; gold {_gold(r)}"
        elif r["answerable"] and (s.get("key_fact_recall") or 0) < 1.0:
            kind, detail = "key fact missing", f"gold: {r['gold_answer'][:80]}"
        if s.get("canary_violations"):
            kind, detail = "canary leaked", ", ".join(s["canary_violations"])
        if kind:
            found.append(
                (ORDER.index(kind), f"| {r['id']} | {r['split']} | {r['category']} | {kind} | {detail} |")
            )
    rows = ["| id | split | category | failure | detail |", "|---|---|---|---|---|"]
    return rows + [row for _, row in sorted(found, key=lambda x: x[0])]


def _tail(section: str | None) -> str:
    return "" if not section else section.split(" > ")[-1][:50]


def _gold(r: dict[str, Any]) -> str:
    return "; ".join(
        req[0].split(":", 1)[0] + ":" + _tail(req[0].split(":", 1)[1]) for req in r["gold_sections"]
    )


def _sig(sig: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v:.3f}" for k, v in sig.items() if v is not None)


def render_markdown(result: dict[str, Any]) -> str:
    meta = result["meta"]
    llm = meta["llm_metrics"]["status"]
    llm_cell = "not run" if llm != "run" else "n/a"
    sysm = meta["system"]
    names = meta["retrievers"]
    sys_retr = next(
        (
            n
            for n in names
            if sysm["pipeline"]["rerank"]
            and n.startswith("hybrid+rerank")
            and n.endswith(f"[{str(sysm['pipeline'].get('rerank_model', '')).split('/')[-1]}]")
        ),
        sysm["pipeline"]["mode"],
    )
    out = [
        f"# Eval run {meta['created_utc']}",
        "",
        f"- golden set: `{meta['golden_set']}`, {meta['n_items']} items; git rev: `{meta['git_rev']}`",
        f"- index: {meta['index']}; embedder `{meta['embedder']}`; store `{meta['store']}`",
        f"- system pipeline: {sysm['pipeline']}, top_k={sysm['top_k']}, generator=`{sysm['generator']}`",
        f"- abstention rule: **{sysm['abstention_rule']}** (fitted on: {sysm['abstention_rule_fitted_on']})",
        f"- LLM-judged metrics (faithfulness, correctness): **{llm}**",
        f"- wall time: {meta['wall_time_s']} s on {meta['platform']}",
        "",
        "## Retrieval (answerable items; relevance = gold doc_id + section)",
        "",
    ]
    for split in ("tune", "heldout"):
        if split in result["by_split"]:
            out += [f"### {split}", "", *retrieval_table(result["by_split"][split]["retrieval"]), ""]
    out += ["### all", "", *retrieval_table(result["overall"]["retrieval"]), ""]
    out += ["## End-to-end system (retrieval -> abstention gate -> generator -> citations)", ""]
    out += [
        "Abstention: positive class = should abstain (unanswerable, out-of-scope, adversarial-unanswerable).",
        "",
    ]
    out += SYSTEM_HEADER
    for split in ("tune", "heldout"):
        if split in result["by_split"]:
            out.append(system_row(split, result["by_split"][split]["system"], llm_cell))
    out.append(system_row("all", result["overall"]["system"], llm_cell))
    out += ["", f"## By category (system retriever = `{sys_retr}` for Recall@5/MRR)", ""]
    out += [
        "| category | split | n | Recall@5 | MRR | abst. correct | false refusal | cite hit"
        " | key-fact recall |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, block in result["by_split_category"].items():
        split, cat = key.split("/")
        out.append(_cat_row(cat, split, block, sys_retr))
    out += [
        "",
        "## By language",
        "",
        "| lang | n | Recall@5 | MRR | abst. correct | false refusal | cite hit |",
        "|---|---|---|---|---|---|---|",
    ]
    for lang, block in result["by_lang"].items():
        r = block["retrieval"].get(sys_retr, {})
        s = block["system"]
        out.append(
            f"| {lang} | {s['n']} | {_f(r.get('recall@5'))} | {_f(r.get('mrr'))} | "
            f"{_f(_abst_acc(s), True)} | "
            f"{_f(s['abstention']['false_refusal_rate'], True)} | {_f(s['citation_hit'])} |"
        )
    out += ["", "## Abstention reasons and guardrail actions (system)", ""]
    out += [
        "| split | abstain reasons | input guard (allow/sanitized/block) | output-check withheld |",
        "|---|---|---|---|",
    ]
    for split in ("tune", "heldout"):
        sm = result["by_split"].get(split, {}).get("system")
        if sm and "abstain_reasons" in sm:
            ga = sm["input_guard_actions"]
            out.append(
                f"| {split} | {sm['abstain_reasons']} | {ga['allow']}/{ga['sanitized']}/{ga['block']} |"
                f" {sm['output_check_failures']} |"
            )
    if "guardrails" in result:
        out += ["", "## Guardrail eval (evals/guardrail_set.yaml + poisoned test fixture)", ""]
        out += ["| metric | value |", "|---|---|"]
        out += [f"| {k} | {_f(v)} |" for k, v in result["guardrails"]["summary"].items()]
    out += ["", "## All failures of the system (grouped by type)", "", *_failures(result, sys_retr), ""]
    out += [
        "`key fact missing` is a lexical check on the extractive answer (it quotes passages, it does not",
        "compose them), so it is a weak proxy; LLM-judged correctness replaces it once a key is set.",
        "",
    ]
    return "\n".join(out) + "\n"


def _abst_acc(s: dict[str, Any]) -> float | None:
    a = s["abstention"]
    n = s["n"]
    if not n:
        return None
    tn = a["n_answerable"] - a["fp"]
    return (a["tp"] + tn) / n


def _cat_row(cat: str, split: str, block: dict[str, Any], sys_retr: str) -> str:
    r = block["retrieval"].get(sys_retr, {})
    s = block["system"]
    frr = _f(s["abstention"]["false_refusal_rate"], True) if s["abstention"]["n_answerable"] else "n/a"
    return (
        f"| {cat} | {split} | {s['n']} | {_f(r.get('recall@5'))} | {_f(r.get('mrr'))} | "
        f"{_f(_abst_acc(s), True)} | {frr} | "
        f"{_f(s['citation_hit'])} | {_f(s['key_fact_recall_answered'])} |"
    )
