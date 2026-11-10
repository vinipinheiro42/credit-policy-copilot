from copilot.evals.report import render_markdown
from copilot.evals.runner import aggregate


def _rec(id_, answerable, abstained, hits, cited_hit=1.0, kf=1.0, split="tune", cat="single_hop", canary=()):
    sysrec = {
        "abstained": abstained,
        "answer": "x",
        "citations": [{"doc_id": "D", "section": "S"}],
        "signals": {"dense": 0.9, "rerank": None, "bm25_norm": 1.0},
        "latency_ms": 10.0,
        "tokens": {"input": 0, "output": 0},
        "cost_usd": 0.0,
        "canary_violations": list(canary),
    }
    if answerable and not abstained:
        sysrec["citation"] = {"hit": cited_hit, "precision": cited_hit, "recall": cited_hit}
        sysrec["key_fact_recall"] = kf
    return {
        "id": id_,
        "split": split,
        "category": cat,
        "lang": "en",
        "answerable": answerable,
        "n_required": 1 if answerable else 0,
        "must_not_contain": ["PWNED"] if canary else [],
        "question": "q",
        "gold_answer": "g",
        "gold_sections": [["D:S"]] if answerable else [],
        "retrievers": {"hybrid": {"hits": hits, "latency_ms": 5.0, "top": [], "signals": {}}},
        "system": sysrec,
    }


def test_aggregate_and_render():
    records = [
        _rec("A", True, False, [[0], [], []]),
        _rec("B", True, False, [[], [0], []], cited_hit=0.0, kf=0.5, split="heldout"),
        _rec("C", True, True, [[], [], []]),
        _rec("D", False, True, [[]], cat="out_of_scope"),
        _rec("E", False, False, [[]], cat="adversarial", canary=["PWNED"], split="heldout"),
    ]
    agg = aggregate(records, ["hybrid"])
    r = agg["retrieval"]["hybrid"]
    assert r["n"] == 3
    assert r["recall@1"] == 1 / 3 and r["recall@5"] == 2 / 3
    assert r["mrr"] == (1 + 0.5 + 0) / 3
    s = agg["system"]
    assert s["abstention"]["tp"] == 1 and s["abstention"]["fp"] == 1 and s["abstention"]["fn"] == 1
    assert s["citation_hit"] == 0.5
    assert s["answerable_fully_supported_rate"] == 1 / 3
    assert s["canary_violation_rate"] == 1.0
    assert s["cost_per_query_usd"] == 0.0
    result = {
        "meta": {
            "created_utc": "t",
            "git_rev": None,
            "golden_set": "g",
            "n_items": 5,
            "index": {},
            "embedder": "e",
            "store": "s",
            "retrievers": ["hybrid"],
            "wall_time_s": 1,
            "platform": "p",
            "system": {
                "pipeline": {"mode": "hybrid", "rerank": False, "rerank_model": None},
                "top_k": 5,
                "generator": "extractive",
                "abstention_rule": "abstain if dense < 0.8",
                "abstention_rule_fitted_on": "manual",
            },
            "llm_metrics": {"status": "not run: no key"},
        },
        "overall": agg,
        "by_split": aggregate(records, ["hybrid"], lambda r: r["split"]),
        "by_category": aggregate(records, ["hybrid"], lambda r: r["category"]),
        "by_lang": aggregate(records, ["hybrid"], lambda r: r["lang"]),
        "by_split_category": aggregate(records, ["hybrid"], lambda r: f"{r['split']}/{r['category']}"),
        "records": records,
    }
    md = render_markdown(result)
    assert "not run" in md and "| heldout |" in md
    assert "false refusal" in md and "| C |" in md and "canary leaked" in md
