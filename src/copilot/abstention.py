"""Abstention gate: decide from retrieval signals whether the copilot should refuse to answer.

Signals (all computed by the retriever, no LLM):
- ``dense``: best cosine similarity of the query to any chunk (always available).
- ``rerank``: cross-encoder score of the top chunk (only when reranking ran).
- ``bm25_norm``: top BM25 score divided by the number of query terms (bm25/hybrid modes).
- ``coverage``: IDF-weighted share of the query's terms that occur in the top-3 chunks (M3; aimed at
  "topic indexed, specific fact missing", e.g. "annual fee" when no chunk mentions an annual fee).

A rule abstains when a signal is below its minimum. With several signals, ``combine="any"`` abstains if
ANY signal is low (strict), ``"all"`` only if ALL are low (lenient). Rules are fitted on the tuning
split by ``python -m copilot.evals calibrate`` and stored as JSON (evals/abstention_rule.json).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

SIGNALS = ("dense", "rerank", "bm25_norm", "coverage")


@dataclass
class AbstentionRule:
    thresholds: dict[str, float] = field(default_factory=lambda: {"dense": 0.83})
    combine: Literal["any", "all"] = "any"
    # provenance (filled by calibration); informational only
    fitted_on: str = "manual"
    pipeline: dict[str, object] = field(default_factory=dict)
    tune_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def needs_rerank(self) -> bool:
        return "rerank" in self.thresholds

    def decide(self, signals: dict[str, float | None]) -> bool:
        """True == abstain. A signal that is missing (None) is ignored; if all are missing -> abstain."""
        lows = [signals[name] < t for name, t in self.thresholds.items() if signals.get(name) is not None]
        if not lows:
            return True
        return any(lows) if self.combine == "any" else all(lows)

    def describe(self) -> str:
        parts = [f"{k} < {v:.4g}" for k, v in self.thresholds.items()]
        joiner = " OR " if self.combine == "any" else " AND "
        return "abstain if " + joiner.join(parts)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> AbstentionRule:
        unknown = set(d.get("thresholds", {})) - set(SIGNALS)
        if unknown:
            raise ValueError(f"Unknown abstention signals: {sorted(unknown)}")
        return cls(
            thresholds={k: float(v) for k, v in d["thresholds"].items()},
            combine=d.get("combine", "any"),
            fitted_on=d.get("fitted_on", "manual"),
            pipeline=d.get("pipeline", {}),
            tune_metrics=d.get("tune_metrics", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> AbstentionRule:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def resolve_rule(rule_path: str | None, dense_threshold: float) -> AbstentionRule:
    """Calibrated rule file if it exists, else the single dense threshold from settings."""
    if rule_path and rule_path.lower() != "none" and Path(rule_path).is_file():
        return AbstentionRule.load(rule_path)
    return AbstentionRule(thresholds={"dense": dense_threshold})
