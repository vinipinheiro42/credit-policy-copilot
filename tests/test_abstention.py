import json

import pytest

from copilot.abstention import AbstentionRule, resolve_rule
from copilot.evals.calibrate import calibrate


def test_rule_any_vs_all():
    any_ = AbstentionRule({"dense": 0.8, "rerank": 0.0}, "any")
    all_ = AbstentionRule({"dense": 0.8, "rerank": 0.0}, "all")
    low_dense = {"dense": 0.7, "rerank": 2.0}
    assert any_.decide(low_dense) is True
    assert all_.decide(low_dense) is False
    assert all_.decide({"dense": 0.7, "rerank": -1.0}) is True


def test_missing_signal_is_ignored_and_all_missing_abstains():
    r = AbstentionRule({"dense": 0.8, "rerank": 0.0}, "any")
    assert r.decide({"dense": 0.9, "rerank": None}) is False
    assert AbstentionRule({"rerank": 0.0}).decide({"dense": 0.9, "rerank": None}) is True


def test_roundtrip_and_resolve(tmp_path):
    r = AbstentionRule({"dense": 0.81, "bm25_norm": 1.5}, "all", fitted_on="x")
    p = tmp_path / "rule.json"
    p.write_text(r.to_json())
    loaded = resolve_rule(str(p), 0.5)
    assert loaded.thresholds == r.thresholds and loaded.combine == "all"
    assert resolve_rule(str(tmp_path / "missing.json"), 0.5).thresholds == {"dense": 0.5}
    assert resolve_rule("", 0.5).thresholds == {"dense": 0.5}
    p.write_text(json.dumps({"thresholds": {"bogus": 1}}))
    with pytest.raises(ValueError):
        resolve_rule(str(p), 0.5)


def test_calibrate_finds_separating_threshold():
    # dense separates perfectly at 0.8; rerank is noise
    sig = [
        {"dense": d, "rerank": r, "bm25_norm": None}
        for d, r in [(0.9, 1), (0.85, -1), (0.82, 2), (0.7, 1), (0.75, -2)]
    ]
    y = [False, False, False, True, True]
    cal = calibrate(sig, y, max_frr=0.0, available=["dense", "rerank"])
    assert cal.metrics["f1"] == 1.0
    assert cal.metrics["false_refusal_rate"] == 0.0
    assert set(cal.rule.thresholds) == {"dense"}  # simpler rule wins the tie
    assert 0.75 < cal.rule.thresholds["dense"] < 0.82


def test_calibrate_respects_false_refusal_budget():
    sig = [{"dense": d} for d in (0.9, 0.6, 0.65, 0.7)]
    y = [False, False, True, True]  # an answerable item (0.6) sits below the unanswerables
    strict = calibrate(sig, y, max_frr=0.0, available=["dense"])
    assert strict.metrics["false_refusal_rate"] == 0.0
    loose = calibrate(sig, y, max_frr=0.5, available=["dense"])
    assert loose.metrics["recall"] == 1.0


# ---------------------------------------------------------------- M3: cross-validated rule choice + coverage


def test_stratified_folds_cover_each_item_once():
    import numpy as np

    from copilot.evals.calibrate import stratified_folds

    y = np.array([1] * 10 + [0] * 30, dtype=bool)
    folds = stratified_folds(y, 5, seed=0)
    allidx = np.concatenate(folds)
    assert sorted(allidx.tolist()) == list(range(40))
    assert all(y[f].sum() == 2 for f in folds)  # positives spread evenly


def test_cross_validate_prefers_informative_signal():
    import random

    from copilot.evals.calibrate import cross_validate

    rnd = random.Random(0)
    sig, lab = [], []
    for i in range(80):
        abstain = i % 4 == 0
        sig.append(
            {
                "dense": rnd.random(),
                "rerank": None,
                "bm25_norm": rnd.random(),
                "coverage": (0.2 if abstain else 0.9) + rnd.uniform(-0.1, 0.1),
            }
        )
        lab.append(abstain)
    cv = cross_validate(sig, lab, max_frr=0.1, k=4, repeats=2, available=["dense", "bm25_norm", "coverage"])
    assert cv.rule.thresholds.keys() == {"coverage"}
    assert cv.table[0]["cv_f1_mean"] > 0.9 and "cv_f1_mean" in cv.rule.tune_metrics


def test_term_coverage_idf_weighted():
    from copilot.retrieval.retriever import term_coverage

    idf = {"fee": 1.0, "pix": 5.0}.get
    assert term_coverage(["fee", "pix"], ["the late fee is 29"], idf) == pytest.approx(1 / 6)
    assert term_coverage(["fee", "pix"], ["pix fee"], idf) == 1.0
    assert term_coverage([], ["x"], idf) == 0.0
