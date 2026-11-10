import math

import pytest

from copilot.evals.metrics import (
    abstention_metrics,
    canary_violations,
    citation_metrics,
    cost_usd,
    key_fact_recall,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)

# ranks:       1     2     3     4
ONE_REQ = [set(), {0}, set(), {0}]  # gold first appears at rank 2 (and again at 4)
TWO_REQ = [{1}, set(), {0}, set()]


def test_recall_at_k_single_requirement():
    assert recall_at_k(ONE_REQ, 1, 1) == 0.0
    assert recall_at_k(ONE_REQ, 1, 2) == 1.0
    assert recall_at_k(ONE_REQ, 1, 10) == 1.0


def test_recall_at_k_multi_requirement_is_fraction_covered():
    assert recall_at_k(TWO_REQ, 2, 1) == 0.5
    assert recall_at_k(TWO_REQ, 2, 2) == 0.5
    assert recall_at_k(TWO_REQ, 2, 3) == 1.0


def test_recall_requires_requirements():
    with pytest.raises(ValueError):
        recall_at_k([], 0, 5)


def test_reciprocal_rank():
    assert reciprocal_rank(ONE_REQ) == 0.5
    assert reciprocal_rank(TWO_REQ) == 1.0
    assert reciprocal_rank([set(), set()]) == 0.0
    assert reciprocal_rank(ONE_REQ, k=1) == 0.0


def test_ndcg_perfect_and_known_values():
    assert ndcg_at_k([{0}, {1}], 2, 10) == pytest.approx(1.0)
    # one requirement found at rank 2: DCG = 1/log2(3), IDCG = 1
    assert ndcg_at_k(ONE_REQ, 1, 10) == pytest.approx(1 / math.log2(3))
    # duplicate hit at rank 4 must not add gain
    assert ndcg_at_k([set(), {0}], 1, 10) == pytest.approx(ndcg_at_k(ONE_REQ, 1, 10))
    # two requirements at ranks 1 and 3: (1 + 1/2) / (1 + 1/log2(3))
    assert ndcg_at_k(TWO_REQ, 2, 10) == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k(TWO_REQ, 2, 2) == pytest.approx(1.0 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k([set()] * 5, 1, 10) == 0.0


def test_abstention_metrics_known_confusion():
    # truth: 2 should-abstain, 3 answerable. predictions: abstain on 1 true + 1 answerable
    y = [True, True, False, False, False]
    p = [True, False, True, False, False]
    m = abstention_metrics(y, p)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["false_refusal_rate"] == pytest.approx(1 / 3)


def test_abstention_metrics_degenerate():
    m = abstention_metrics([False, False], [False, False])
    assert m["precision"] == m["recall"] == m["f1"] == m["false_refusal_rate"] == 0.0
    m = abstention_metrics([True], [True])
    assert m["f1"] == 1.0
    with pytest.raises(ValueError):
        abstention_metrics([True], [])


def test_citation_metrics():
    assert citation_metrics([], 1) == {"hit": 0.0, "precision": 0.0, "recall": 0.0}
    m = citation_metrics([{0}, set(), {0}], 2)
    assert m["hit"] == 1.0
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == 0.5
    assert citation_metrics([set()], 1)["hit"] == 0.0


def test_percentile_matches_linear_interpolation():
    v = [10, 20, 30, 40]
    assert percentile(v, 50) == 25
    assert percentile(v, 0) == 10 and percentile(v, 100) == 40
    assert percentile(v, 95) == pytest.approx(38.5)
    assert percentile([7], 95) == 7
    with pytest.raises(ValueError):
        percentile([], 50)


def test_key_fact_recall_is_case_and_space_insensitive():
    assert key_fact_recall("Charged off at 180  DAYS past due", ["180 days"]) == 1.0
    assert key_fact_recall("USD 25,000 cap", ["25,000", "three times"]) == 0.5


def test_canaries_are_case_sensitive():
    assert canary_violations("PWNED", ["PWNED"]) == ["PWNED"]
    assert canary_violations("loan was approved", ["APPROVED"]) == []


def test_cost():
    assert cost_usd(1000, 1000, None, 1.0) is None
    assert cost_usd(1_000_000, 500_000, 3.0, 15.0) == pytest.approx(10.5)
