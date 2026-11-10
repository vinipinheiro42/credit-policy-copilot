import pytest

from copilot.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_matches_hand_computed_scores():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["d"] == pytest.approx(1 / 63)


def test_rrf_rewards_agreement_between_rankers():
    order = [d for d, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)]
    assert order[0] == "b"


def test_rrf_is_deterministic_on_ties_and_ignores_duplicates():
    r1 = reciprocal_rank_fusion([["x", "y"], ["y", "x"]])
    r2 = reciprocal_rank_fusion([["y", "x"], ["x", "y"]])
    assert [d for d, _ in r1] == [d for d, _ in r2] == ["x", "y"]
    dup = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=0))
    assert dup == {"a": 1.0, "b": 1 / 2}


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
    assert [d for d, _ in reciprocal_rank_fusion([["a"], []])] == ["a"]
