"""The golden set is verified against the real corpus: every gold section must resolve to exactly one
parsed section and every evidence quote must appear in it (whitespace/case-insensitive)."""

from collections import Counter

import pytest

from copilot.config import REPO_ROOT
from copilot.evals.golden import CATEGORIES, load_corpus_sections, load_golden, section_matches, verify_item

GOLDEN = REPO_ROOT / "evals" / "golden_set.yaml"
CORPUS = REPO_ROOT / "data" / "corpus"


@pytest.fixture(scope="module")
def items():
    return load_golden(GOLDEN)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus_sections(CORPUS, CORPUS / "MANIFEST.yaml")


def test_size_ids_and_mix(items):
    assert len(items) >= 120
    ids = [i.id for i in items]
    assert len(ids) == len(set(ids))
    cats = Counter(i.category for i in items)
    assert set(cats) == set(CATEGORIES)
    assert cats["unanswerable"] >= 10 and cats["out_of_scope"] >= 8 and cats["adversarial"] >= 6
    assert sum(1 for i in items if i.lang == "pt") >= 15


def test_splits_are_stratified_about_25_percent(items):
    assert {i.split for i in items} == {"tune", "heldout"}
    share = sum(i.split == "heldout" for i in items) / len(items)
    assert 0.2 <= share <= 0.3
    for cat in CATEGORIES:
        assert any(i.split == "heldout" and i.category == cat for i in items), cat


def test_every_item_verified_against_corpus(items, corpus):
    problems = {i.id: p for i in items if (p := verify_item(i, corpus))}
    assert not problems, problems


def test_section_suffix_matching():
    assert section_matches("Art. 3º", "CAPÍTULO II — DAS DEFINIÇÕES > Art. 3º")
    assert not section_matches("Art. 3º", "CAPÍTULO II > Art. 31º")
    assert not section_matches("3º", "CAPÍTULO II > Art. 3º")
    assert section_matches("7. Retention", "7. Retention")
