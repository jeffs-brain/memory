# SPDX-License-Identifier: Apache-2.0
"""Trigram index tests."""

from __future__ import annotations

import pytest

from jeffs_brain_memory.search.trigram import (
    TRIGRAM_JACCARD_THRESHOLD,
    TrigramIndex,
    jaccard,
    slug_text,
    trigrams,
)


def test_trigrams_empty_input() -> None:
    assert trigrams("") == set()


def test_trigrams_single_word() -> None:
    assert trigrams("bosch") == {"$bo", "bos", "osc", "sch", "ch$"}


def test_trigrams_multi_word_padding() -> None:
    expected = {
        "$ou",
        "oud",
        "ude",
        "de$",
        "$re",
        "rei",
        "eim",
        "ime",
        "mer",
        "er$",
    }
    assert trigrams("oude reimer") == expected


def test_trigrams_punctuation_becomes_whitespace() -> None:
    expected = {
        "$ou",
        "oud",
        "ude",
        "de$",
        "$re",
        "rei",
        "eim",
        "ime",
        "mer",
        "er$",
        "$md",
        "md$",
    }
    assert trigrams("oude-reimer.md") == expected


def test_trigrams_case_folded() -> None:
    assert trigrams("BOSCH") == {"$bo", "bos", "osc", "sch", "ch$"}


def test_trigrams_short_word_keeps_boundary() -> None:
    assert trigrams("ai") == {"$ai", "ai$"}


def test_trigrams_digits_preserved() -> None:
    expected = {
        "$v2",
        "v2$",
        "$pl",
        "pla",
        "lan",
        "an$",
    }
    assert trigrams("v2 plan") == expected


def test_jaccard_of_disjoint_sets_is_zero() -> None:
    assert jaccard({"abc"}, {"xyz"}) == 0.0


def test_jaccard_of_identical_sets_is_one() -> None:
    assert jaccard({"abc", "bcd"}, {"abc", "bcd"}) == 1.0


def test_jaccard_empty_set_is_zero() -> None:
    assert jaccard(set(), {"abc"}) == 0.0


def test_slug_text_strips_md_and_path() -> None:
    assert slug_text("clients/oude-reimer.md") == "oude reimer"


def test_slug_text_handles_no_slash() -> None:
    assert slug_text("bosch.md") == "bosch"


def test_slug_text_lowercases() -> None:
    # ``.MD`` lowercases to ``.md``, which is then stripped as the
    # extension by :func:`slug_text`.
    assert slug_text("clients/BOSCH.MD") == "bosch"


def test_slug_text_preserves_non_md_extension() -> None:
    assert slug_text("clients/bosch.txt") == "bosch txt"


def test_build_trigram_index_populates_paths() -> None:
    idx = TrigramIndex(
        [
            "clients/oude-reimer.md",
            "clients/bosch.md",
            "projects/a-ware.md",
        ]
    )
    assert len(idx.paths) == 3


def test_build_trigram_index_deduplicates_paths() -> None:
    idx = TrigramIndex(["clients/bosch.md", "clients/bosch.md"])
    assert len(idx.paths) == 1


def test_fuzzy_exact_match_ranks_first() -> None:
    idx = TrigramIndex(
        [
            "clients/oude-reimer.md",
            "clients/bosch.md",
        ]
    )
    hits = idx.fuzzy_search("oude", top_k=5)
    assert hits
    assert hits[0].path == "clients/oude-reimer.md"
    assert hits[0].score > 0.0


def test_fuzzy_typo_match() -> None:
    idx = TrigramIndex(
        [
            "clients/oude-reimer.md",
            "clients/bosch.md",
            "projects/royal-aware.md",
        ]
    )
    hits = idx.fuzzy_search("dude reimer", top_k=5)
    assert hits
    assert hits[0].path == "clients/oude-reimer.md"
    assert 0 < hits[0].score < 1.0


def test_fuzzy_miss_returns_empty() -> None:
    idx = TrigramIndex(
        [
            "clients/oude-reimer.md",
            "clients/bosch.md",
        ]
    )
    assert idx.fuzzy_search("kubernetes", top_k=5) == []


def test_fuzzy_threshold_is_respected() -> None:
    idx = TrigramIndex(["clients/oude-reimer.md", "projects/royal-aware.md"])
    strict = idx.fuzzy_search("oude", top_k=5, threshold=0.99)
    assert strict == []


def test_jaccard_threshold_constant_matches_spec() -> None:
    assert TRIGRAM_JACCARD_THRESHOLD == 0.3


def test_fuzzy_empty_query_returns_empty() -> None:
    idx = TrigramIndex(["clients/bosch.md"])
    assert idx.fuzzy_search("", top_k=5) == []


def test_fuzzy_top_k_caps_output() -> None:
    paths = [f"clients/{slug}-reimer.md" for slug in ("oude", "oudy", "oudz", "oudq")]
    idx = TrigramIndex(paths)
    hits = idx.fuzzy_search("oude reimer", top_k=2)
    assert len(hits) <= 2


@pytest.mark.parametrize(
    "query,expected_top",
    [
        ("bosch", "clients/bosch.md"),
        ("oude", "clients/oude-reimer.md"),
    ],
)
def test_fuzzy_search_is_deterministic(query: str, expected_top: str) -> None:
    idx = TrigramIndex(
        [
            "clients/oude-reimer.md",
            "clients/bosch.md",
            "projects/royal-aware.md",
        ]
    )
    assert idx.fuzzy_search(query, top_k=3)[0].path == expected_top


def test_trigram_index_tie_break_on_path() -> None:
    """Equal similarity ties must break on path ascending."""
    idx = TrigramIndex(["b/foo.md", "a/foo.md"])
    hits = idx.fuzzy_search("foo", top_k=5)
    assert [hit.path for hit in hits] == ["a/foo.md", "b/foo.md"]
