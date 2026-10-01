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
    assert trigrams("zenco") == {"$ze", "zen", "enc", "nco", "co$"}


def test_trigrams_multi_word_padding() -> None:
    expected = {
        "$mi",
        "mil",
        "ill",
        "ll$",
        "$br",
        "bro",
        "roo",
        "ook",
        "ok$",
    }
    assert trigrams("mill brook") == expected


def test_trigrams_punctuation_becomes_whitespace() -> None:
    expected = {
        "$mi",
        "mil",
        "ill",
        "ll$",
        "$br",
        "bro",
        "roo",
        "ook",
        "ok$",
        "$md",
        "md$",
    }
    assert trigrams("mill-brook.md") == expected


def test_trigrams_case_folded() -> None:
    assert trigrams("ZENCO") == {"$ze", "zen", "enc", "nco", "co$"}


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
    assert slug_text("clients/mill-brook.md") == "mill brook"


def test_slug_text_handles_no_slash() -> None:
    assert slug_text("zenco.md") == "zenco"


def test_slug_text_lowercases() -> None:
    # ``.MD`` lowercases to ``.md``, which is then stripped as the
    # extension by :func:`slug_text`.
    assert slug_text("clients/ZENCO.MD") == "zenco"


def test_slug_text_preserves_non_md_extension() -> None:
    assert slug_text("clients/zenco.txt") == "zenco txt"


def test_build_trigram_index_populates_paths() -> None:
    idx = TrigramIndex(
        [
            "clients/mill-brook.md",
            "clients/zenco.md",
            "projects/e-volt.md",
        ]
    )
    assert len(idx.paths) == 3


def test_build_trigram_index_deduplicates_paths() -> None:
    idx = TrigramIndex(["clients/zenco.md", "clients/zenco.md"])
    assert len(idx.paths) == 1


def test_fuzzy_exact_match_ranks_first() -> None:
    idx = TrigramIndex(
        [
            "clients/mill-brook.md",
            "clients/zenco.md",
        ]
    )
    hits = idx.fuzzy_search("mill", top_k=5)
    assert hits
    assert hits[0].path == "clients/mill-brook.md"
    assert hits[0].score > 0.0


def test_fuzzy_typo_match() -> None:
    idx = TrigramIndex(
        [
            "clients/mill-brook.md",
            "clients/zenco.md",
            "projects/nova-evolt.md",
        ]
    )
    hits = idx.fuzzy_search("hill brook", top_k=5)
    assert hits
    assert hits[0].path == "clients/mill-brook.md"
    assert 0 < hits[0].score < 1.0


def test_fuzzy_miss_returns_empty() -> None:
    idx = TrigramIndex(
        [
            "clients/mill-brook.md",
            "clients/zenco.md",
        ]
    )
    assert idx.fuzzy_search("kubernetes", top_k=5) == []


def test_fuzzy_threshold_is_respected() -> None:
    idx = TrigramIndex(["clients/mill-brook.md", "projects/nova-evolt.md"])
    strict = idx.fuzzy_search("mill", top_k=5, threshold=0.99)
    assert strict == []


def test_jaccard_threshold_constant_matches_spec() -> None:
    assert TRIGRAM_JACCARD_THRESHOLD == 0.3


def test_fuzzy_empty_query_returns_empty() -> None:
    idx = TrigramIndex(["clients/zenco.md"])
    assert idx.fuzzy_search("", top_k=5) == []


def test_fuzzy_top_k_caps_output() -> None:
    paths = [f"clients/{slug}-brook.md" for slug in ("mill", "milx", "milz", "milq")]
    idx = TrigramIndex(paths)
    hits = idx.fuzzy_search("mill brook", top_k=2)
    assert len(hits) <= 2


@pytest.mark.parametrize(
    "query,expected_top",
    [
        ("zenco", "clients/zenco.md"),
        ("mill", "clients/mill-brook.md"),
    ],
)
def test_fuzzy_search_is_deterministic(query: str, expected_top: str) -> None:
    idx = TrigramIndex(
        [
            "clients/mill-brook.md",
            "clients/zenco.md",
            "projects/nova-evolt.md",
        ]
    )
    assert idx.fuzzy_search(query, top_k=3)[0].path == expected_top


def test_trigram_index_tie_break_on_path() -> None:
    """Equal similarity ties must break on path ascending."""
    idx = TrigramIndex(["b/foo.md", "a/foo.md"])
    hits = idx.fuzzy_search("foo", top_k=5)
    assert [hit.path for hit in hits] == ["a/foo.md", "b/foo.md"]
