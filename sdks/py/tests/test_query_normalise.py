# SPDX-License-Identifier: Apache-2.0
"""Tests for query.normalise — ported from the Go unit tests."""

from __future__ import annotations

import pytest

from jeffs_brain_memory.query.normalise import (
    STOP_WORD_SET,
    count_significant_terms,
    count_tokens,
    normalise_for_cache,
    significant_terms,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("hello", 1),
        ("hello world", 2),
        ("  hello   world  ", 2),
        ("one two three four five", 5),
        ("tabs\there\ttoo", 3),
        ("newlines\nwork\ntoo", 3),
    ],
)
def test_count_tokens(text: str, expected: int) -> None:
    assert count_tokens(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("the is a", 0),
        ("kubernetes deployment rollback", 3),
        ("what is kubernetes?", 1),
        ("error: panic in goroutine", 3),
    ],
)
def test_count_significant_terms(text: str, expected: int) -> None:
    # Use the module-level stopword set loaded from the spec fixtures.
    assert count_significant_terms(text) == expected


def test_significant_terms_lowercased_and_stripped() -> None:
    terms = significant_terms("  Kubernetes, Deployment!  Rollback.  ")
    assert terms == ["kubernetes", "deployment", "rollback"]


def test_stopword_set_loaded_from_fixture() -> None:
    # Sanity: the loader read the spec-shipped English stopword list.
    assert "the" in STOP_WORD_SET
    assert "and" in STOP_WORD_SET
    # 183 entries in the current fixture; treat count as a lower bound.
    assert len(STOP_WORD_SET) >= 100


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  Hello  World  ", "hello world"),
        ("UPPER", "upper"),
        ("zero\u200bwidth", "zerowidth"),
        ("non\u00a0breaking", "non breaking"),
        ("bom\ufeffchar", "bomchar"),
        ("tabs\there", "tabs here"),
        ("multi\n\nline", "multi line"),
    ],
)
def test_normalise_for_cache(text: str, expected: str) -> None:
    assert normalise_for_cache(text) == expected


def test_count_significant_terms_with_custom_stopset() -> None:
    # Passing an explicit set overrides the global fixture-backed default.
    assert count_significant_terms("foo bar baz", {"foo"}) == 2
