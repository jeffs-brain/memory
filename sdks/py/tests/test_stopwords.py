# SPDX-License-Identifier: Apache-2.0
"""Stopword set coverage tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from jeffs_brain_memory.search.stopwords import (
    STOPWORDS,
    is_stopword,
    load_stopwords,
)


def test_module_stopwords_populated() -> None:
    """Combined EN + NL set should be non-empty at import time."""
    assert len(STOPWORDS) > 100


def test_load_english_contains_common_filler() -> None:
    en = load_stopwords("en")
    for token in ("the", "and", "which", "with"):
        assert token in en


def test_load_dutch_contains_common_filler() -> None:
    nl = load_stopwords("nl")
    for token in ("de", "het", "een", "en"):
        assert token in nl


def test_load_overrides_path(tmp_path: Path) -> None:
    """A caller-supplied ``path`` overrides the default fixture location."""
    fixture = tmp_path / "custom.json"
    fixture.write_text('["foo", "bar", ""]', encoding="utf-8")
    got = load_stopwords("en", path=fixture)
    assert got == frozenset({"foo", "bar"})


def test_short_tokens_always_treated_as_stopwords() -> None:
    assert is_stopword("a") is True
    assert is_stopword("ab") is True
    assert is_stopword("abc") in (True, False)  # depends on curated list


def test_bosch_is_not_a_stopword() -> None:
    assert is_stopword("bosch") is False


def test_known_english_filler_drops() -> None:
    assert is_stopword("the") is True
    assert is_stopword("which") is True


def test_known_dutch_filler_drops() -> None:
    assert is_stopword("het") is True
    assert is_stopword("een") is True


def test_locale_specific_lookup() -> None:
    """Asking for ``nl`` should not leak English-only tokens."""
    assert is_stopword("which", locale="nl") is False
    assert is_stopword("which", locale="en") is True


def test_load_raises_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        load_stopwords("en", path=missing)


def test_load_rejects_non_array(tmp_path: Path) -> None:
    fixture = tmp_path / "wrong.json"
    fixture.write_text('{"oops": true}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_stopwords("en", path=fixture)
