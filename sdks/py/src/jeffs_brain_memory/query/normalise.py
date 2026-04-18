# SPDX-License-Identifier: Apache-2.0
"""Normalisation helpers: whitespace-aware token and term counts.

Ports the Go ``query/normalise.go`` surface. The stopword list is loaded
from ``spec/fixtures/stopwords/en.json`` so that every SDK shares the same
source of truth.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Final

__all__ = [
    "STOP_WORD_SET",
    "count_tokens",
    "count_significant_terms",
    "normalise_for_cache",
    "tokenise",
    "significant_terms",
]


def _load_stopwords() -> frozenset[str]:
    # Walk up from this file to find the memory/spec fixtures directory.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "spec" / "fixtures" / "stopwords" / "en.json"
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return frozenset(str(w).lower() for w in data)
    raise FileNotFoundError(
        "query.normalise: could not locate spec/fixtures/stopwords/en.json"
    )


STOP_WORD_SET: Final[frozenset[str]] = _load_stopwords()


def count_tokens(text: str) -> int:
    """Return a whitespace-based token count. Mirrors Go ``countTokens``."""

    count = 0
    in_token = False
    for ch in text:
        if ch.isspace():
            if in_token:
                count += 1
                in_token = False
        else:
            in_token = True
    if in_token:
        count += 1
    return count


def _strip_punct(word: str) -> str:
    # Mirror Go's ``strings.TrimFunc`` with "non letter and non digit" as the
    # predicate: only trim leading and trailing non-alphanumeric runes.
    start = 0
    end = len(word)
    while start < end and not (word[start].isalpha() or word[start].isdigit()):
        start += 1
    while end > start and not (word[end - 1].isalpha() or word[end - 1].isdigit()):
        end -= 1
    return word[start:end]


def count_significant_terms(text: str, stopwords: set[str] | frozenset[str] | None = None) -> int:
    """Return non-stopword tokens. Mirrors Go ``countSignificantTerms``."""

    stops = stopwords if stopwords is not None else STOP_WORD_SET
    count = 0
    for raw_word in text.split():
        cleaned = _strip_punct(raw_word).lower()
        if cleaned and cleaned not in stops:
            count += 1
    return count


def tokenise(text: str) -> list[str]:
    """Return the whitespace-split token list used by :class:`Query`."""

    return text.split()


def significant_terms(
    text: str, stopwords: set[str] | frozenset[str] | None = None
) -> list[str]:
    """Return the list of non-stopword terms in ``text``."""

    stops = stopwords if stopwords is not None else STOP_WORD_SET
    out: list[str] = []
    for raw_word in text.split():
        cleaned = _strip_punct(raw_word).lower()
        if cleaned and cleaned not in stops:
            out.append(cleaned)
    return out


def normalise_for_cache(text: str) -> str:
    """NFC, strip zero-width/BOM, fold NBSP, collapse whitespace, lowercase.

    Matches the Go ``normaliseForCache`` helper bit-for-bit.
    """

    text = unicodedata.normalize("NFC", text.strip())
    text = text.lower()
    out: list[str] = []
    prev_space = False
    for ch in text:
        if ch in ("\u200b", "\ufeff"):
            continue
        if ch == "\u00a0":
            ch = " "
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                prev_space = True
        else:
            out.append(ch)
            prev_space = False
    return "".join(out).strip()
