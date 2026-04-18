# SPDX-License-Identifier: Apache-2.0
"""Trigram index for the fuzzy fallback leg.

Ported from the Go SDK's ``sdks/go/search/trigram.go``. The index maps
boundary-padded 3-grams to the slugs that contain them. Jaccard
similarity over the trigram sets drives ranking; the default threshold
of ``0.3`` matches ``spec/ALGORITHMS.md`` (Trigram fallback details).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "TrigramIndex",
    "TrigramHit",
    "jaccard",
    "trigrams",
    "slug_text",
    "TRIGRAM_JACCARD_THRESHOLD",
]

TRIGRAM_JACCARD_THRESHOLD: float = 0.3
"""Minimum Jaccard similarity a slug must clear to survive the fallback."""

_DEFAULT_FUZZY_TOPK: int = 10


@dataclass(frozen=True, slots=True)
class TrigramHit:
    """One fuzzy match.

    Similarity is Jaccard over the padded trigram sets, in ``[0, 1]``.
    """

    path: str
    score: float


def trigrams(text: str) -> set[str]:
    """Return the set of boundary-padded 3-grams for ``text``.

    Text is lowercased and non-alphanumeric characters are replaced
    with spaces. Each whitespace-separated word is padded with a ``$``
    boundary marker at both ends so short-slug matches retain their
    word-boundary signal. Words shorter than three characters after
    padding are dropped.
    """
    out: set[str] = set()
    if not text:
        return out
    buf: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            buf.append(" ")
    cleaned = "".join(buf)
    for word in cleaned.split():
        padded = f"${word}$"
        if len(padded) < 3:
            continue
        for i in range(len(padded) - 2):
            out.add(padded[i : i + 3])
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    """Compute ``|A ∩ B| / |A ∪ B|`` for two trigram sets."""
    if not a or not b:
        return 0.0
    if len(b) < len(a):
        a, b = b, a
    intersection = sum(1 for g in a if g in b)
    union = len(a) + len(b) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def slug_text(path: str) -> str:
    """Return the whitespace-separated trigram source for ``path``.

    Mirrors Go's ``slugText``: lowercase, keep the last slash-separated
    segment, strip a trailing ``.md``, and collapse non-alphanumerics
    to spaces. ``clients/oude-reimer.md`` becomes ``oude reimer``.
    """
    s = path.lower()
    if "/" in s:
        s = s.rsplit("/", 1)[1]
    if s.endswith(".md"):
        s = s[:-3]
    buf: list[str] = []
    for ch in s:
        buf.append(ch if ch.isalnum() else " ")
    return " ".join("".join(buf).split())


class TrigramIndex:
    """Lazy in-memory trigram index keyed by slug text.

    Thread-safety: build-once-then-read. Mutation after
    :meth:`__init__` is not supported.
    """

    __slots__ = ("_index", "_paths", "_path_grams")

    def __init__(self, paths: Iterable[str]) -> None:
        self._index: dict[str, list[str]] = {}
        self._paths: list[str] = []
        self._path_grams: dict[str, set[str]] = {}

        seen: set[str] = set()
        for raw in paths:
            if not raw or raw in seen:
                continue
            seen.add(raw)
            self._paths.append(raw)
            grams = trigrams(slug_text(raw))
            self._path_grams[raw] = grams
            for gram in grams:
                self._index.setdefault(gram, []).append(raw)

    @property
    def paths(self) -> list[str]:
        """Return a fresh copy of the indexed slug set."""
        return list(self._paths)

    def fuzzy_search(
        self,
        query: str,
        top_k: int = _DEFAULT_FUZZY_TOPK,
        *,
        threshold: float = TRIGRAM_JACCARD_THRESHOLD,
    ) -> list[TrigramHit]:
        """Return the top-k slugs ranked by Jaccard similarity.

        ``threshold`` defaults to :data:`TRIGRAM_JACCARD_THRESHOLD`.
        Slugs below the threshold are discarded. Ties break on path
        ascending so the order is deterministic across runs.
        """
        if not self._index:
            return []
        if top_k <= 0:
            top_k = _DEFAULT_FUZZY_TOPK

        query_grams = trigrams(query)
        if not query_grams:
            return []

        candidates: set[str] = set()
        for gram in query_grams:
            for path in self._index.get(gram, ()):
                candidates.add(path)
        if not candidates:
            return []

        hits: list[TrigramHit] = []
        for path in candidates:
            grams = self._path_grams.get(path)
            if not grams:
                continue
            sim = jaccard(query_grams, grams)
            if sim < threshold:
                continue
            hits.append(TrigramHit(path=path, score=sim))

        hits.sort(key=lambda h: (-h.score, h.path))
        if len(hits) > top_k:
            hits = hits[:top_k]
        return hits
