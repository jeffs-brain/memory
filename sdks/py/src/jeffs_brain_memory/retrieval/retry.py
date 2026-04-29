# SPDX-License-Identifier: Apache-2.0
"""Retry ladder helpers: sanitisation, strongest term, trigram index.

Mirrors ``go/retrieval/retry.go``. The five-rung ladder itself is
orchestrated from :mod:`retriever`; this module exposes the pure
helpers every rung leans on.

Stop-word set stays intentionally duplicated from the Go reference so
the ladder keeps the same English + small Dutch overlap whether or not
the caller wires a larger stop-word payload through search.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .source import TrigramChunk

TRIGRAM_JACCARD_THRESHOLD = 0.3
"""Fixed by the spec. Raising it drops recall sharply."""

RETRY_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "is", "are", "was",
        "what", "who", "when", "where", "why", "how", "you", "for",
        "from", "about", "advice", "any", "been", "can", "choose",
        "current", "decide", "deciding", "feeling", "find", "getting",
        "help", "helpful", "idea", "ideas", "interesting", "ive",
        "look", "looking", "make", "making", "might", "need", "needs",
        "noticed", "planning", "recent", "recently", "recommend",
        "recommendation", "recommendations", "should", "some", "soon",
        "suggest", "suggestion", "suggestions", "sure", "thinking",
        "tips", "together", "trying", "upcoming", "useful", "want",
        "weekend", "with", "would", "again", "becoming", "bit",
        "combined", "having", "items", "keep", "keeping", "kind",
        "kinds", "lately", "many", "seen", "show", "tonight",
        "trouble", "type", "types", "watch", "have", "has", "had",
        "de", "het", "een", "en", "of",
    }
)


# Unicode punctuation + symbols. Matches the Go regexp ``[\p{P}\p{S}]+``.
_PUNCT_CATEGORIES = ("P", "S")


def _is_punct_or_symbol(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith(_PUNCT_CATEGORIES)


def sanitise_query(q: str) -> str:
    """Strip Unicode punctuation and symbols, collapse whitespace.

    Mirrors the Go reference and the TS port bit-for-bit.
    """
    if not q:
        return ""
    buf: list[str] = []
    for ch in q:
        buf.append(" " if _is_punct_or_symbol(ch) else ch)
    return " ".join("".join(buf).split())


def strongest_term(q: str) -> str:
    """Return the longest non-stop-word token of at least three chars."""
    best = ""
    for tok in _normalise_retry_tokens(q):
        if len(tok) < 3:
            continue
        if tok in RETRY_STOP_WORDS:
            continue
        if len(tok) > len(best):
            best = tok
    return best


def query_tokens(q: str) -> list[str]:
    """Return every non-stop-word token of length >= 3, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _normalise_retry_tokens(q):
        if len(tok) < 3:
            continue
        if tok in RETRY_STOP_WORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _normalise_retry_tokens(q: str) -> list[str]:
    cleaned = sanitise_query(q)
    if not cleaned:
        return []
    return cleaned.lower().split()


def force_refresh_index() -> None:
    """Documented no-op for SQLite / WAL backed indices.

    The spec keeps this call in the rung list so attempt traces stay
    diffable across SDKs; callers may re-wire it to drive a genuine
    reindex when a backend warrants it.
    """
    # Intentionally empty. See ``docs/ALGORITHMS.md`` rung 2.
    return None


# Non-alphanumeric squasher for slug text. Unicode-aware so accented
# characters survive.
_NON_ALNUM_RE = re.compile(r"[^\w]+", re.UNICODE)


def _collapse_non_alnum(s: str) -> str:
    return _NON_ALNUM_RE.sub(" ", s).strip()


def slug_text_for(p: str) -> str:
    """Keep only the filename stem so single-word queries match slugs
    without being drowned out by parent-directory noise."""
    s = p.lower()
    idx = s.rfind("/")
    if idx >= 0:
        s = s[idx + 1 :]
    if s.endswith(".md"):
        s = s[:-3]
    return _collapse_non_alnum(s)


def compute_trigrams(text: str) -> set[str]:
    """Return the ``$``-padded 3-gram set for ``text``."""
    out: set[str] = set()
    if not text:
        return out
    cleaned = _collapse_non_alnum(text.lower())
    for word in cleaned.split():
        padded = f"${word}$"
        if len(padded) < 3:
            continue
        for i in range(len(padded) - 2):
            out.add(padded[i : i + 3])
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    if union == 0:
        return 0.0
    return intersection / union


@dataclass(slots=True)
class TrigramHit:
    id: str = ""
    path: str = ""
    similarity: float = 0.0
    title: str = ""
    summary: str = ""
    content: str = ""


@dataclass(slots=True)
class _TrigramEntry:
    id: str
    path: str
    grams: set[str]
    title: str
    summary: str
    content: str


@dataclass(slots=True)
class TrigramIndex:
    """Lazy trigram index over slug text derived from chunk paths."""

    entries: list[_TrigramEntry] = field(default_factory=list)
    by_gram: dict[str, list[int]] = field(default_factory=dict)

    def search(self, tokens: list[str], limit: int) -> list[TrigramHit]:
        if not tokens or not self.entries or limit <= 0:
            return []

        best: dict[str, tuple[int, float]] = {}
        for tok in tokens:
            q_grams = compute_trigrams(tok)
            if not q_grams:
                continue
            candidate_positions: set[int] = set()
            for g in q_grams:
                for pos in self.by_gram.get(g, []):
                    candidate_positions.add(pos)
            for pos in candidate_positions:
                entry = self.entries[pos]
                if not entry.grams:
                    continue
                sim = jaccard(q_grams, entry.grams)
                if sim < TRIGRAM_JACCARD_THRESHOLD:
                    continue
                prev = best.get(entry.id)
                if prev is None or sim > prev[1]:
                    best[entry.id] = (pos, sim)

        hits: list[TrigramHit] = []
        for pos, sim in best.values():
            e = self.entries[pos]
            hits.append(
                TrigramHit(
                    id=e.id,
                    path=e.path,
                    similarity=sim,
                    title=e.title,
                    summary=e.summary,
                    content=e.content,
                )
            )

        hits.sort(key=lambda h: (-h.similarity, h.path))
        if len(hits) > limit:
            hits = hits[:limit]
        return hits


def build_trigram_index(chunks: list[TrigramChunk]) -> TrigramIndex:
    """Construct a slug trigram index. Duplicate IDs collapse to the
    first occurrence."""
    idx = TrigramIndex()
    seen: set[str] = set()
    for c in chunks:
        if not c.id or c.id in seen:
            continue
        seen.add(c.id)
        grams = compute_trigrams(slug_text_for(c.path))
        entry = _TrigramEntry(
            id=c.id,
            path=c.path,
            grams=grams,
            title=c.title,
            summary=c.summary,
            content=c.content,
        )
        pos = len(idx.entries)
        idx.entries.append(entry)
        for g in grams:
            idx.by_gram.setdefault(g, []).append(pos)
    return idx
