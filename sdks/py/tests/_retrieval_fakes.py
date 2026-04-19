# SPDX-License-Identifier: Apache-2.0
"""Shared in-memory fakes for the retrieval test suite.

Mirrors the Go ``fake_source_test.go``: a deterministic token-count
BM25 and a fake-embedder cosine for the vector leg so the tests reach
the orchestration paths without standing up FTS5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

from jeffs_brain_memory.llm.fake import FakeEmbedder
from jeffs_brain_memory.retrieval import (
    BM25Hit,
    Filters,
    Source,
    TrigramChunk,
    VectorHit,
)


@dataclass(slots=True)
class FakeChunk:
    id: str = ""
    path: str = ""
    title: str = ""
    summary: str = ""
    content: str = ""
    tags: list[str] = field(default_factory=list)
    scope: str = ""


def _tokenise_expr(expr: str) -> list[str]:
    lowered = expr.lower()
    for ch in "()\"*^:":
        lowered = lowered.replace(ch, " ")
    out: list[str] = []
    for tok in lowered.split():
        if tok in ("and", "or", "not"):
            continue
        if not tok:
            continue
        out.append(tok)
    return out


def _matches_filter(chunk: FakeChunk, filters: Filters) -> bool:
    if not filters.matches_path(chunk.path):
        return False
    if filters.scope and chunk.scope != filters.scope:
        return False
    if filters.tags:
        for want in filters.tags:
            if want not in chunk.tags:
                return False
    return True


def _chunk_id(c: FakeChunk) -> str:
    return c.id if c.id else c.path


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    aa = 0.0
    bb = 0.0
    for av, bv in zip(a, b):
        dot += av * bv
        aa += av * av
        bb += bv * bv
    if aa == 0.0 or bb == 0.0:
        return 0.0
    return dot / (math.sqrt(aa) * math.sqrt(bb))


class FakeSource:
    """In-memory :class:`Source` used by the retrieval tests."""

    embed_dim: int = 16

    def __init__(self, chunks: list[FakeChunk]) -> None:
        self.chunks_data: list[FakeChunk] = list(chunks)
        self.bm25_fail: Exception | None = None
        self.vector_fail: Exception | None = None
        self.bm25_calls: list[str] = []
        # Signature: (expr) -> (hits, override) — when override is True
        # the returned list replaces the computed BM25 list for that
        # call. Used by retry-ladder tests to force zero hits.
        self.bm25_override: Callable[[str], tuple[list[BM25Hit], bool]] | None = None

    async def search_bm25(
        self, expr: str, k: int, filters: Filters
    ) -> list[BM25Hit]:
        self.bm25_calls.append(expr)
        if self.bm25_fail is not None:
            raise self.bm25_fail
        if self.bm25_override is not None:
            hits, override = self.bm25_override(expr)
            if override:
                return hits

        tokens = _tokenise_expr(expr)
        if not tokens:
            return []

        scored: list[tuple[FakeChunk, int]] = []
        for c in self.chunks_data:
            if not _matches_filter(c, filters):
                continue
            corpus = " \n ".join(
                [c.path, c.title, c.summary, c.content]
            ).lower()
            score = 0
            for t in tokens:
                score += corpus.count(t)
            if score == 0:
                continue
            scored.append((c, score))

        scored.sort(key=lambda pair: (-pair[1], pair[0].path))
        if k > 0 and len(scored) > k:
            scored = scored[:k]

        return [
            BM25Hit(
                id=_chunk_id(c),
                path=c.path,
                title=c.title,
                summary=c.summary,
                content=c.content,
                score=float(s),
            )
            for c, s in scored
        ]

    async def search_vector(
        self, embedding: Sequence[float], k: int, filters: Filters
    ) -> list[VectorHit]:
        if self.vector_fail is not None:
            raise self.vector_fail
        if not embedding:
            return []
        embedder = FakeEmbedder(self.embed_dim)
        scored: list[tuple[FakeChunk, float]] = []
        for c in self.chunks_data:
            if not _matches_filter(c, filters):
                continue
            seed = " ".join(
                part for part in (c.title, c.summary, c.content) if part
            ).strip()
            if not seed:
                seed = c.path
            vecs = await embedder.embed([seed])
            if not vecs:
                continue
            scored.append((c, _cosine(embedding, vecs[0])))
        scored.sort(key=lambda pair: (-pair[1], pair[0].path))
        if k > 0 and len(scored) > k:
            scored = scored[:k]
        return [
            VectorHit(
                id=_chunk_id(c),
                path=c.path,
                title=c.title,
                summary=c.summary,
                content=c.content,
                similarity=sim,
            )
            for c, sim in scored
        ]

    async def chunks(self) -> list[TrigramChunk]:
        return [
            TrigramChunk(
                id=_chunk_id(c),
                path=c.path,
                title=c.title,
                summary=c.summary,
                content=c.content,
            )
            for c in self.chunks_data
        ]


# Protocol sanity: FakeSource conforms to Source at import time.
_source: Source = FakeSource([])  # pragma: no cover
