# SPDX-License-Identifier: Apache-2.0
"""Reciprocal Rank Fusion (Cormack, Clarke, Buettcher 2009).

Merges an arbitrary number of ranked lists into a single ranking with
``rrf_score(doc) = sum over lists of 1 / (k + rank(doc) + 1)``. Order
of the input lists does not change the fused score; it only controls
which list seeds the metadata for a candidate seen in multiple lists.

Mirrors ``sdks/go/retrieval/rrf.go`` bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import RetrievedChunk

RRF_DEFAULT_K = 60
"""Canonical constant. The spec pins k = 60 across every SDK."""


@dataclass(slots=True)
class RRFCandidate:
    """Input shape accepted by :func:`reciprocal_rank_fusion`."""

    id: str = ""
    path: str = ""
    title: str = ""
    summary: str = ""
    content: str = ""
    bm25_rank: int = 0
    have_bm25_rank: bool = False
    vector_similarity: float = 0.0
    have_vector_sim: bool = False


@dataclass(slots=True)
class _Bucket:
    id: str
    path: str
    title: str
    summary: str
    content: str
    bm25_rank: int
    have_bm25_rank: bool
    vector_similarity: float
    have_vector_sim: bool
    score: float


def reciprocal_rank_fusion(
    lists: list[list[RRFCandidate]], k: int = RRF_DEFAULT_K
) -> list[RetrievedChunk]:
    """Fuse ranked lists into a single ranking.

    The one-way fill rule from the spec: later lists seed metadata
    (title/summary/content/bm25_rank/vector_similarity) only when the
    first-seen entry left them empty. Ties on score break by path
    ascending for stable output.
    """
    safe_k = k if k > 0 else RRF_DEFAULT_K

    buckets: dict[str, _Bucket] = {}
    order: list[str] = []

    for candidates in lists:
        for rank, cand in enumerate(candidates):
            if not cand.id:
                continue
            contribution = 1.0 / float(safe_k + rank + 1)
            existing = buckets.get(cand.id)
            if existing is None:
                buckets[cand.id] = _Bucket(
                    id=cand.id,
                    path=cand.path,
                    title=cand.title,
                    summary=cand.summary,
                    content=cand.content,
                    bm25_rank=cand.bm25_rank if cand.have_bm25_rank else 0,
                    have_bm25_rank=cand.have_bm25_rank,
                    vector_similarity=(
                        cand.vector_similarity if cand.have_vector_sim else 0.0
                    ),
                    have_vector_sim=cand.have_vector_sim,
                    score=contribution,
                )
                order.append(cand.id)
                continue

            if not existing.title and cand.title:
                existing.title = cand.title
            if not existing.summary and cand.summary:
                existing.summary = cand.summary
            if not existing.content and cand.content:
                existing.content = cand.content
            if not existing.have_bm25_rank and cand.have_bm25_rank:
                existing.bm25_rank = cand.bm25_rank
                existing.have_bm25_rank = True
            if not existing.have_vector_sim and cand.have_vector_sim:
                existing.vector_similarity = cand.vector_similarity
                existing.have_vector_sim = True
            existing.score += contribution

    out: list[RetrievedChunk] = []
    for id_ in order:
        b = buckets[id_]
        chunk = RetrievedChunk(
            chunk_id=b.id,
            document_id=b.id,
            path=b.path,
            score=b.score,
            text=b.content,
            title=b.title,
            summary=b.summary,
        )
        if b.have_bm25_rank:
            chunk.bm25_rank = b.bm25_rank
        if b.have_vector_sim:
            chunk.vector_similarity = b.vector_similarity
        out.append(chunk)

    # Stable sort: score desc, then path asc.
    out.sort(key=lambda c: (-c.score, c.path))
    return out
