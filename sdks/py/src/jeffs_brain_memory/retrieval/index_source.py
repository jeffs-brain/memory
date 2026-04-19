# SPDX-License-Identifier: Apache-2.0
"""Adapter from a ``search``-style index to the retrieval
:class:`Source` protocol.

Mirrors ``sdks/go/retrieval/index_source.go``. The Python ``search``
package is still a stub, so this adapter takes a duck-typed index
object exposing ``search_bm25``, ``search_vector`` and ``all_rows``
methods. Callers wiring a real FTS5 backend later only have to supply
matching shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Protocol, Sequence, runtime_checkable

from .source import BM25Hit, TrigramChunk, VectorHit
from .types import Filters

FILTER_FETCH_MULTIPLIER = 4
FILTER_FETCH_MAX_MULTIPLIER = 8


@dataclass(slots=True)
class IndexedRow:
    """Search-layer row. Matches the Go ``search.IndexedRow``."""

    path: str = ""
    title: str = ""
    summary: str = ""
    content: str = ""
    snippet: str = ""
    tags: str = ""
    scope: str = ""
    project_slug: str = ""
    session_date: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@runtime_checkable
class SearchIndex(Protocol):
    """Minimal interface the adapter expects from a search index."""

    async def search_bm25(
        self, expr: str, k: int, filters: Filters
    ) -> list[IndexedRow]: ...

    async def all_rows(self) -> list[IndexedRow]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Minimal interface for a companion vector store."""

    async def search(
        self, embedding: Sequence[float], model: str, k: int, filters: Filters
    ) -> list[IndexedRow]: ...


def _scope_matches_filter(row_scope: str, want: str) -> bool:
    if not want:
        return True
    aliases = {
        "memory": {"global_memory", "project_memory"},
        "global": {"global_memory"},
        "global_memory": {"global_memory"},
        "project": {"project_memory"},
        "project_memory": {"project_memory"},
        "agent": {"agent_memory"},
        "agent_memory": {"agent_memory"},
        "raw": {"raw_document", "raw_lme"},
        "raw_document": {"raw_document"},
        "raw_lme": {"raw_lme"},
        "wiki": {"wiki"},
        "sources": {"sources"},
    }
    expected = aliases.get(want.strip().lower(), {want.strip().lower()})
    return row_scope.strip().lower() in expected if row_scope else True


def _row_passes_filters(row: IndexedRow, filters: Filters) -> bool:
    """Report whether ``row`` satisfies the filter fields."""
    if not filters.matches_path(row.path):
        return False

    if not _scope_matches_filter(row.scope, filters.scope):
        return False
    project = filters.project.strip().lower()
    row_project = row.project_slug.strip().lower()
    if project and row_project and row_project != project:
        return False
    if filters.tags:
        row_tags = {tag.strip().lower() for tag in row.tags.split() if tag.strip()}
        for tag in filters.tags:
            if tag.strip().lower() not in row_tags:
                return False
    return True


def _fetch_limit(limit: int) -> int:
    return limit if limit > 0 else 20


def _has_retrieval_filters(filters: Filters) -> bool:
    return filters.has_any()


class IndexSource:
    """Adapts a search index (plus optional vector store) to the
    retrieval :class:`Source` contract.

    The Go SDK collapses document-level rows into chunk identity by
    reusing the path; the Python adapter follows the same convention so
    callers that want real chunk fragmentation can layer their own
    segmentation in front and provide a different :class:`Source`.
    """

    def __init__(
        self,
        search_index: SearchIndex,
        embedder: object | None = None,
        *,
        vectors: VectorStore | None = None,
        model: str = "",
    ) -> None:
        if search_index is None:
            raise ValueError(
                "retrieval: IndexSource requires a non-nil search index"
            )
        if vectors is not None and not model:
            raise ValueError(
                "retrieval: IndexSource: model is required when vectors is set"
            )
        self._index = search_index
        self._vectors = vectors
        self._embedder = embedder
        self._model = model

    async def search_bm25(
        self, expr: str, k: int, filters: Filters
    ) -> list[BM25Hit]:
        if not expr:
            return []
        limit = _fetch_limit(k)
        if not _has_retrieval_filters(filters):
            results = await self._index.search_bm25(expr, limit, filters)
            return self._bm25_hits(results, limit, filters)

        fetch_limit = max(limit * 10, 200)
        results = await self._index.search_bm25(expr, fetch_limit, filters)
        return self._bm25_hits(results, limit, filters)

    async def search_vector(
        self, embedding: Sequence[float], k: int, filters: Filters
    ) -> list[VectorHit]:
        if self._vectors is None:
            return []
        if not embedding:
            return []
        limit = _fetch_limit(k)
        if not _has_retrieval_filters(filters):
            hits = await self._vectors.search(
                embedding, self._model, limit, filters
            )
            return self._vector_hits(hits, limit, filters)

        fetch_limit = max(limit * 10, 200)
        hits = await self._vectors.search(
            embedding, self._model, fetch_limit, filters
        )
        return self._vector_hits(hits, limit, filters)

    async def chunks(self) -> list[TrigramChunk]:
        rows = await self._index.all_rows()
        return [
            TrigramChunk(
                id=r.path,
                path=r.path,
                title=r.title,
                summary=r.summary,
                content=r.content,
            )
            for r in rows
        ]

    def _bm25_hits(
        self, rows: list[IndexedRow], limit: int, filters: Filters
    ) -> list[BM25Hit]:
        out: list[BM25Hit] = []
        for row in rows:
            if not _row_passes_filters(row, filters):
                continue
            out.append(
                BM25Hit(
                    id=row.path,
                    path=row.path,
                    title=row.title,
                    summary=row.summary,
                    content=row.content or row.snippet,
                    metadata=dict(row.metadata),
                    score=row.score,
                )
            )
            if len(out) >= limit:
                break
        return out

    def _vector_hits(
        self, rows: list[IndexedRow], limit: int, filters: Filters
    ) -> list[VectorHit]:
        out: list[VectorHit] = []
        for row in rows:
            if not _row_passes_filters(row, filters):
                continue
            out.append(
                VectorHit(
                    id=row.path,
                    path=row.path,
                    title=row.title,
                    summary=row.summary,
                    content=row.content,
                    metadata=dict(row.metadata),
                    similarity=float(row.score),
                )
            )
            if len(out) >= limit:
                break
        return out
