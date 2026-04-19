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
    }
    expected = aliases.get(want.strip().lower(), {want.strip().lower()})
    return row_scope.strip().lower() in expected if row_scope else True


def _row_passes_filters(row: IndexedRow, filters: Filters) -> bool:
    """Report whether ``row`` satisfies the filter fields."""
    if not filters.path_prefix:
        path_ok = True
    elif len(row.path) < len(filters.path_prefix):
        path_ok = False
    else:
        path_ok = row.path[: len(filters.path_prefix)] == filters.path_prefix
    if not path_ok:
        return False

    if not _scope_matches_filter(row.scope, filters.scope):
        return False
    if filters.project and row.project_slug and row.project_slug != filters.project:
        return False
    if filters.tags:
        row_tags = set(row.tags.split())
        for tag in filters.tags:
            if tag not in row_tags:
                return False
    return True


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
        results = await self._index.search_bm25(expr, k, filters)
        out: list[BM25Hit] = []
        for r in results:
            if not _row_passes_filters(r, filters):
                continue
            out.append(
                BM25Hit(
                    id=r.path,
                    path=r.path,
                    title=r.title,
                    summary=r.summary,
                    content=r.content or r.snippet,
                    metadata=dict(r.metadata),
                    score=r.score,
                )
            )
            if k > 0 and len(out) >= k:
                break
        return out

    async def search_vector(
        self, embedding: Sequence[float], k: int, filters: Filters
    ) -> list[VectorHit]:
        if self._vectors is None:
            return []
        if not embedding:
            return []
        hits = await self._vectors.search(embedding, self._model, k, filters)
        out: list[VectorHit] = []
        for h in hits:
            if not _row_passes_filters(h, filters):
                continue
            out.append(
                VectorHit(
                    id=h.path,
                    path=h.path,
                    title=h.title,
                    summary=h.summary,
                    content=h.content,
                    metadata=dict(h.metadata),
                    similarity=float(h.score),
                )
            )
            if k > 0 and len(out) >= k:
                break
        return out

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
