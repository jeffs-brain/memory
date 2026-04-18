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

from dataclasses import dataclass
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
    score: float = 0.0


@runtime_checkable
class SearchIndex(Protocol):
    """Minimal interface the adapter expects from a search index."""

    async def search_bm25(
        self, expr: str, k: int, scope: str, project: str
    ) -> list[IndexedRow]: ...

    async def all_rows(self) -> list[IndexedRow]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Minimal interface for a companion vector store."""

    async def search(
        self, embedding: Sequence[float], model: str, k: int
    ) -> list[IndexedRow]: ...


def path_passes_filters(path: str, filters: Filters) -> bool:
    """Report whether ``path`` satisfies the path-shaped filter fields."""
    if not filters.path_prefix:
        return True
    if len(path) < len(filters.path_prefix):
        return False
    return path[: len(filters.path_prefix)] == filters.path_prefix


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
        results = await self._index.search_bm25(
            expr, k, filters.scope, filters.project
        )
        out: list[BM25Hit] = []
        for r in results:
            if not path_passes_filters(r.path, filters):
                continue
            out.append(
                BM25Hit(
                    id=r.path,
                    path=r.path,
                    title=r.title,
                    summary=r.summary,
                    content=r.snippet or r.content,
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
        hits = await self._vectors.search(embedding, self._model, k)
        out: list[VectorHit] = []
        for h in hits:
            if not path_passes_filters(h.path, filters):
                continue
            out.append(
                VectorHit(
                    id=h.path,
                    path=h.path,
                    title=h.title,
                    summary=h.summary,
                    content=h.content,
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
