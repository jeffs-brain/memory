# SPDX-License-Identifier: Apache-2.0
"""Source protocol and the leaf hit shapes the retrieval pipeline
consumes.

Mirrors ``sdks/go/retrieval/source.go``. Implementations are expected
to be safe for concurrent use and must not mutate returned values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Protocol, Sequence, runtime_checkable

from .types import Filters


@dataclass(slots=True)
class BM25Hit:
    """One candidate emitted by the BM25 leg."""

    id: str = ""
    path: str = ""
    title: str = ""
    summary: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass(slots=True)
class VectorHit:
    """One candidate emitted by the vector leg."""

    id: str = ""
    path: str = ""
    title: str = ""
    summary: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity: float = 0.0


@dataclass(slots=True)
class TrigramChunk:
    """Minimum payload the trigram fallback needs to build its index."""

    id: str = ""
    path: str = ""
    title: str = ""
    summary: str = ""
    content: str = ""


@runtime_checkable
class Source(Protocol):
    """Retrieval layer's view of an index.

    Production callers plug a ``search.Index`` adapter in; tests use an
    in-memory fake to drive deterministic retrieval behaviour without
    standing up FTS5.
    """

    async def search_bm25(
        self, expr: str, k: int, filters: Filters
    ) -> list[BM25Hit]: ...

    async def search_vector(
        self, embedding: Sequence[float], k: int, filters: Filters
    ) -> list[VectorHit]: ...

    async def chunks(self) -> list[TrigramChunk]: ...
