# SPDX-License-Identifier: Apache-2.0
"""Dataclasses and enums for the knowledge surface.

Mirrors the Go SDK shapes (see ``sdks/go/knowledge/schema.go``) with
Pythonic naming conventions. Every Document is persisted via the bound
brain ``Store`` under ``raw/documents/<slug>.md``; compilation reads the
same tree and emits ``Chunk`` records suitable for BM25 indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, BinaryIO

from ..path import BrainPath, DocumentID

__all__ = [
    "SearchMode",
    "Document",
    "Chunk",
    "IngestRequest",
    "IngestResponse",
    "CompileOptions",
    "CompileResult",
    "SearchRequest",
    "SearchResponse",
    "SearchHit",
]


class SearchMode(str, Enum):
    """Selects which retrieval strategy backs :func:`Base.search`.

    ``AUTO`` defers to the bound retriever when available and falls back
    to ``BM25`` otherwise. ``HYBRID_RERANK`` is reserved for a future
    cross-encoder pass; it currently behaves like ``HYBRID``.
    """

    AUTO = "auto"
    BM25 = "bm25"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass(slots=True)
class Document:
    """A persisted ingest in the brain.

    ``path`` is the logical destination under ``raw/documents/``. ``body``
    is the extracted plain-text that the segmenter chunks over. ``chunks``
    is populated opportunistically by callers that compile inline.
    """

    id: DocumentID
    brain_id: str
    path: BrainPath
    title: str = ""
    source: str = ""
    content_type: str = ""
    summary: str = ""
    body: str = ""
    bytes: int = 0
    tags: list[str] = field(default_factory=list)
    ingested: datetime | None = None
    modified: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunks: list["Chunk"] = field(default_factory=list)


@dataclass(slots=True)
class Chunk:
    """A single segment of a :class:`Document`, indexed for retrieval."""

    id: str
    document_id: DocumentID
    path: BrainPath
    ordinal: int = 0
    heading: str = ""
    text: str = ""
    tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None


@dataclass(slots=True)
class IngestRequest:
    """Input to :func:`Base.ingest`.

    Exactly one of ``content`` or ``path`` must be supplied. When
    ``content`` is set, ``content_type`` is required (or will be detected
    from ``path`` if that carries a familiar extension) and ``path`` is
    used as a hint for the source label. When ``path`` points at a local
    file, ``content`` is ignored and the file is read from disk.
    """

    brain_id: str = ""
    path: str = ""
    content_type: str = ""
    content: bytes | BinaryIO | None = None
    title: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IngestResponse:
    """Summary of a single ingest call."""

    document_id: DocumentID
    path: BrainPath
    chunk_count: int = 0
    bytes: int = 0
    took_ms: int = 0


@dataclass(slots=True)
class CompileOptions:
    """Tuning knobs for :func:`Base.compile`.

    ``max_batch`` caps the number of documents walked per call; ``0``
    means unlimited. ``dry_run`` walks the documents without forwarding
    chunks to the search index.
    """

    paths: list[BrainPath] = field(default_factory=list)
    max_batch: int = 0
    dry_run: bool = False


@dataclass(slots=True)
class CompileResult:
    """Summary of a compile run."""

    documents: int = 0
    chunks: int = 0
    skipped: int = 0
    errors: int = 0
    elapsed_ms: int = 0


@dataclass(slots=True)
class SearchRequest:
    """Input to :func:`Base.search`."""

    query: str
    max_results: int = 0
    mode: SearchMode = SearchMode.AUTO
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResponse:
    """Output of :func:`Base.search`."""

    hits: list["SearchHit"] = field(default_factory=list)
    mode: str = ""
    elapsed_ms: int = 0
    fell_back: bool = False
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    """A single retrieval hit returned by :func:`Base.search`."""

    path: BrainPath
    title: str = ""
    summary: str = ""
    snippet: str = ""
    score: float = 0.0
    document_id: DocumentID | None = None
    modified: datetime | None = None
    source: str = ""
