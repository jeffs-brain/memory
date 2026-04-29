# SPDX-License-Identifier: Apache-2.0
"""Dataclasses and enums shared across the retrieval package.

The public shapes mirror the Go SDK in ``go/retrieval/types.go``.
They are deliberately plain ``@dataclass`` values so trace emission can
serialise them with the stdlib ``dataclasses.asdict`` when needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Mode(str, Enum):
    """Selects which retrievers participate in a hybrid search."""

    AUTO = "auto"
    BM25 = "bm25"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid-rerank"


@dataclass(slots=True)
class Filters:
    """Narrows retrieval to a subset of the corpus.

    Empty fields are treated as no filter. ``path_prefix`` is inclusive
    of the exact prefix; ``paths`` is an exact allow-list; ``tags`` are
    matched such that every tag must be present for a hit to survive.
    """

    path_prefix: str = ""
    paths: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    scope: str = ""
    project: str = ""

    def has_any(self) -> bool:
        return bool(
            self.path_prefix or self.paths or self.tags or self.scope or self.project
        )

    def matches_path(self, path: str) -> bool:
        prefix = self.path_prefix.strip()
        if prefix and not path.startswith(prefix):
            return False
        wanted = {candidate.strip() for candidate in self.paths if candidate.strip()}
        if wanted and path not in wanted:
            return False
        return True


@dataclass(slots=True)
class Request:
    """Drives a single retrieval call."""

    query: str = ""
    question_date: str = ""
    top_k: int = 0
    mode: Mode = Mode.AUTO
    brain_id: str = ""
    filters: Filters = field(default_factory=Filters)
    candidate_k: int = 0
    rerank_top_n: int = 0
    skip_retry_ladder: bool = False


@dataclass(slots=True)
class RetrievedChunk:
    """A single ranked hit."""

    chunk_id: str = ""
    document_id: str = ""
    path: str = ""
    score: float = 0.0
    text: str = ""
    title: str = ""
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    bm25_rank: int = 0
    vector_similarity: float = 0.0
    rerank_score: float = 0.0


@dataclass(slots=True)
class Attempt:
    """Records one rung of the retry ladder."""

    rung: int = 0
    mode: Mode = Mode.BM25
    top_k: int = 0
    reason: str = ""
    chunks: int = 0
    query: str = ""


@dataclass(slots=True)
class Trace:
    """Records every decision the pipeline made."""

    requested_mode: Mode = Mode.AUTO
    effective_mode: Mode = Mode.AUTO
    intent: str = ""
    used_retry: bool = False
    rrf_k: int = 0
    candidate_k: int = 0
    rerank_top_n: int = 0
    fell_back_to_bm25: bool = False
    embedder_used: bool = False
    reranked: bool = False
    rerank_provider: str = ""
    rerank_skip_reason: str = ""
    bm25_hits: int = 0
    vector_hits: int = 0
    fused_hits: int = 0
    agreements: int = 0
    unanimity_skipped: bool = False


@dataclass(slots=True)
class Response:
    """Bundles the ranked hits with the trace and attempt log."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    took_ms: int = 0
    trace: Trace = field(default_factory=Trace)
    attempts: list[Attempt] = field(default_factory=list)
