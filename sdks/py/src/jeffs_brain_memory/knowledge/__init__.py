# SPDX-License-Identifier: Apache-2.0
"""Knowledge pipelines.

Ports the minimal Go SDK knowledge surface to Python: ingest routing,
markdown compile chunker, and hybrid search delegation. The richer
two-phase wiki compile pipeline from ``jeff/apps/jeff/internal/knowledge``
depends on the LLM provider and wiki index and stays out of this port.
"""

from __future__ import annotations

from .base import Base, Options, new
from .frontmatter import Frontmatter, parse_frontmatter
from .ingest import (
    CONTENT_TYPE_HTML,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_MARKDOWN,
    CONTENT_TYPE_PDF,
    CONTENT_TYPE_TEXT,
    CONTENT_TYPE_YAML,
    DefaultFetcher,
    Fetcher,
    MAX_READ_BYTES,
    RAW_DOCUMENTS_PREFIX,
    detect_content_type,
    extract_plain,
    normalise_url,
    raw_document_path,
    slugify,
    strip_html,
)
from .compile import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    estimate_tokens,
    segment_document,
)
from .search import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_MAX_RESULTS,
    IndexLike,
    Retriever,
    InMemoryScorer,
    score_document,
    snippet_for,
    tokenise_query,
)
from .types import (
    Chunk,
    CompileOptions,
    CompileResult,
    Document,
    IngestRequest,
    IngestResponse,
    SearchHit,
    SearchMode,
    SearchRequest,
    SearchResponse,
)

# Aliases matching the Go SDK spelling (``SearchBM25``, ``SearchHybrid``).
# Kept as bare constants instead of an extra enum so user code can use
# either spelling without importing both.
SearchBM25 = SearchMode.BM25
SearchHybrid = SearchMode.HYBRID

__all__ = [
    # Types
    "Base",
    "Chunk",
    "CompileOptions",
    "CompileResult",
    "Document",
    "Frontmatter",
    "IngestRequest",
    "IngestResponse",
    "Options",
    "SearchHit",
    "SearchMode",
    "SearchBM25",
    "SearchHybrid",
    "SearchRequest",
    "SearchResponse",
    # Protocols
    "Fetcher",
    "DefaultFetcher",
    "IndexLike",
    "Retriever",
    "InMemoryScorer",
    # Factories
    "new",
    # Helpers (Go parity)
    "parse_frontmatter",
    "detect_content_type",
    "extract_plain",
    "normalise_url",
    "raw_document_path",
    "slugify",
    "strip_html",
    "score_document",
    "snippet_for",
    "tokenise_query",
    "segment_document",
    "estimate_tokens",
    # Constants
    "CONTENT_TYPE_HTML",
    "CONTENT_TYPE_JSON",
    "CONTENT_TYPE_MARKDOWN",
    "CONTENT_TYPE_PDF",
    "CONTENT_TYPE_TEXT",
    "CONTENT_TYPE_YAML",
    "MAX_READ_BYTES",
    "RAW_DOCUMENTS_PREFIX",
    "CHUNK_MIN_CHARS",
    "CHUNK_MAX_CHARS",
    "MIN_CHUNK_TOKENS",
    "MAX_CHUNK_TOKENS",
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_MAX_RESULTS",
]
