# SPDX-License-Identifier: Apache-2.0
"""Search stage — SQLite BM25 (FTS5), sqlite-vec, trigram fallback.

See ``spec/ALGORITHMS.md`` for ranking details and
``spec/QUERY-DSL.md`` for the parser contract. The public surface
mirrors the Go SDK module at ``sdks/go/search/``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..path import ChunkID
from .aliases import AliasTable
from .index import (
    BM25Hit,
    Chunk,
    EMBED_BATCH_SIZE,
    EMBED_TEXT_MAX,
    Index,
    SearchOpts,
    TrigramHit,
    VectorHit,
)
from .query_parser import (
    FTSExpr,
    QueryAST,
    Token,
    TokenKind,
    compile,
    compile_fts,
    expand_aliases,
    parse,
    sanitise_query,
    strongest_term,
)
from .stopwords import is_stopword, load_stopwords
from .trigram import TRIGRAM_JACCARD_THRESHOLD, TrigramIndex, jaccard, slug_text, trigrams

__all__ = [
    "AliasTable",
    "BM25Hit",
    "Chunk",
    "EMBED_BATCH_SIZE",
    "EMBED_TEXT_MAX",
    "FTSExpr",
    "Index",
    "QueryAST",
    "SearchHit",
    "SearchOpts",
    "Token",
    "TokenKind",
    "TRIGRAM_JACCARD_THRESHOLD",
    "TrigramHit",
    "TrigramIndex",
    "VectorHit",
    "compile",
    "compile_fts",
    "expand_aliases",
    "is_stopword",
    "jaccard",
    "load_stopwords",
    "parse",
    "sanitise_query",
    "slug_text",
    "strongest_term",
    "trigrams",
]


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Back-compat shape used by the rerank / retrieval layers."""

    chunk_id: ChunkID
    score: float
    snippet: str | None = None
