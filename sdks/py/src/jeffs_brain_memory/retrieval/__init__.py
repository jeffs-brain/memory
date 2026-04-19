# SPDX-License-Identifier: Apache-2.0
"""Hybrid retrieval — BM25 + vector search with RRF fusion.

See ``spec/ALGORITHMS.md`` for the pipeline definition and
``sdks/go/retrieval`` for the reference Go implementation this package
tracks bit-for-bit.
"""

from __future__ import annotations

from .index_source import IndexedRow, IndexSource, SearchIndex, VectorStore
from .intent import (
    ATOMIC_EVENT_NOTE_RE,
    DATE_TAG_RE,
    ENUMERATION_OR_TOTAL_QUERY_RE,
    FACT_LOOKUP_VERB_RE,
    FIRST_PERSON_FACT_LOOKUP_RE,
    GENERIC_NOTE_RE,
    PREFERENCE_NOTE_RE,
    PREFERENCE_QUERY_RE,
    ROLLUP_NOTE_RE,
    RetrievalIntent,
    concrete_fact_intent_multiplier,
    detect_retrieval_intent,
    preference_intent_multiplier,
    retrieval_intent_multiplier,
    retrieval_result_text,
    reweight_shared_memory_ranking,
)
from .reranker import (
    AutoReranker,
    HTTPReranker,
    LLMReranker,
    RERANK_SNIPPET_LIMIT,
    Reranker,
    compose_rerank_text,
)
from .retriever import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_RERANK_TOP_N,
    DEFAULT_TOP_K,
    Retriever,
    UNANIMITY_AGREE_MIN,
    UNANIMITY_WINDOW,
)
from .retry import (
    RETRY_STOP_WORDS,
    TRIGRAM_JACCARD_THRESHOLD,
    TrigramHit,
    TrigramIndex,
    build_trigram_index,
    compute_trigrams,
    force_refresh_index,
    jaccard,
    query_tokens,
    sanitise_query,
    slug_text_for,
    strongest_term,
)
from .rrf import RRF_DEFAULT_K, RRFCandidate, reciprocal_rank_fusion
from .source import BM25Hit, Source, TrigramChunk, VectorHit
from .temporal import (
    augment_query_with_temporal,
    resolved_temporal_hint_line,
    temporal_query_variants,
)
from .types import (
    Attempt,
    Filters,
    Mode,
    Request,
    Response,
    RetrievedChunk,
    Trace,
)

__all__ = [
    "ATOMIC_EVENT_NOTE_RE",
    "Attempt",
    "AutoReranker",
    "BM25Hit",
    "DATE_TAG_RE",
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_RERANK_TOP_N",
    "DEFAULT_TOP_K",
    "ENUMERATION_OR_TOTAL_QUERY_RE",
    "FACT_LOOKUP_VERB_RE",
    "FIRST_PERSON_FACT_LOOKUP_RE",
    "Filters",
    "GENERIC_NOTE_RE",
    "HTTPReranker",
    "IndexedRow",
    "IndexSource",
    "LLMReranker",
    "Mode",
    "PREFERENCE_NOTE_RE",
    "PREFERENCE_QUERY_RE",
    "RERANK_SNIPPET_LIMIT",
    "RETRY_STOP_WORDS",
    "ROLLUP_NOTE_RE",
    "RRF_DEFAULT_K",
    "Request",
    "Reranker",
    "Response",
    "RetrievalIntent",
    "RetrievedChunk",
    "Retriever",
    "RRFCandidate",
    "SearchIndex",
    "Source",
    "TRIGRAM_JACCARD_THRESHOLD",
    "Trace",
    "TrigramChunk",
    "TrigramHit",
    "TrigramIndex",
    "UNANIMITY_AGREE_MIN",
    "UNANIMITY_WINDOW",
    "VectorHit",
    "VectorStore",
    "augment_query_with_temporal",
    "build_trigram_index",
    "compose_rerank_text",
    "compute_trigrams",
    "concrete_fact_intent_multiplier",
    "detect_retrieval_intent",
    "force_refresh_index",
    "jaccard",
    "preference_intent_multiplier",
    "query_tokens",
    "reciprocal_rank_fusion",
    "resolved_temporal_hint_line",
    "retrieval_intent_multiplier",
    "retrieval_result_text",
    "reweight_shared_memory_ranking",
    "sanitise_query",
    "slug_text_for",
    "strongest_term",
    "temporal_query_variants",
]
