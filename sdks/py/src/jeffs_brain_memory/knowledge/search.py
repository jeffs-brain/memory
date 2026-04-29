# SPDX-License-Identifier: Apache-2.0
"""Hybrid + BM25 search delegate with an in-memory fallback scorer.

Ported from ``go/knowledge/search.go``. Routing mirrors the Go
SDK:

* Hybrid (or Auto with a retriever bound) goes through the bound
  retriever. Errors or empty hit lists fall back to BM25 so the caller
  never sees a silent dead end.
* BM25 (or Auto without a retriever) goes through the bound
  ``search.Index``. When no index is bound the request drops to a
  simple in-memory scorer that walks ``raw/documents/``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..path import BrainPath
from .frontmatter import Frontmatter, parse_frontmatter
from .ingest import RAW_DOCUMENTS_PREFIX
from .types import SearchHit, SearchMode, SearchRequest, SearchResponse

__all__ = [
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_CANDIDATE_K",
    "IndexLike",
    "Retriever",
    "score_document",
    "snippet_for",
    "tokenise_query",
]


# Caps the returned hit count when the caller does not set
# ``SearchRequest.max_results``. Matches the Go constant.
DEFAULT_MAX_RESULTS = 10

# Per-retriever slate size requested when a hybrid retriever is
# available. Matches jeff's ``defaultHybridCandidateK``.
DEFAULT_CANDIDATE_K = 50


@runtime_checkable
class IndexLike(Protocol):
    """Minimal interface the BM25 path needs from a search index."""

    async def search_bm25(self, query: str, *, limit: int) -> list[SearchHit]:
        """Return ranked BM25 hits for ``query``."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Minimal interface for the hybrid retriever.

    Concrete implementations live in the ``retrieval`` package. The
    protocol captures the inputs and outputs the knowledge package
    relies on; anything richer stays in the retriever itself.
    """

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int = 10,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        brain_id: str = "",
        mode: SearchMode = SearchMode.AUTO,
    ) -> tuple[list[SearchHit], dict[str, Any]]:
        """Return ``(hits, trace)`` for ``query``.

        ``trace`` surfaces the retriever's effective mode and any other
        diagnostic it wants the caller to see.
        """
        ...


async def run_search(
    *,
    req: SearchRequest,
    index: IndexLike | None,
    retriever: Retriever | None,
    in_memory_fallback: "InMemoryScorer | None" = None,
    brain_id: str = "",
) -> SearchResponse:
    """Execute a search request honouring the routing rules.

    See the module docstring for the routing matrix. ``in_memory_fallback``
    provides the store-walk fallback; the knowledge base injects one that
    knows how to list documents and read bodies.
    """

    query = (req.query or "").strip()
    if not query:
        return SearchResponse(hits=[], mode="", elapsed_ms=0)

    max_results = req.max_results if req.max_results > 0 else DEFAULT_MAX_RESULTS
    mode = req.mode

    if mode == SearchMode.AUTO:
        mode = SearchMode.HYBRID if retriever is not None else SearchMode.BM25

    if mode in (SearchMode.HYBRID, SearchMode.HYBRID_RERANK, SearchMode.SEMANTIC):
        if retriever is not None:
            try:
                hits, trace = await retriever.retrieve(
                    query=query,
                    top_k=max_results,
                    candidate_k=DEFAULT_CANDIDATE_K,
                    brain_id=brain_id,
                    mode=mode,
                )
            except Exception:  # noqa: BLE001
                hits, trace = [], {"error": "retriever-failed"}
            if hits:
                effective = trace.get("effective_mode", mode.value) if isinstance(trace, dict) else mode.value
                return SearchResponse(
                    hits=hits[:max_results],
                    mode=str(effective),
                    trace=trace if isinstance(trace, dict) else {},
                )
            # Degrade to BM25 below.
            bm25_hits = await _run_bm25(index, query, max_results, in_memory_fallback)
            return SearchResponse(hits=bm25_hits, mode="bm25", fell_back=True)

    # BM25 path.
    bm25_hits = await _run_bm25(index, query, max_results, in_memory_fallback)
    return SearchResponse(hits=bm25_hits, mode="bm25")


async def _run_bm25(
    index: IndexLike | None,
    query: str,
    max_results: int,
    in_memory: "InMemoryScorer | None",
) -> list[SearchHit]:
    """Ask the bound index, or drop to the in-memory scorer."""
    if index is not None:
        hits = await index.search_bm25(query, limit=max_results)
        # Tag the source so callers can distinguish index vs memory hits.
        for hit in hits:
            if not hit.source:
                hit.source = "bm25"
        return hits[:max_results]
    if in_memory is not None:
        return await in_memory.search(query, max_results)
    return []


def tokenise_query(query: str) -> list[str]:
    """Lowercase and return non-empty query tokens."""
    tokens = query.strip().lower().split()
    punct = ".,;:!?\"'()[]{}<>"
    out: list[str] = []
    for token in tokens:
        stripped = token.strip(punct)
        if stripped:
            out.append(stripped)
    return out


def score_document(terms: list[str], fm: Frontmatter, body: str) -> float:
    """Coarse multi-field score used by the in-memory fallback."""
    title_l = (fm.title or fm.name or "").lower()
    summary_l = (fm.summary or fm.description or "").lower()
    tags_l = " ".join(fm.tags).lower()
    body_l = body.lower()

    score = 0.0
    for term in terms:
        score += title_l.count(term) * 3.0
        score += summary_l.count(term) * 2.0
        score += tags_l.count(term) * 2.0
        score += body_l.count(term) * 1.0
    return score


def snippet_for(body: str, terms: list[str]) -> str:
    """Return up to 200 characters of body surrounding the first hit."""
    if not body or not terms:
        return ""
    lower = body.lower()
    for term in terms:
        if not term:
            continue
        idx = lower.find(term)
        if idx < 0:
            continue
        start = max(0, idx - 60)
        end = min(len(body), idx + len(term) + 140)
        return body[start:end].strip()
    return ""


class InMemoryScorer:
    """Store-walking fallback scorer.

    The knowledge base injects an instance wired to its own store. This
    keeps the module pure: no direct store import, no knowledge about
    the concrete ``Store`` implementation.
    """

    def __init__(
        self,
        *,
        list_documents: "ListDocumentsFn",
        read_document: "ReadDocumentFn",
    ) -> None:
        self._list = list_documents
        self._read = read_document

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        terms = tokenise_query(query)
        if not terms:
            return []

        try:
            paths = await self._list()
        except Exception:  # noqa: BLE001
            return []

        hits: list[SearchHit] = []
        for path in paths:
            try:
                data = await self._read(path)
            except Exception:  # noqa: BLE001
                continue
            fm, body = parse_frontmatter(data.decode("utf-8", errors="ignore"))
            score = score_document(terms, fm, body)
            if score <= 0:
                continue
            hits.append(
                SearchHit(
                    path=path,
                    title=fm.title or fm.name or _default_title(path),
                    summary=fm.summary or fm.description,
                    snippet=snippet_for(body, terms),
                    score=score,
                    source="memory",
                )
            )

        hits.sort(key=lambda h: (-h.score, str(h.path)))
        return hits[:max_results]


def _default_title(path: str) -> str:
    """Strip the directory and ``.md`` extension to derive a title."""
    base = path.rsplit("/", 1)[-1]
    if base.endswith(".md"):
        base = base[: -len(".md")]
    return base


# Callable types intentionally kept anonymous to avoid dragging the
# Store protocol into the search module.
from typing import Awaitable, Callable  # noqa: E402 - keep close to usage

ListDocumentsFn = Callable[[], Awaitable[list[BrainPath]]]
ReadDocumentFn = Callable[[BrainPath], Awaitable[bytes]]

__all__ += ["InMemoryScorer", "ListDocumentsFn", "ReadDocumentFn", "run_search"]
