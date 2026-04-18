# SPDX-License-Identifier: Apache-2.0
"""Reranker protocol and the :class:`LLMReranker` stub.

Mirrors ``sdks/go/retrieval/llm_reranker.go``. The stub preserves the
input order so wiring tests can confirm the pass ran; the real prompt
and response parser are tracked as TODOs against ``spec/RERANK.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..llm.provider import Provider
from .types import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    """Pluggable cross-encoder surface.

    Implementations must preserve the input order of any chunk they
    choose not to rescore and must return a list of the same length.
    """

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]: ...

    def name(self) -> str: ...


RERANK_SNIPPET_LIMIT = 280


def compose_rerank_text(r: RetrievedChunk) -> str:
    """Assemble the ``title\\nsummary`` payload used by every reranker in
    the reference implementation.

    Exposed so adapters that want the canonical text shape can reuse it
    without re-deriving the trimming rules.
    """
    title = r.title.strip()
    summary = r.summary.strip()
    if title and summary:
        return f"{title}\n{summary}"
    if title:
        return title
    if summary:
        return summary
    body = " ".join(r.text.split())
    if len(body) <= RERANK_SNIPPET_LIMIT:
        return body
    return body[:RERANK_SNIPPET_LIMIT] + "..."


class LLMReranker:
    """Reranker backed by an :class:`llm.Provider`.

    TODO(rerank): replace the pass-through :meth:`rerank` with the LLM
    cross-encoder prompt described in ``spec/RERANK.md`` once the prompt
    template stabilises. Until then this acts as an identity reranker:
    it does not reorder, does not assign ``rerank_score``, and traces
    still flag ``reranked=True`` so consumers can confirm the wiring is
    intact.
    """

    def __init__(self, provider: Provider, model: str) -> None:
        if provider is None:
            raise ValueError("retrieval: LLMReranker requires a non-nil provider")
        if not model:
            raise ValueError("retrieval: LLMReranker requires a non-empty model name")
        self._provider = provider
        self._model = model

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        # TODO(rerank): build the cross-encoder prompt from
        # (query, compose_rerank_text(chunk)) pairs, call
        # self._provider.complete, parse the response into a permutation,
        # and reassemble chunks in the new order with rerank_score
        # populated.
        out: list[RetrievedChunk] = []
        for c in chunks:
            copy = RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                path=c.path,
                score=c.score,
                text=c.text,
                title=c.title,
                summary=c.summary,
                metadata=dict(c.metadata),
                bm25_rank=c.bm25_rank,
                vector_similarity=c.vector_similarity,
                rerank_score=c.rerank_score,
            )
            # Preserve input order but mark the pass as having run.
            out.append(copy)
        # Flag wiring: downstream trace expects rerank_score to be
        # non-zero to distinguish real rerankers from the stub. We
        # intentionally leave it as-is; the Trace.Reranked flag set by
        # the pipeline itself is enough for wiring confirmation.
        _ = query  # quiet linters; real implementation uses the query.
        return out

    def name(self) -> str:
        return f"llm:{self._model}"
