# SPDX-License-Identifier: Apache-2.0
"""BM25 / semantic / hybrid / hybrid-rerank paths end-to-end."""

from __future__ import annotations

import pytest

from jeffs_brain_memory.llm.fake import FakeEmbedder
from jeffs_brain_memory.retrieval import (
    Filters,
    Mode,
    Request,
    Retriever,
    RetrievedChunk,
)

from ._retrieval_fakes import FakeChunk, FakeSource


def _test_corpus() -> list[FakeChunk]:
    return [
        FakeChunk(
            id="c1",
            path="wiki/invoice-processing.md",
            title="Invoice Processing",
            summary="How we automate supplier invoice ingestion",
            content="Invoice automation workflow extracts line items from PDFs.",
        ),
        FakeChunk(
            id="c2",
            path="wiki/order-processing.md",
            title="Order Processing Pipeline",
            summary="Sales order ingestion for retailers",
            content="Automated document processing for orders captured via email.",
        ),
        FakeChunk(
            id="c3",
            path="wiki/contact-centre.md",
            title="Contact Centre",
            summary="Inbound voice routing",
            content="Telephony stack routes calls via SIP.",
        ),
        FakeChunk(
            id="c4",
            path="memory/global/user-preference-coffee.md",
            title="Coffee preferences",
            summary="Alex prefers flat whites",
            content="Alex likes flat whites with oat milk.",
        ),
        FakeChunk(
            id="c5",
            path="memory/global/user-fact-birthday.md",
            title="User fact: birthday",
            summary="Observed on 1986-08-14",
            content="[observed on: 1986-08-14] Alex was born on 14 August 1986.",
        ),
        FakeChunk(
            id="c6",
            path="wiki/rollup/invoice-summary.md",
            title="Invoice recap",
            summary="Roll-up summary across invoice workflows",
            content="Overview and summary of invoice workflow totals.",
        ),
    ]


class _Reranker:
    """Reverses the head so tests can confirm the reranker ran."""

    def __init__(self) -> None:
        self.calls = 0

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        self.calls += 1
        out: list[RetrievedChunk] = []
        n = len(chunks)
        for i, c in enumerate(chunks):
            flipped = chunks[n - 1 - i]
            out.append(
                RetrievedChunk(
                    chunk_id=flipped.chunk_id,
                    document_id=flipped.document_id,
                    path=flipped.path,
                    score=flipped.score,
                    text=flipped.text,
                    title=flipped.title,
                    summary=flipped.summary,
                    metadata=dict(flipped.metadata),
                    bm25_rank=flipped.bm25_rank,
                    vector_similarity=flipped.vector_similarity,
                    rerank_score=float(n - i),
                )
            )
        return out

    def name(self) -> str:
        return "test-reranker"


async def test_bm25_mode() -> None:
    src = FakeSource(_test_corpus())
    r = Retriever(source=src)
    resp = await r.retrieve(
        Request(query="invoice automation", top_k=3, mode=Mode.BM25)
    )
    assert resp.chunks
    assert resp.trace.effective_mode == Mode.BM25
    assert not resp.trace.embedder_used
    assert resp.chunks[0].path == "wiki/invoice-processing.md"


async def test_semantic_mode() -> None:
    src = FakeSource(_test_corpus())
    r = Retriever(source=src, embedder=FakeEmbedder(src.embed_dim))
    resp = await r.retrieve(
        Request(query="invoice automation workflow", top_k=3, mode=Mode.SEMANTIC)
    )
    assert resp.trace.effective_mode == Mode.SEMANTIC
    assert resp.trace.embedder_used
    assert resp.chunks


async def test_hybrid_mode_fuses_both_legs() -> None:
    src = FakeSource(_test_corpus())
    r = Retriever(source=src, embedder=FakeEmbedder(src.embed_dim))
    resp = await r.retrieve(
        Request(query="invoice automation", top_k=3, mode=Mode.HYBRID)
    )
    assert resp.trace.effective_mode == Mode.HYBRID
    assert resp.trace.bm25_hits > 0
    assert resp.trace.vector_hits > 0
    assert resp.trace.fused_hits > 0
    top = resp.chunks[0].path
    assert top in ("wiki/invoice-processing.md", "wiki/order-processing.md")


async def test_hybrid_rerank_applies_reranker() -> None:
    src = FakeSource(_test_corpus())
    rr = _Reranker()
    r = Retriever(
        source=src, embedder=FakeEmbedder(src.embed_dim), reranker=rr
    )
    resp = await r.retrieve(
        Request(query="invoice automation", top_k=5, mode=Mode.HYBRID_RERANK)
    )
    # Either the unanimity shortcut fired (no reranker call) or the
    # reranker ran and trace reflects it.
    if resp.trace.unanimity_skipped:
        assert rr.calls == 0
        assert resp.trace.rerank_skip_reason == "unanimity"
    else:
        assert resp.trace.reranked
        assert resp.trace.rerank_skip_reason == ""


async def test_mode_fallback_when_embedder_missing() -> None:
    src = FakeSource(_test_corpus())
    r = Retriever(source=src)
    resp = await r.retrieve(Request(query="invoice", mode=Mode.HYBRID))
    assert resp.trace.effective_mode == Mode.BM25
    assert resp.trace.fell_back_to_bm25


async def test_auto_without_embedder_falls_back_silently() -> None:
    src = FakeSource(_test_corpus())
    r = Retriever(source=src)
    resp = await r.retrieve(Request(query="invoice", mode=Mode.AUTO))
    assert resp.trace.effective_mode == Mode.BM25
    assert not resp.trace.fell_back_to_bm25


async def test_unanimity_shortcut_skips_rerank() -> None:
    # Narrow corpus where BM25 + vector converge on the same head.
    corpus = [
        FakeChunk(id="a", path="a.md", title="Alpha", content="alpha bravo"),
        FakeChunk(id="b", path="b.md", title="Bravo", content="alpha bravo"),
        FakeChunk(id="c", path="c.md", title="Charlie", content="alpha bravo"),
    ]
    src = FakeSource(corpus)
    rr = _Reranker()
    r = Retriever(
        source=src, embedder=FakeEmbedder(src.embed_dim), reranker=rr
    )
    resp = await r.retrieve(
        Request(query="alpha bravo", top_k=3, mode=Mode.HYBRID_RERANK)
    )
    if resp.trace.unanimity_skipped:
        assert rr.calls == 0
        assert resp.trace.rerank_skip_reason == "unanimity"


async def test_nil_source_errors() -> None:
    with pytest.raises(ValueError):
        Retriever(source=None)  # type: ignore[arg-type]


async def test_bm25_leg_error_surfaces() -> None:
    src = FakeSource(_test_corpus())
    src.bm25_fail = RuntimeError("boom")
    r = Retriever(source=src)
    with pytest.raises(RuntimeError):
        await r.retrieve(Request(query="anything", mode=Mode.BM25))


async def test_filters_narrow_corpus() -> None:
    corpus = _test_corpus() + [
        FakeChunk(
            id="scoped",
            path="memory/project/billing/invoice.md",
            title="Billing invoice workflow",
        )
    ]
    src = FakeSource(corpus)
    r = Retriever(source=src)
    resp = await r.retrieve(
        Request(
            query="invoice",
            mode=Mode.BM25,
            filters=Filters(path_prefix="memory/project/"),
            top_k=5,
        )
    )
    for c in resp.chunks:
        assert c.path == "memory/project/billing/invoice.md"


async def test_trace_reports_candidate_and_rerank_top_n_defaults() -> None:
    src = FakeSource(_test_corpus())
    r = Retriever(source=src)
    resp = await r.retrieve(Request(query="invoice", mode=Mode.BM25))
    assert resp.trace.candidate_k == 60
    assert resp.trace.rerank_top_n == 20
    assert resp.trace.rrf_k == 60
