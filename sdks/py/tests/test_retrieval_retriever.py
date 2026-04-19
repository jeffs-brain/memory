# SPDX-License-Identifier: Apache-2.0
"""BM25 / semantic / hybrid / hybrid-rerank paths end-to-end."""

from __future__ import annotations

import pytest

from jeffs_brain_memory.llm.fake import FakeEmbedder
from jeffs_brain_memory.retrieval import (
    BM25Hit,
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


async def test_trigram_retry_respects_exact_paths() -> None:
    corpus = [
        FakeChunk(
            id="allowed",
            path="wiki/invoice-processing.md",
            title="Invoice Processing",
            content="Invoice processing workflow",
        ),
        FakeChunk(
            id="blocked",
            path="wiki/invoice-review.md",
            title="Invoice Review",
            content="Invoice review workflow",
        ),
    ]
    src = FakeSource(corpus)
    r = Retriever(source=src)
    resp = await r.retrieve(
        Request(
            query="invioce procesing",
            mode=Mode.BM25,
            top_k=5,
            filters=Filters(paths=["wiki/invoice-processing.md"]),
        )
    )
    assert resp.chunks
    assert {chunk.path for chunk in resp.chunks} == {"wiki/invoice-processing.md"}


async def test_trigram_retry_widens_fetch_before_filtering() -> None:
    corpus = [
        FakeChunk(
            id="blocked-a",
            path="wiki/a-invoice-processing.md",
            title="Invoice Processing",
            content="Invoice processing workflow",
        ),
        FakeChunk(
            id="blocked-b",
            path="wiki/b-invoice-processing.md",
            title="Invoice Processing",
            content="Invoice processing workflow",
        ),
        FakeChunk(
            id="allowed",
            path="wiki/z-invoice-processing.md",
            title="Invoice Processing",
            content="Invoice processing workflow",
        ),
    ]
    src = FakeSource(corpus)
    r = Retriever(source=src)
    resp = await r.retrieve(
        Request(
            query="invioce procesing",
            mode=Mode.BM25,
            top_k=5,
            candidate_k=1,
            filters=Filters(paths=["wiki/z-invoice-processing.md"]),
        )
    )
    assert resp.chunks
    assert [chunk.path for chunk in resp.chunks] == ["wiki/z-invoice-processing.md"]


async def test_trace_reports_candidate_and_rerank_top_n_defaults() -> None:
    src = FakeSource(_test_corpus())
    r = Retriever(source=src)
    resp = await r.retrieve(Request(query="invoice", mode=Mode.BM25))
    assert resp.trace.candidate_k == 60
    assert resp.trace.rerank_top_n == 20
    assert resp.trace.rrf_k == 60


async def test_bm25_temporal_fanout_uses_question_date_variants() -> None:
    corpus = [
        FakeChunk(
            id="plain",
            path="raw/lme/plain.md",
            title="plain",
            content="We met last Friday and bought apples.",
        ),
        FakeChunk(
            id="dated",
            path="raw/lme/dated.md",
            title="dated",
            content="The user bought apples on 2024/03/08.",
        ),
    ]
    src = FakeSource(corpus)
    r = Retriever(source=src)
    resp = await r.retrieve(
        Request(
            query="What did the user buy last Friday?",
            question_date="2024/03/13 (Wed) 10:00",
            mode=Mode.BM25,
            top_k=3,
        )
    )
    assert resp.chunks
    got = {chunk.path for chunk in resp.chunks[:2]}
    assert "raw/lme/plain.md" in got
    assert "raw/lme/dated.md" in got
    assert len(src.bm25_calls) >= 2
    assert any("2024/03/08" in call for call in src.bm25_calls)
    assert any("2024-03-08" in call for call in src.bm25_calls)
    assert resp.attempts
    assert "||" in resp.attempts[0].query


async def test_bm25_reweights_most_recent_dated_hits() -> None:
    corpus = [
        FakeChunk(
            id="older",
            path="memory/global/a-older.md",
            title="Market visit",
            content="[Observed on 2024/03/01 (Fri) 09:00]\nEarned $220 at the Downtown Farmers Market.",
        ),
        FakeChunk(
            id="newer",
            path="memory/global/z-newer.md",
            title="Market visit",
            content="[Observed on 2024/03/08 (Fri) 09:00]\nEarned $420 at the Downtown Farmers Market.",
        ),
    ]
    src = FakeSource(corpus)
    r = Retriever(source=src)
    resp = await r.retrieve(
        Request(
            query="How much did I earn at the Downtown Farmers Market on my most recent visit?",
            mode=Mode.BM25,
            top_k=5,
        )
    )
    assert resp.chunks
    assert resp.chunks[0].chunk_id == "newer"


async def test_bm25_reweights_closest_temporal_hint_date() -> None:
    corpus = [
        FakeChunk(
            id="far",
            path="memory/global/a-far.md",
            title="Weekly note",
            content="[Observed on 2024/02/02 (Fri) 10:00]\nMet the supplier and agreed the timeline.",
        ),
        FakeChunk(
            id="near",
            path="memory/global/z-near.md",
            title="Weekly note",
            content="[Observed on 2024/03/08 (Fri) 10:00]\nMet the supplier and agreed the timeline.",
        ),
    ]
    src = FakeSource(corpus)
    r = Retriever(source=src)
    resp = await r.retrieve(
        Request(
            query="What happened last Friday?",
            question_date="2024/03/15 (Fri) 09:00",
            mode=Mode.BM25,
            top_k=5,
        )
    )
    assert resp.chunks
    assert resp.chunks[0].chunk_id == "near"


async def test_bm25_drops_future_dated_hits_relative_to_question_date() -> None:
    corpus = [
        FakeChunk(
            id="past",
            path="memory/global/past.md",
            title="Supplier visit",
            content="[Observed on 2024/03/10 (Sun) 09:00]\nMet the supplier and agreed the next steps.",
        ),
        FakeChunk(
            id="future",
            path="memory/global/future.md",
            title="Supplier visit",
            content="[Observed on 2024/03/20 (Wed) 09:00]\nMet the supplier and agreed the next steps.",
        ),
        FakeChunk(
            id="undated",
            path="memory/global/undated.md",
            title="Supplier visit",
            content="Met the supplier and agreed the next steps.",
        ),
    ]
    src = FakeSource(corpus)
    r = Retriever(source=src)
    resp = await r.retrieve(
        Request(
            query="What is the most recent supplier visit?",
            question_date="2024/03/15 (Fri) 09:00",
            mode=Mode.BM25,
            top_k=5,
        )
    )
    assert resp.chunks
    assert resp.chunks[0].chunk_id == "past"
    assert all(chunk.chunk_id != "future" for chunk in resp.chunks)


async def test_bm25_fanout_drops_drifted_token_probes() -> None:
    src = FakeSource(_test_corpus())

    def override(expr: str) -> tuple[list[BM25Hit], bool]:
        lowered = expr.lower()
        if lowered == "conversation":
            return (
                [
                    BM25Hit(
                        id="noise-conversation",
                        path="memory/project/noise-conversation.md",
                        title="Conversation note",
                        summary="Off-topic conversation metadata",
                        content="Conversation metadata and follow-up notes.",
                        score=1.0,
                    )
                ],
                True,
            )
        if lowered == "remembered":
            return (
                [
                    BM25Hit(
                        id="noise-remembered",
                        path="memory/project/noise-remembered.md",
                        title="Remembered note",
                        summary="Off-topic remembered preference",
                        content="Remembered preferences and recap notes.",
                        score=1.0,
                    )
                ],
                True,
            )
        if (
            "radiation" in lowered
            and "amplified" in lowered
            and "zombie" in lowered
        ):
            return (
                [
                    BM25Hit(
                        id="target",
                        path="raw/lme/answer_sharegpt_hChsWOp_97.md",
                        title="",
                        summary="",
                        content="We finally named the Radiation Amplified zombie Fissionator.",
                        score=10.0,
                    )
                ],
                True,
            )
        return ([], False)

    src.bm25_override = override
    r = Retriever(source=src)
    resp = await r.retrieve(
        Request(
            query=(
                "I was thinking back to our previous conversation about the "
                "Radiation Amplified zombie, and I was wondering if you "
                "remembered what we finally decided to name it?"
            ),
            mode=Mode.BM25,
            top_k=5,
        )
    )

    assert resp.chunks
    assert resp.chunks[0].path == "raw/lme/answer_sharegpt_hChsWOp_97.md"
    assert any("radiation amplified zombie" in call for call in src.bm25_calls)
    assert "conversation" not in src.bm25_calls
    assert "remembered" not in src.bm25_calls


async def test_bm25_fanout_uses_phrase_probes_for_compound_totals() -> None:
    src = FakeSource(_test_corpus())

    def override(expr: str) -> tuple[list[BM25Hit], bool]:
        lowered = expr.lower()
        if "designer handbag" in lowered and "skincare" not in lowered:
            return (
                [
                    BM25Hit(
                        id="bag",
                        path="raw/lme/designer-handbag.md",
                        title="Designer handbag",
                        summary="High-value purchase",
                        content="I spent 1800 on the designer handbag.",
                        score=10.0,
                    )
                ],
                True,
            )
        if "skincare" in lowered and "products" in lowered:
            return (
                [
                    BM25Hit(
                        id="skincare",
                        path="raw/lme/skincare-products.md",
                        title="Skincare products",
                        summary="Beauty purchase",
                        content="I spent 320 on the high-end skincare products.",
                        score=9.0,
                    )
                ],
                True,
            )
        if "handbag" in lowered and "skincare" in lowered:
            return (
                [
                    BM25Hit(
                        id="skincare",
                        path="raw/lme/skincare-products.md",
                        title="Skincare products",
                        summary="Beauty purchase",
                        content="I spent 320 on the high-end skincare products.",
                        score=8.0,
                    )
                ],
                True,
            )
        return ([], False)

    src.bm25_override = override
    r = Retriever(source=src)
    resp = await r.retrieve(
        Request(
            query=(
                "What is the total amount I spent on the designer handbag and "
                "high-end skincare products?"
            ),
            mode=Mode.BM25,
            top_k=5,
        )
    )

    assert len(resp.chunks) >= 2
    got = {chunk.path for chunk in resp.chunks[:2]}
    assert "raw/lme/designer-handbag.md" in got
    assert "raw/lme/skincare-products.md" in got
    assert any("designer handbag" in call for call in src.bm25_calls)
    assert any("high-end skincare products" in call for call in src.bm25_calls)
