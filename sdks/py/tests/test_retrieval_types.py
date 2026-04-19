# SPDX-License-Identifier: Apache-2.0
"""Dataclass shapes and Mode enum mapping."""

from __future__ import annotations

from jeffs_brain_memory.retrieval import (
    Attempt,
    Filters,
    Mode,
    Request,
    Response,
    RetrievedChunk,
    Trace,
)


def test_mode_enum_has_canonical_values() -> None:
    assert Mode.AUTO.value == "auto"
    assert Mode.BM25.value == "bm25"
    assert Mode.SEMANTIC.value == "semantic"
    assert Mode.HYBRID.value == "hybrid"
    assert Mode.HYBRID_RERANK.value == "hybrid-rerank"


def test_filters_has_any_detects_non_zero_fields() -> None:
    assert not Filters().has_any()
    assert Filters(path_prefix="memory/").has_any()
    assert Filters(paths=["wiki/a.md"]).has_any()
    assert Filters(tags=["a"]).has_any()
    assert Filters(scope="global_memory").has_any()
    assert Filters(project="alpha").has_any()


def test_request_defaults_are_sane() -> None:
    req = Request(query="hello")
    assert req.query == "hello"
    assert req.question_date == ""
    assert req.top_k == 0
    assert req.mode == Mode.AUTO
    assert req.filters.path_prefix == ""
    assert req.filters.paths == []
    assert not req.skip_retry_ladder


def test_retrieved_chunk_round_trips() -> None:
    c = RetrievedChunk(
        chunk_id="x",
        document_id="x",
        path="wiki/x.md",
        score=0.42,
        title="X",
        summary="sum",
        text="body",
        metadata={"brain": "default"},
    )
    assert c.chunk_id == "x"
    assert c.metadata == {"brain": "default"}


def test_attempt_and_trace_defaults() -> None:
    a = Attempt(rung=3, reason="refreshed_sanitised", chunks=4)
    assert a.mode == Mode.BM25
    t = Trace()
    assert t.requested_mode == Mode.AUTO
    assert t.effective_mode == Mode.AUTO
    assert not t.used_retry


def test_response_bundle_shape() -> None:
    resp = Response(
        chunks=[RetrievedChunk(chunk_id="a", path="a.md")],
        took_ms=7,
    )
    assert len(resp.chunks) == 1
    assert resp.chunks[0].path == "a.md"
    assert resp.took_ms == 7
    assert resp.attempts == []
