# SPDX-License-Identifier: Apache-2.0
"""LLM reranker regression coverage."""

from __future__ import annotations

import httpx

from jeffs_brain_memory.llm.fake import FakeProvider
from jeffs_brain_memory.retrieval import (
    AutoReranker,
    HTTPReranker,
    LLMReranker,
    RetrievedChunk,
    compose_rerank_text,
)


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id="a", path="a.md", score=0.2, title="Alpha", summary="one"),
        RetrievedChunk(chunk_id="b", path="b.md", score=0.1, title="Bravo", summary="two"),
    ]


async def test_llm_reranker_reorders_by_score() -> None:
    reranker = LLMReranker(
        provider=FakeProvider(['[{"id":1,"score":9},{"id":0,"score":1}]']),
        model="judge-m",
    )

    out = await reranker.rerank("q", _chunks())

    assert [chunk.chunk_id for chunk in out] == ["b", "a"]
    assert out[0].rerank_score == 9.0
    assert out[1].rerank_score == 1.0


async def test_llm_reranker_retries_with_strict_prompt() -> None:
    reranker = LLMReranker(
        provider=FakeProvider([
            "not json",
            '[{"id":0,"score":1},{"id":1,"score":9}]',
        ]),
        model="judge-m",
    )

    out = await reranker.rerank("q", _chunks())

    assert [chunk.chunk_id for chunk in out] == ["b", "a"]
    assert out[0].rerank_score == 9.0


async def test_llm_reranker_accepts_positional_object_scores_without_ids() -> None:
    reranker = LLMReranker(
        provider=FakeProvider(['[{"score":1},{"score":9}]']),
        model="judge-m",
    )

    out = await reranker.rerank("q", _chunks())

    assert [chunk.chunk_id for chunk in out] == ["b", "a"]
    assert out[0].rerank_score == 9.0


async def test_llm_reranker_accepts_bare_numeric_score_arrays() -> None:
    reranker = LLMReranker(
        provider=FakeProvider(["[1, 9]"]),
        model="judge-m",
    )

    out = await reranker.rerank("q", _chunks())

    assert [chunk.chunk_id for chunk in out] == ["b", "a"]
    assert out[0].rerank_score == 9.0


async def test_llm_reranker_returns_identity_when_all_batches_fail() -> None:
    reranker = LLMReranker(
        provider=FakeProvider(["still bad", "still bad"]),
        model="judge-m",
    )

    out = await reranker.rerank("q", _chunks())

    assert [chunk.chunk_id for chunk in out] == ["a", "b"]
    assert [chunk.rerank_score for chunk in out] == [0.0, 0.0]


def test_compose_rerank_text_includes_body_excerpt() -> None:
    chunk = RetrievedChunk(
        chunk_id="a",
        path="memory/global/alpha.md",
        score=0.2,
        title="Alpha",
        summary="Summary line",
        text="Body line with 2024-02-01 and $250 raised.",
    )

    rendered = compose_rerank_text(chunk)

    assert "summary: Summary line" in rendered
    assert "summary: Summary line\n\n    content:" in rendered
    assert "content: Body line with 2024-02-01 and $250 raised." in rendered


async def test_http_reranker_reorders_by_cross_encoder_score() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/rerank"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://rerank.test",
    )
    reranker = HTTPReranker(endpoint="http://rerank.test", client=client)

    out = await reranker.rerank("q", _chunks())

    assert [chunk.chunk_id for chunk in out] == ["b", "a"]
    assert out[0].rerank_score == 0.9
    assert out[1].rerank_score == 0.2

    await reranker.close()
    await client.aclose()


async def test_http_reranker_probes_health_once_and_memoises_result() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/health":
            return httpx.Response(200, text="ok")
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://rerank.test",
    )
    reranker = HTTPReranker(
        endpoint="http://rerank.test",
        client=client,
        probe_ttl_s=60.0,
    )

    assert await reranker.is_available() is True
    assert await reranker.is_available() is True
    assert calls == [("GET", "/health")]

    await reranker.close()
    await client.aclose()


async def test_http_reranker_falls_back_to_info_when_health_unavailable() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method != "GET":
            raise AssertionError(f"unexpected request {request.method} {request.url}")
        if request.url.path == "/health":
            return httpx.Response(404, text="missing")
        if request.url.path == "/info":
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://rerank.test",
    )
    reranker = HTTPReranker(
        endpoint="http://rerank.test/rerank",
        client=client,
        probe_ttl_s=0.0,
    )

    assert await reranker.is_available() is True
    assert calls == [("GET", "/health"), ("GET", "/info")]

    await reranker.close()
    await client.aclose()


class _StubReranker:
    def __init__(self, *, available: bool, fail: bool = False) -> None:
        self.available = available
        self.fail = fail
        self.calls = 0

    async def is_available(self) -> bool:
        return self.available

    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return chunks

    def name(self) -> str:
        return "stub"


async def test_auto_reranker_falls_back_when_primary_unavailable() -> None:
    primary = _StubReranker(available=False)
    fallback = _StubReranker(available=True)
    reranker = AutoReranker(primary=primary, fallback=fallback)

    out = await reranker.rerank("q", _chunks())

    assert [chunk.chunk_id for chunk in out] == ["a", "b"]
    assert primary.calls == 0
    assert fallback.calls == 1


async def test_auto_reranker_falls_back_when_primary_errors() -> None:
    primary = _StubReranker(available=True, fail=True)
    fallback = _StubReranker(available=True)
    reranker = AutoReranker(primary=primary, fallback=fallback)

    out = await reranker.rerank("q", _chunks())

    assert [chunk.chunk_id for chunk in out] == ["a", "b"]
    assert primary.calls == 1
    assert fallback.calls == 1
