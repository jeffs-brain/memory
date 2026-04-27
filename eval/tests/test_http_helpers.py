# SPDX-License-Identifier: Apache-2.0
"""HTTP helper unit tests."""
from __future__ import annotations

import asyncio
import json

import httpx

from http_helpers import (
    ask_one,
    ask_path,
    build_request_spec,
    extract_delta,
    parse_sse_frame,
    search_path,
)


class TestHttpHelpers:
    def test_ask_path_encodes_brain(self) -> None:
        assert ask_path("eval") == "/v1/brains/eval/ask"
        assert ask_path("team alpha") == "/v1/brains/team%20alpha/ask"

    def test_search_path_encodes_brain(self) -> None:
        assert search_path("eval") == "/v1/brains/eval/search"
        assert search_path("team alpha") == "/v1/brains/team%20alpha/search"

    def test_build_request_spec_for_ask_basic(self) -> None:
        spec = build_request_spec(
            brain="eval",
            item={},
            question="where?",
            top_k=3,
            mode="hybrid",
            scenario="ask-basic",
        )
        assert spec.path == "/v1/brains/eval/ask"
        assert spec.streaming is True
        assert spec.body == {"question": "where?", "topK": 3, "mode": "hybrid"}

    def test_build_request_spec_for_ask_augmented_uses_question_date(self) -> None:
        spec = build_request_spec(
            brain="eval",
            item={"question_date": "2024/05/26 (Sun) 09:00"},
            question="where?",
            top_k=3,
            mode="hybrid",
            scenario="ask-augmented",
        )
        assert spec.path == "/v1/brains/eval/ask"
        assert spec.streaming is True
        assert spec.body == {
            "question": "where?",
            "topK": 3,
            "mode": "hybrid",
            "readerMode": "augmented",
            "questionDate": "2024/05/26 (Sun) 09:00",
        }

    def test_build_request_spec_for_search_retrieve_only(self) -> None:
        spec = build_request_spec(
            brain="eval",
            item={"questionDate": "2024-05-26T09:00:00Z"},
            question="where?",
            top_k=4,
            mode="bm25",
            scenario="search-retrieve-only",
            candidate_k=80,
            rerank_top_n=40,
        )
        assert spec.path == "/v1/brains/eval/search"
        assert spec.streaming is False
        assert spec.body == {
            "query": "where?",
            "topK": 4,
            "mode": "bm25",
            "questionDate": "2024-05-26T09:00:00Z",
            "candidateK": 80,
            "rerankTopN": 40,
        }

    def test_build_request_spec_omits_optional_search_knobs_when_zero(self) -> None:
        spec = build_request_spec(
            brain="eval",
            item={},
            question="where?",
            top_k=4,
            mode="auto",
            scenario="search-retrieve-only",
        )
        assert spec.body == {"query": "where?", "topK": 4, "mode": "auto"}

    def test_parse_sse_frame_reads_event_and_data(self) -> None:
        frame = 'event: answer_delta\ndata: {"delta": "Hi "}'
        assert parse_sse_frame(frame) == ("answer_delta", '{"delta": "Hi "}')

    def test_parse_sse_frame_skips_comments_and_empty(self) -> None:
        assert parse_sse_frame(": keepalive\n\n") is None
        assert parse_sse_frame("") is None

    def test_parse_sse_frame_defaults_event_to_message(self) -> None:
        assert parse_sse_frame('data: {"ok": true}') == ("message", '{"ok": true}')

    def test_extract_delta_prefers_delta_then_falls_back(self) -> None:
        assert extract_delta({"delta": "abc"}) == "abc"
        assert extract_delta({"token": "xyz"}) == "xyz"
        assert extract_delta({"text": "t"}) == "t"
        assert extract_delta({}) == ""

    def test_ask_one_posts_augmented_ask_shape(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                text=(
                    'event: retrieve\ndata: {"chunks":[]}\n\n'
                    'event: answer_delta\ndata: {"text":"Hello"}\n\n'
                    'event: done\ndata: {"answer":"Hello"}\n\n'
                ),
                headers={"content-type": "text/event-stream"},
            )

        async def run_once() -> object:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(base_url="http://test", transport=transport) as client:
                spec = build_request_spec(
                    brain="eval",
                    item={"question_date": "2024/05/26 (Sun) 09:00"},
                    question="where?",
                    top_k=3,
                    mode="hybrid",
                    scenario="ask-augmented",
                )
                return await ask_one(client, spec=spec)

        outcome = asyncio.run(run_once())
        assert captured["path"] == "/v1/brains/eval/ask"
        assert captured["body"] == {
            "question": "where?",
            "topK": 3,
            "mode": "hybrid",
            "readerMode": "augmented",
            "questionDate": "2024/05/26 (Sun) 09:00",
        }
        assert outcome.answer == "Hello"

    def test_ask_one_posts_search_retrieve_only_shape(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "chunks": [
                        {
                            "chunkId": "c1",
                            "path": "wiki/a.md",
                            "title": "A",
                            "score": 0.9,
                            "text": "alpha",
                        },
                        {
                            "chunkId": "c2",
                            "path": "wiki/b.md",
                            "title": "B",
                            "score": 0.7,
                            "summary": "beta",
                        },
                    ]
                },
            )

        async def run_once() -> object:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(base_url="http://test", transport=transport) as client:
                spec = build_request_spec(
                    brain="eval",
                    item={"question_date": "2024/05/26 (Sun) 09:00"},
                    question="where?",
                    top_k=4,
                    mode="semantic",
                    scenario="search-retrieve-only",
                    candidate_k=80,
                    rerank_top_n=40,
                )
                return await ask_one(client, spec=spec)

        outcome = asyncio.run(run_once())
        assert captured["path"] == "/v1/brains/eval/search"
        assert captured["body"] == {
            "query": "where?",
            "topK": 4,
            "mode": "semantic",
            "questionDate": "2024/05/26 (Sun) 09:00",
            "candidateK": 80,
            "rerankTopN": 40,
        }
        assert outcome.answer == "alpha\n\nbeta"
        assert outcome.citations == [
            {"chunkId": "c1", "path": "wiki/a.md", "title": "A", "score": 0.9},
            {"chunkId": "c2", "path": "wiki/b.md", "title": "B", "score": 0.7},
        ]
