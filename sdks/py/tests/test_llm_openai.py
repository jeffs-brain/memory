# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`OpenAIProvider` via ``httpx.MockTransport``."""

from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from jeffs_brain_memory.llm import (
    CompleteRequest,
    LLMError,
    Message,
    OpenAIEmbedder,
    OpenAIProvider,
    Role,
    StopReason,
    ToolDef,
)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


async def test_openai_complete_non_streaming() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi there"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    client = _client(handler)
    provider = OpenAIProvider(
        api_key="test-key",
        base_url="https://api.test/v1",
        http_client=client,
    )
    try:
        resp = await provider.complete(
            CompleteRequest(
                model="gpt-test",
                messages=[Message(role=Role.USER, content="hello")],
            )
        )
    finally:
        await provider.close()
        await client.aclose()
    assert resp.text == "hi there"
    assert resp.stop is StopReason.END_TURN
    assert resp.tokens_in == 5
    assert resp.tokens_out == 2
    assert captured["path"] == "/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["model"] == "gpt-test"


async def test_openai_default_model_used_when_request_blank() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            },
        )

    client = _client(handler)
    provider = OpenAIProvider(
        api_key="k",
        base_url="https://api.test/v1",
        model="fallback-model",
        http_client=client,
    )
    try:
        await provider.complete(
            CompleteRequest(messages=[Message(role=Role.USER, content="hi")])
        )
    finally:
        await provider.close()
        await client.aclose()
    assert captured["body"]["model"] == "fallback-model"


async def test_openai_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [
            b'data: {"choices":[{"index":0,"delta":{"content":"hel"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=b"".join(chunks),
        )

    client = _client(handler)
    provider = OpenAIProvider(api_key="k", base_url="https://api.test/v1", http_client=client)
    try:
        gen = await provider.complete_stream(
            CompleteRequest(
                model="m",
                messages=[Message(role=Role.USER, content="hi")],
            )
        )
        text = ""
        stop: StopReason | None = None
        async for chunk in gen:
            text += chunk.delta_text
            if chunk.stop is not None:
                stop = chunk.stop
    finally:
        await provider.close()
        await client.aclose()
    assert text == "hello"
    assert stop is StopReason.END_TURN


async def test_openai_tool_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"q":"x"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = _client(handler)
    provider = OpenAIProvider(api_key="k", base_url="https://api.test/v1", http_client=client)
    try:
        resp = await provider.complete(
            CompleteRequest(
                model="m",
                messages=[Message(role=Role.USER, content="call it")],
                tools=[
                    ToolDef(
                        name="lookup",
                        description="test",
                        schema={"type": "object"},
                    )
                ],
            )
        )
    finally:
        await provider.close()
        await client.aclose()
    assert resp.stop is StopReason.TOOL_USE
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "lookup"
    assert json.loads(tc.arguments) == {"q": "x"}


async def test_openai_error_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "bad key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )

    client = _client(handler)
    provider = OpenAIProvider(api_key="k", base_url="https://api.test/v1", http_client=client)
    try:
        with pytest.raises(LLMError) as excinfo:
            await provider.complete(
                CompleteRequest(messages=[Message(role=Role.USER, content="hi")])
            )
    finally:
        await provider.close()
        await client.aclose()
    assert "bad key" in str(excinfo.value)
    assert "401" in str(excinfo.value)


async def test_openai_embedder() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                ]
            },
        )

    client = _client(handler)
    embedder = OpenAIEmbedder(
        api_key="k",
        base_url="https://api.test/v1",
        dimensions=3,
        http_client=client,
    )
    try:
        vectors = await embedder.embed(["a", "b"])
    finally:
        await embedder.close()
        await client.aclose()
    assert captured["path"] == "/v1/embeddings"
    assert len(vectors) == 2
    assert vectors[1][2] == pytest.approx(0.6)
    assert embedder.dimensions() == 3


async def test_openai_base_url_override() -> None:
    """OPENAI_BASE_URL routing lets us target a self-hosted model server."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["host"] = request.url.host
        captured["path"] = request.url.path
        body = json.loads(request.content.decode("utf-8"))
        captured["model"] = body["model"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "local"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            },
        )

    client = _client(handler)
    provider = OpenAIProvider(
        api_key="k",
        base_url="https://gemma.lan:8080/v1",
        model="gemma4-31b-it",
        http_client=client,
    )
    try:
        resp = await provider.complete(
            CompleteRequest(messages=[Message(role=Role.USER, content="hi")])
        )
    finally:
        await provider.close()
        await client.aclose()
    assert resp.text == "local"
    assert captured["host"] == "gemma.lan"
    assert captured["path"] == "/v1/chat/completions"
    assert captured["model"] == "gemma4-31b-it"
