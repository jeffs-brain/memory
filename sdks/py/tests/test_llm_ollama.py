# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`OllamaProvider` and :class:`OllamaEmbedder`."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from jeffs_brain_memory.llm import (
    CompleteRequest,
    LLMError,
    Message,
    OllamaEmbedder,
    OllamaProvider,
    Role,
    StopReason,
)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_ollama_complete() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "model": "llama3",
                "message": {"role": "assistant", "content": "howdy"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 4,
                "eval_count": 2,
            },
        )

    client = _client(handler)
    provider = OllamaProvider(base_url="http://ollama.test", http_client=client)
    try:
        resp = await provider.complete(
            CompleteRequest(
                model="llama3",
                messages=[Message(role=Role.USER, content="hi")],
            )
        )
    finally:
        await provider.close()
        await client.aclose()
    assert resp.text == "howdy"
    assert resp.stop is StopReason.END_TURN
    assert resp.tokens_in == 4
    assert resp.tokens_out == 2
    assert captured["path"] == "/api/chat"


async def test_ollama_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            b'{"message":{"role":"assistant","content":"he"},"done":false}\n',
            b'{"message":{"role":"assistant","content":"llo"},"done":false}\n',
            b'{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}\n',
        ]
        return httpx.Response(
            200,
            headers={"Content-Type": "application/x-ndjson"},
            content=b"".join(lines),
        )

    client = _client(handler)
    provider = OllamaProvider(base_url="http://ollama.test", http_client=client)
    try:
        gen = await provider.complete_stream(
            CompleteRequest(
                model="llama3",
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


async def test_ollama_embedder() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
        )

    client = _client(handler)
    embedder = OllamaEmbedder(
        base_url="http://ollama.test",
        model="bge-m3",
        dimensions=2,
        http_client=client,
    )
    try:
        vectors = await embedder.embed(["a", "b"])
    finally:
        await embedder.close()
        await client.aclose()
    assert captured["path"] == "/api/embed"
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert embedder.dimensions() == 2


async def test_ollama_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    client = _client(handler)
    provider = OllamaProvider(base_url="http://ollama.test", http_client=client)
    try:
        with pytest.raises(LLMError) as excinfo:
            await provider.complete(
                CompleteRequest(messages=[Message(role=Role.USER, content="hi")])
            )
    finally:
        await provider.close()
        await client.aclose()
    assert "500" in str(excinfo.value)
