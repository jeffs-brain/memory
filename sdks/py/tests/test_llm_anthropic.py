# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`AnthropicProvider`."""

from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from jeffs_brain_memory.llm import (
    AnthropicProvider,
    CompleteRequest,
    EmptyMessagesError,
    LLMError,
    Message,
    Role,
    StopReason,
)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_anthropic_complete() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["key"] = request.headers.get("x-api-key")
        captured["version"] = request.headers.get("anthropic-version")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )

    client = _client(handler)
    provider = AnthropicProvider(api_key="k", base_url="https://api.test", http_client=client)
    try:
        resp = await provider.complete(
            CompleteRequest(
                model="claude-test",
                messages=[
                    Message(role=Role.SYSTEM, content="be brief"),
                    Message(role=Role.USER, content="hello"),
                ],
                max_tokens=32,
            )
        )
    finally:
        await provider.close()
        await client.aclose()
    assert resp.text == "hi"
    assert resp.stop is StopReason.END_TURN
    assert resp.tokens_in == 3
    assert resp.tokens_out == 1
    assert captured["path"] == "/v1/messages"
    assert captured["key"] == "k"
    assert captured["version"] == "2023-06-01"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["system"] == "be brief"


async def test_anthropic_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        events = [
            b'data: {"type":"message_start"}\n\n',
            b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
            b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"he"}}\n\n',
            b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"llo"}}\n\n',
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n',
            b'data: {"type":"message_stop"}\n\n',
        ]
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=b"".join(events),
        )

    client = _client(handler)
    provider = AnthropicProvider(api_key="k", base_url="https://api.test", http_client=client)
    try:
        gen = await provider.complete_stream(
            CompleteRequest(
                model="claude-test",
                messages=[Message(role=Role.USER, content="say hi")],
                max_tokens=32,
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


async def test_anthropic_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}},
        )

    client = _client(handler)
    provider = AnthropicProvider(api_key="k", base_url="https://api.test", http_client=client)
    try:
        with pytest.raises(LLMError) as excinfo:
            await provider.complete(
                CompleteRequest(
                    messages=[Message(role=Role.USER, content="hi")],
                    max_tokens=1,
                )
            )
    finally:
        await provider.close()
        await client.aclose()
    assert "bad" in str(excinfo.value)


async def test_anthropic_rejects_empty_messages() -> None:
    provider = AnthropicProvider(api_key="k", base_url="http://unused")
    try:
        with pytest.raises(EmptyMessagesError):
            await provider.complete(CompleteRequest())
    finally:
        await provider.close()
