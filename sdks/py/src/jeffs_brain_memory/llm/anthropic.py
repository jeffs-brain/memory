# SPDX-License-Identifier: Apache-2.0
"""Anthropic Messages API provider."""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .types import (
    CompleteRequest,
    CompleteResponse,
    EmptyMessagesError,
    LLMError,
    Role,
    StopReason,
    StreamChunk,
    ToolCall,
)

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_DEFAULT_VERSION = "2023-06-01"


class AnthropicProvider:
    """Provider backed by the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        version: str = _DEFAULT_VERSION,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._default_model = model or ""
        self._version = version or _DEFAULT_VERSION
        self._client = http_client or httpx.AsyncClient()
        self._owns_client = http_client is None

    async def complete(self, req: CompleteRequest) -> CompleteResponse:
        if not req.messages:
            raise EmptyMessagesError()
        model = req.model or self._default_model
        payload = _build_request(req, model, stream=False)
        resp = await self._client.post(
            f"{self._base_url}/v1/messages",
            json=payload,
            headers=self._headers(),
        )
        if resp.status_code >= 400:
            raise _parse_error(resp.status_code, resp.content)
        data = resp.json()
        if data.get("error"):
            raise LLMError(f"llm: anthropic: {data['error'].get('message', '')}")
        usage = data.get("usage") or {}
        out = CompleteResponse(
            stop=_map_stop(data.get("stop_reason", "")),
            tokens_in=int(usage.get("input_tokens") or 0),
            tokens_out=int(usage.get("output_tokens") or 0),
        )
        text_parts: list[str] = []
        for block in data.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text") or "")
            elif btype == "tool_use":
                out.tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=json.dumps(block.get("input") or {}),
                    )
                )
        out.text = "".join(text_parts)
        return out

    async def complete_stream(self, req: CompleteRequest) -> AsyncIterator[StreamChunk]:
        if not req.messages:
            raise EmptyMessagesError()
        model = req.model or self._default_model
        payload = _build_request(req, model, stream=True)
        return _anthropic_stream(self._client, self._base_url, self._headers(), payload)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": self._version,
        }


async def _anthropic_stream(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    payload: dict[str, object],
) -> AsyncIterator[StreamChunk]:
    stream_headers = {**headers, "Accept": "text/event-stream"}
    async with client.stream(
        "POST",
        f"{base_url}/v1/messages",
        json=payload,
        headers=stream_headers,
    ) as resp:
        if resp.status_code >= 400:
            body = await resp.aread()
            raise _parse_error(resp.status_code, body)
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            body = line[len("data: "):]
            if not body:
                continue
            try:
                ev = json.loads(body)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            if etype == "content_block_delta":
                delta = ev.get("delta") or {}
                text = delta.get("text") or ""
                if text:
                    yield StreamChunk(delta_text=text)
            elif etype == "message_delta":
                delta = ev.get("delta") or {}
                stop = delta.get("stop_reason") or ""
                if stop:
                    yield StreamChunk(stop=_map_stop(stop))
            elif etype == "message_stop":
                return


def _build_request(
    req: CompleteRequest,
    model: str,
    *,
    stream: bool,
) -> dict[str, object]:
    out: dict[str, object] = {
        "model": model,
        "stream": stream,
        "max_tokens": req.max_tokens if req.max_tokens else 1024,
    }
    if req.temperature:
        out["temperature"] = req.temperature
    if req.stop:
        out["stop_sequences"] = list(req.stop)
    system_parts: list[str] = []
    messages: list[dict[str, object]] = []
    for m in req.messages:
        if m.role is Role.SYSTEM:
            system_parts.append(m.content)
            continue
        role = "assistant" if m.role is Role.ASSISTANT else "user"
        messages.append(
            {
                "role": role,
                "content": [{"type": "text", "text": m.content}],
            }
        )
    if system_parts:
        out["system"] = "\n\n".join(system_parts)
    out["messages"] = messages
    if req.tools:
        out["tools"] = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.schema,
            }
            for t in req.tools
        ]
    return out


def _map_stop(reason: str) -> StopReason:
    match reason:
        case "end_turn":
            return StopReason.END_TURN
        case "max_tokens":
            return StopReason.MAX_TOKENS
        case "tool_use":
            return StopReason.TOOL_USE
        case "stop_sequence":
            return StopReason.STOP
        case _:
            return StopReason.END_TURN


def _parse_error(status: int, body: bytes) -> LLMError:
    try:
        data = json.loads(body)
        err = (data or {}).get("error")
        if err and err.get("message"):
            return LLMError(f"llm: anthropic {status}: {err['message']}")
    except json.JSONDecodeError:
        pass
    text = body.decode("utf-8", errors="replace").strip() or f"status {status}"
    return LLMError(f"llm: anthropic {status}: {text}")
