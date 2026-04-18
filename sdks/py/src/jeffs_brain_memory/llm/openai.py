# SPDX-License-Identifier: Apache-2.0
"""OpenAI chat completions and embeddings provider."""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .types import (
    CompleteRequest,
    CompleteResponse,
    EmptyMessagesError,
    LLMError,
    StopReason,
    StreamChunk,
    ToolCall,
)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"
_DEFAULT_EMBED_DIMS = 1536


class OpenAIProvider:
    """Provider backed by the OpenAI Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._default_model = model or ""
        self._client = http_client or httpx.AsyncClient()
        self._owns_client = http_client is None

    async def complete(self, req: CompleteRequest) -> CompleteResponse:
        if not req.messages:
            raise EmptyMessagesError()
        model = req.model or self._default_model
        payload = _build_request(req, model, stream=False)
        resp = await self._client.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
        )
        if resp.status_code >= 400:
            raise _parse_error(resp.status_code, resp.content)
        data = resp.json()
        if data.get("error"):
            raise LLMError(f"llm: openai: {data['error'].get('message', '')}")
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("llm: openai returned no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        out = CompleteResponse(
            text=message.get("content") or "",
            stop=_map_finish(choice.get("finish_reason", "")),
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
        )
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            out.tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", "") or "",
                )
            )
        return out

    async def complete_stream(self, req: CompleteRequest) -> AsyncIterator[StreamChunk]:
        if not req.messages:
            raise EmptyMessagesError()
        model = req.model or self._default_model
        payload = _build_request(req, model, stream=True)
        return _openai_stream(self._client, self._base_url, self._headers(), payload)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }


async def _openai_stream(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    payload: dict[str, object],
) -> AsyncIterator[StreamChunk]:
    stream_headers = {**headers, "Accept": "text/event-stream"}
    async with client.stream(
        "POST",
        f"{base_url}/chat/completions",
        json=payload,
        headers=stream_headers,
    ) as resp:
        if resp.status_code >= 400:
            body = await resp.aread()
            raise _parse_error(resp.status_code, body)
        tool_buffers: dict[int, dict[str, str]] = {}
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload_s = line[len("data: "):]
            if payload_s == "[DONE]":
                return
            try:
                chunk = json.loads(payload_s)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                return
            for c in chunk.get("choices") or []:
                delta = c.get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    yield StreamChunk(delta_text=content)
                for tc in delta.get("tool_calls") or []:
                    idx = int(tc.get("index", 0))
                    buf = tool_buffers.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        buf["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        buf["name"] = fn["name"]
                    if fn.get("arguments"):
                        buf["args"] += fn["arguments"]
                finish = c.get("finish_reason") or ""
                if finish:
                    for idx in sorted(tool_buffers.keys()):
                        buf = tool_buffers[idx]
                        yield StreamChunk(
                            tool_call=ToolCall(
                                id=buf["id"],
                                name=buf["name"],
                                arguments=buf["args"],
                            )
                        )
                    yield StreamChunk(stop=_map_finish(finish))


def _build_request(
    req: CompleteRequest,
    model: str,
    *,
    stream: bool,
) -> dict[str, object]:
    out: dict[str, object] = {
        "model": model,
        "messages": [{"role": m.role.value, "content": m.content} for m in req.messages],
        "stream": stream,
    }
    if req.temperature:
        out["temperature"] = req.temperature
    if req.max_tokens:
        out["max_tokens"] = req.max_tokens
    if req.stop:
        out["stop"] = list(req.stop)
    if req.tools:
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema,
                },
            }
            for t in req.tools
        ]
    return out


def _map_finish(reason: str) -> StopReason:
    match reason:
        case "stop":
            return StopReason.END_TURN
        case "length":
            return StopReason.MAX_TOKENS
        case "tool_calls" | "function_call":
            return StopReason.TOOL_USE
        case _:
            return StopReason.END_TURN


def _parse_error(status: int, body: bytes) -> LLMError:
    try:
        data = json.loads(body)
        err = (data or {}).get("error")
        if err and err.get("message"):
            return LLMError(f"llm: openai {status}: {err['message']}")
    except json.JSONDecodeError:
        pass
    text = body.decode("utf-8", errors="replace").strip() or f"status {status}"
    return LLMError(f"llm: openai {status}: {text}")


class OpenAIEmbedder:
    """Embedder backed by OpenAI's embeddings endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._model = model or _DEFAULT_EMBED_MODEL
        self._dimensions = dimensions or _DEFAULT_EMBED_DIMS
        self._client = http_client or httpx.AsyncClient()
        self._owns_client = http_client is None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.post(
            f"{self._base_url}/embeddings",
            json={"input": texts, "model": self._model},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        if resp.status_code >= 400:
            raise _parse_error(resp.status_code, resp.content)
        data = resp.json()
        if data.get("error"):
            raise LLMError(f"llm: openai embed: {data['error'].get('message', '')}")
        entries = data.get("data") or []
        out: list[list[float]] = [[] for _ in range(len(entries))]
        for entry in entries:
            idx = int(entry.get("index") or 0)
            if 0 <= idx < len(out):
                out[idx] = [float(x) for x in entry.get("embedding") or []]
        return out

    def dimensions(self) -> int:
        return self._dimensions

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
