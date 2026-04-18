# SPDX-License-Identifier: Apache-2.0
"""Ollama chat and embedding provider."""

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
)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_EMBED_DIMS = 1024


class OllamaProvider:
    """Provider backed by a local Ollama instance."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._default_model = model or ""
        self._client = http_client or httpx.AsyncClient()
        self._owns_client = http_client is None

    async def complete(self, req: CompleteRequest) -> CompleteResponse:
        if not req.messages:
            raise EmptyMessagesError()
        model = req.model or self._default_model
        payload = _build_request(req, model, stream=False)
        resp = await self._client.post(
            f"{self._base_url}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code >= 400:
            raise _ollama_error(resp.status_code, resp.content)
        data = resp.json()
        if data.get("error"):
            raise LLMError(f"llm: ollama: {data['error']}")
        msg = data.get("message") or {}
        return CompleteResponse(
            text=msg.get("content") or "",
            stop=_map_done(data.get("done_reason", "")),
            tokens_in=int(data.get("prompt_eval_count") or 0),
            tokens_out=int(data.get("eval_count") or 0),
        )

    async def complete_stream(self, req: CompleteRequest) -> AsyncIterator[StreamChunk]:
        if not req.messages:
            raise EmptyMessagesError()
        model = req.model or self._default_model
        payload = _build_request(req, model, stream=True)
        return _ollama_stream(self._client, self._base_url, payload)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


async def _ollama_stream(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, object],
) -> AsyncIterator[StreamChunk]:
    async with client.stream(
        "POST",
        f"{base_url}/api/chat",
        json=payload,
        headers={"Content-Type": "application/json"},
    ) as resp:
        if resp.status_code >= 400:
            body = await resp.aread()
            raise _ollama_error(resp.status_code, body)
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = evt.get("message") or {}
            content = msg.get("content") or ""
            if content:
                yield StreamChunk(delta_text=content)
            if evt.get("done"):
                yield StreamChunk(stop=_map_done(evt.get("done_reason", "")))
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
        "messages": [
            {"role": m.role.value, "content": m.content} for m in req.messages
        ],
    }
    if req.temperature or req.max_tokens or req.stop:
        options: dict[str, object] = {}
        if req.temperature:
            options["temperature"] = req.temperature
        if req.max_tokens:
            options["num_predict"] = req.max_tokens
        if req.stop:
            options["stop"] = list(req.stop)
        out["options"] = options
    return out


def _map_done(reason: str) -> StopReason:
    match reason:
        case "stop":
            return StopReason.END_TURN
        case "length":
            return StopReason.MAX_TOKENS
        case _:
            return StopReason.END_TURN


def _ollama_error(status: int, body: bytes) -> LLMError:
    text = body.decode("utf-8", errors="replace").strip() or f"status {status}"
    return LLMError(f"llm: ollama {status}: {text}")


class OllamaEmbedder:
    """Embedder backed by Ollama's /api/embed endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._model = model or DEFAULT_EMBED_MODEL
        self._dimensions = dimensions or DEFAULT_EMBED_DIMS
        self._client = http_client or httpx.AsyncClient()
        self._owns_client = http_client is None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": texts},
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code >= 400:
            raise _ollama_error(resp.status_code, resp.content)
        data = resp.json()
        if data.get("error"):
            raise LLMError(f"llm: ollama embed: {data['error']}")
        return [
            [float(x) for x in row] for row in (data.get("embeddings") or [])
        ]

    def dimensions(self) -> int:
        return self._dimensions

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
