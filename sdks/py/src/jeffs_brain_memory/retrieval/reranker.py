# SPDX-License-Identifier: Apache-2.0
"""Reranker protocol and the :class:`LLMReranker`.

Mirrors the Go and TypeScript reranker prompts closely enough for the
daemon retrieval path to use real rerank scores during tri-SDK evals.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import replace
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..llm.provider import Provider
from ..llm.types import CompleteRequest, Message, Role
from .types import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    """Pluggable cross-encoder surface."""

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]: ...

    def name(self) -> str: ...


RERANK_SNIPPET_LIMIT = 1200
HTTP_RERANK_DEFAULT_TIMEOUT_S = 5.0
LLM_RERANK_DEFAULT_MAX_BATCH = 5
LLM_RERANK_MAX_TOKENS = 2048
LLM_RERANK_TEMPERATURE = 0.0
DEFAULT_SHARED_RERANK_CONCURRENCY = 4
HTTP_RERANK_DEFAULT_PROBE_TTL_S = 30.0

_shared_rerank_gates: dict[int, asyncio.Semaphore] = {}

RERANK_SYSTEM_PROMPT = """You are scoring wiki articles against a user's question.
Return ONLY a JSON array of objects matching the input order:
[{"id": 0, "score": 8.5}, {"id": 1, "score": 2.0}, ...]
Score 0 means irrelevant. Score 10 means perfectly answers the
question. Use British English."""

RERANK_SYSTEM_PROMPT_STRICT = """You are scoring wiki articles against a user's question.
Return ONLY a raw JSON array and NOTHING ELSE. No prose, no markdown,
no backticks, no commentary. The array must have one object per input
article in the same order:
[{"id": 0, "score": 8.5}, {"id": 1, "score": 2.0}, ...]
Each score is a number between 0 (irrelevant) and 10 (perfect match).
Use British English."""


def compose_rerank_text(r: RetrievedChunk) -> str:
    """Assemble the rerank payload with summary plus body excerpt."""

    title = r.title.strip()
    summary = r.summary.strip()
    body = " ".join(r.text.split())
    if body:
        snippet = (
            body
            if len(body) <= RERANK_SNIPPET_LIMIT
            else body[:RERANK_SNIPPET_LIMIT] + "..."
        )
    else:
        snippet = "(no body excerpt available)"
    summary_line = summary if summary else "(no summary available)"
    title_line = title if title else "(untitled)"
    return "\n".join(
        [
            f"title: {title_line}",
            f"    path: {r.path}",
            f"    summary: {summary_line}",
            "",
            f"    content: {snippet}",
        ]
    )


def _normalise_rerank_endpoint(endpoint: str) -> str:
    trimmed = endpoint.strip().rstrip("/")
    if trimmed == "":
        raise ValueError("retrieval: HTTPReranker requires a non-empty endpoint")
    if "://" not in trimmed:
        return trimmed
    tail = trimmed.split("://", 1)[1]
    if "/" in tail:
        return trimmed
    return trimmed + "/v1/rerank"


def _probe_base_url(endpoint: str) -> str:
    trimmed = endpoint.strip().rstrip("/")
    if trimmed == "":
        raise ValueError("retrieval: HTTPReranker requires a non-empty endpoint")
    if "://" not in trimmed:
        return trimmed.removesuffix("/rerank")
    parsed = urlsplit(trimmed)
    path = parsed.path.rstrip("/")
    if path.endswith("/rerank"):
        path = path[: -len("/rerank")]
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _join_probe_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _rerank_concurrency_limit() -> int:
    raw = os.getenv("JB_RERANK_CONCURRENCY", "").strip()
    if not raw:
        return DEFAULT_SHARED_RERANK_CONCURRENCY
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_SHARED_RERANK_CONCURRENCY
    return parsed if parsed > 0 else DEFAULT_SHARED_RERANK_CONCURRENCY


async def _run_with_shared_rerank_concurrency(task):
    limit = _rerank_concurrency_limit()
    gate = _shared_rerank_gates.get(limit)
    if gate is None:
        gate = asyncio.Semaphore(limit)
        _shared_rerank_gates[limit] = gate
    async with gate:
        return await task()


async def _reranker_available(reranker: object | None) -> bool:
    if reranker is None:
        return False
    probe = getattr(reranker, "is_available", None)
    if callable(probe):
        return await probe()
    return True


class AutoReranker:
    """Prefer a primary backend and fall back when unavailable or failing."""

    def __init__(
        self,
        *,
        primary: Reranker,
        fallback: Reranker | None = None,
        label: str = "auto-rerank",
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._label = label

    def name(self) -> str:
        return self._label

    async def is_available(self) -> bool:
        if await _reranker_available(self._primary):
            return True
        return await _reranker_available(self._fallback)

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        primary_available = await _reranker_available(self._primary)
        if primary_available:
            try:
                return await self._primary.rerank(query, chunks)
            except Exception:  # noqa: BLE001
                if self._fallback is None:
                    raise
        if self._fallback is not None:
            return await self._fallback.rerank(query, chunks)
        raise RuntimeError("auto-rerank: no available reranker")

    async def close(self) -> None:
        for reranker in (self._primary, self._fallback):
            closer = getattr(reranker, "close", None)
            if callable(closer):
                await closer()


class HTTPReranker:
    """Reranker backed by a cross-encoder HTTP endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str = "",
        model: str = "bge-reranker-v2-m3",
        timeout_s: float = HTTP_RERANK_DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
        probe_ttl_s: float = HTTP_RERANK_DEFAULT_PROBE_TTL_S,
    ) -> None:
        self._probe_base_url = _probe_base_url(endpoint)
        self._endpoint = _normalise_rerank_endpoint(endpoint)
        self._api_key = api_key.strip()
        self._model = model.strip() or "bge-reranker-v2-m3"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._probe_ttl_s = probe_ttl_s if probe_ttl_s > 0 else HTTP_RERANK_DEFAULT_PROBE_TTL_S
        self._availability_memo: tuple[bool, float] | None = None
        self._availability_probe: asyncio.Task[bool] | None = None

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        return await _run_with_shared_rerank_concurrency(
            lambda: self._rerank_impl(query, chunks)
        )

    async def _rerank_impl(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:

        headers = {
            "content-type": "application/json",
            "accept": "application/json",
        }
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

        try:
            response = await self._client.post(
                self._endpoint,
                json={
                    "model": self._model,
                    "query": query,
                    "documents": [compose_rerank_text(chunk) for chunk in chunks],
                    "top_n": len(chunks),
                },
                headers=headers,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError:
            self._set_availability(response.status_code < 500)
            raise
        except Exception:  # noqa: BLE001
            self._set_availability(False)
            raise
        try:
            parsed = response.json()
        except ValueError:
            self._set_availability(False)
            raise
        if not isinstance(parsed, dict):
            self._set_availability(False)
            raise ValueError("retrieval: HTTPReranker returned invalid JSON shape")
        results = parsed.get("results")
        if not isinstance(results, list):
            self._set_availability(False)
            raise ValueError("retrieval: HTTPReranker returned no scored documents")

        scores = [float("-inf")] * len(chunks)
        saw_score = False
        for item in results:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(idx, int) or idx < 0 or idx >= len(scores):
                continue
            if not isinstance(score, int | float):
                continue
            scores[idx] = float(score)
            saw_score = True
        if not saw_score:
            self._set_availability(False)
            raise ValueError("retrieval: HTTPReranker returned no scored documents")
        self._set_availability(True)

        ranked = list(enumerate(chunks))
        ranked.sort(
            key=lambda pair: (
                -scores[pair[0]],
                -pair[1].score,
                pair[0],
            )
        )
        return [
            replace(
                chunk,
                metadata=dict(chunk.metadata),
                rerank_score=(
                    0.0 if scores[idx] == float("-inf") else scores[idx]
                ),
            )
            for idx, chunk in ranked
        ]

    def name(self) -> str:
        return f"http:{self._model}"

    def _set_availability(self, value: bool) -> None:
        self._availability_memo = (value, time.monotonic())

    async def is_available(self) -> bool:
        cached = self._availability_memo
        now = time.monotonic()
        if cached is not None and now - cached[1] < self._probe_ttl_s:
            return cached[0]
        if self._availability_probe is not None:
            return await self._availability_probe

        async def _probe() -> bool:
            try:
                value = await _run_with_shared_rerank_concurrency(
                    self._probe_availability
                )
            finally:
                self._availability_probe = None
            self._set_availability(value)
            return value

        self._availability_probe = asyncio.create_task(_probe())
        return await self._availability_probe

    async def _probe_availability(self) -> bool:
        for path in ("/health", "/info"):
            try:
                response = await self._client.get(
                    _join_probe_url(self._probe_base_url, path),
                    headers={"accept": "application/json"},
                )
            except Exception:  # noqa: BLE001
                continue
            if response.is_success:
                return True
        return False

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class LLMReranker:
    """Reranker backed by an :class:`llm.Provider`."""

    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        max_batch: int = LLM_RERANK_DEFAULT_MAX_BATCH,
    ) -> None:
        if provider is None:
            raise ValueError("retrieval: LLMReranker requires a non-nil provider")
        if not model:
            raise ValueError("retrieval: LLMReranker requires a non-empty model name")
        self._provider = provider
        self._model = model
        self._max_batch = max_batch if max_batch > 0 else LLM_RERANK_DEFAULT_MAX_BATCH

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        scores = [0.0] * len(chunks)
        scored = [False] * len(chunks)
        for start in range(0, len(chunks), self._max_batch):
            batch = chunks[start : start + self._max_batch]
            batch_scores = await self._call_batch(query, batch)
            if batch_scores is None:
                continue
            for local_idx, score in enumerate(batch_scores):
                global_idx = start + local_idx
                if global_idx >= len(scores):
                    break
                scores[global_idx] = score
                scored[global_idx] = True

        if not any(scored):
            return [replace(chunk, metadata=dict(chunk.metadata)) for chunk in chunks]

        ranked = list(enumerate(chunks))
        ranked.sort(
            key=lambda pair: (
                -scores[pair[0]],
                -pair[1].score,
                pair[0],
            )
        )
        return [
            replace(
                chunk,
                metadata=dict(chunk.metadata),
                rerank_score=scores[idx],
            )
            for idx, chunk in ranked
        ]

    def name(self) -> str:
        return f"llm:{self._model}"

    async def is_available(self) -> bool:
        return self._provider is not None

    async def _call_batch(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[float] | None:
        payload = _render_user_prompt(query, chunks)
        base_req = CompleteRequest(
            model=self._model,
            messages=[
                Message(role=Role.SYSTEM, content=RERANK_SYSTEM_PROMPT),
                Message(role=Role.USER, content=payload),
            ],
            temperature=LLM_RERANK_TEMPERATURE,
            max_tokens=LLM_RERANK_MAX_TOKENS,
        )

        try:
            resp = await _run_with_shared_rerank_concurrency(
                lambda: self._provider.complete(base_req)
            )
        except Exception:  # noqa: BLE001
            resp = None
        if resp is not None:
            parsed = _parse_rerank_scores(resp.text, len(chunks))
            if parsed is not None:
                return parsed

        strict_req = CompleteRequest(
            model=self._model,
            messages=[
                Message(role=Role.SYSTEM, content=RERANK_SYSTEM_PROMPT_STRICT),
                Message(role=Role.USER, content=payload),
            ],
            temperature=LLM_RERANK_TEMPERATURE,
            max_tokens=LLM_RERANK_MAX_TOKENS,
        )
        try:
            strict_resp = await _run_with_shared_rerank_concurrency(
                lambda: self._provider.complete(strict_req)
            )
        except Exception:  # noqa: BLE001
            return None
        return _parse_rerank_scores(strict_resp.text, len(chunks))


def _render_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    lines = ["## Question", query.strip(), "", "## Articles"]
    for idx, chunk in enumerate(chunks):
        lines.append(f"[{idx}] {compose_rerank_text(chunk).strip()}")
        lines.append("")
    return "\n".join(lines)


def _parse_rerank_scores(raw: str, expected: int) -> list[float] | None:
    payload = _extract_json_array(raw)
    if payload is None:
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None

    scores = [0.0] * expected
    saw_score = False
    if parsed and all(isinstance(item, dict) for item in parsed):
        for position, item in enumerate(parsed):
            idx = item.get("id") if isinstance(item, dict) else None
            score = item.get("score") if isinstance(item, dict) else None
            if not isinstance(score, int | float):
                continue
            target = idx if isinstance(idx, int) else position
            if not isinstance(target, int) or target < 0 or target >= expected:
                continue
            scores[target] = float(score)
            saw_score = True
        return scores if saw_score else None

    if parsed and all(isinstance(item, int | float) for item in parsed):
        for index, score in enumerate(parsed[:expected]):
            scores[index] = float(score)
        return scores

    return None


def _extract_json_array(raw: str) -> str | None:
    trimmed = raw.strip()
    start = trimmed.find("[")
    end = trimmed.rfind("]")
    if start < 0 or end < start:
        return None
    return trimmed[start : end + 1]
