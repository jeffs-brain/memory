# SPDX-License-Identifier: Apache-2.0
"""Tests for the query distiller's caching, provider plumbing and prompt."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import pytest

from jeffs_brain_memory.llm.types import (
    CompleteRequest,
    CompleteResponse,
    Role,
    StreamChunk,
)
from jeffs_brain_memory.query.distill import DISTILL_SYSTEM_PROMPT, Distiller


class FakeProvider:
    """Minimal ``Provider`` fake that records calls and scripted responses."""

    def __init__(self, responses: list[str] | None = None, *, error: Exception | None = None):
        self._responses = list(responses) if responses else []
        self._error = error
        self.calls = 0
        self.last_request: CompleteRequest | None = None

    async def complete(self, req: CompleteRequest) -> CompleteResponse:
        self.calls += 1
        self.last_request = req
        if self._error is not None:
            raise self._error
        if not self._responses:
            return CompleteResponse(text="")
        return CompleteResponse(text=self._responses.pop(0))

    async def complete_stream(
        self, req: CompleteRequest
    ) -> AsyncIterator[StreamChunk]:  # pragma: no cover - not used
        raise NotImplementedError

    async def close(self) -> None:
        return None


async def test_empty_query_returns_untouched() -> None:
    provider = FakeProvider(["should-not-be-used"])
    distiller = Distiller(provider)
    assert await distiller.distill("") == ""
    assert await distiller.distill("   \t\n  ") == ""
    assert provider.calls == 0


async def test_cache_hit_avoids_provider_call() -> None:
    provider = FakeProvider(["distilled output"])
    distiller = Distiller(provider)

    first = await distiller.distill("Kubernetes deployment failing at sync step")
    second = await distiller.distill("Kubernetes deployment failing at sync step")

    assert first == "distilled output"
    assert second == "distilled output"
    assert provider.calls == 1


async def test_cache_key_is_case_and_whitespace_insensitive() -> None:
    provider = FakeProvider(["distilled"])
    distiller = Distiller(provider)
    await distiller.distill("Kubernetes Deploy")
    await distiller.distill("  kubernetes deploy  ")
    # Second call hits the cache: same (model, lower.strip()) key.
    assert provider.calls == 1


async def test_different_models_cache_independently() -> None:
    provider = FakeProvider(["a", "b"])
    distiller = Distiller(provider)
    await distiller.distill("same query", model="sonnet")
    await distiller.distill("same query", model="haiku")
    assert provider.calls == 2


async def test_lru_evicts_oldest_entry() -> None:
    provider = FakeProvider(["r1", "r2", "r3"])
    distiller = Distiller(provider, cache_size=2)

    await distiller.distill("one")
    await distiller.distill("two")
    await distiller.distill("three")  # should evict "one"

    # Re-requesting "one" should re-hit the provider; "two" and "three" should not.
    await distiller.distill("one")
    assert provider.calls == 4


async def test_lru_move_to_end_on_get() -> None:
    provider = FakeProvider(["r1", "r2", "r3"])
    distiller = Distiller(provider, cache_size=2)

    await distiller.distill("one")
    await distiller.distill("two")
    # Touch "one" so it becomes most-recently-used.
    await distiller.distill("one")
    # Adding "three" should now evict "two", not "one".
    await distiller.distill("three")

    # "two" evicted: re-requesting forces a provider call.
    await distiller.distill("two")
    assert provider.calls == 4


async def test_provider_error_propagates_and_does_not_cache() -> None:
    provider = FakeProvider(error=RuntimeError("connection refused"))
    distiller = Distiller(provider)

    with pytest.raises(RuntimeError, match="connection refused"):
        await distiller.distill("long input that forces a distill")

    # Nothing was cached — a retry should call the provider again.
    assert distiller.cache_size() == 0


async def test_request_message_shape_matches_prompt() -> None:
    provider = FakeProvider(["distilled"])
    distiller = Distiller(provider)
    await distiller.distill("raw user input", model="sonnet")

    req = provider.last_request
    assert req is not None
    assert req.model == "sonnet"
    assert len(req.messages) == 2
    assert req.messages[0].role == Role.SYSTEM
    assert req.messages[0].content == DISTILL_SYSTEM_PROMPT
    assert req.messages[1].role == Role.USER
    assert req.messages[1].content == "raw user input"
    assert req.temperature == 0.1
    assert req.max_tokens == 512


async def test_empty_llm_response_falls_back_to_raw() -> None:
    provider = FakeProvider([""])
    distiller = Distiller(provider)
    got = await distiller.distill("kubernetes deploy")
    assert got == "kubernetes deploy"


async def test_cache_size_is_clamped_to_at_least_one() -> None:
    provider = FakeProvider(["r1", "r2"])
    distiller = Distiller(provider, cache_size=0)
    await distiller.distill("one")
    await distiller.distill("two")  # evicts "one"
    assert distiller.cache_size() == 1


def test_prompt_matches_go_source_verbatim() -> None:
    # Read the Go source and extract the raw string literal; compare
    # byte-for-byte against the Python constant.
    go_src = Path(
        "/home/jaythegeek/code/jeffs-brain/memory/sdks/go/query/prompt.go"
    ).read_text()
    start = go_src.index("`") + 1
    end = go_src.index("`", start)
    assert go_src[start:end] == DISTILL_SYSTEM_PROMPT
