# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the parser orchestrator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import pytest

from jeffs_brain_memory.llm.types import (
    CompleteRequest,
    CompleteResponse,
    StreamChunk,
)
from jeffs_brain_memory.query.parser import parse
from jeffs_brain_memory.query.types import Options

ANCHOR = datetime(2023, 4, 10, 23, 7, tzinfo=timezone.utc)


class CountingProvider:
    def __init__(self, response: str = "distilled", *, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls = 0

    async def complete(self, req: CompleteRequest) -> CompleteResponse:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return CompleteResponse(text=self._response)

    async def complete_stream(
        self, req: CompleteRequest
    ) -> AsyncIterator[StreamChunk]:  # pragma: no cover - unused
        raise NotImplementedError

    async def close(self) -> None:
        return None


async def test_default_parse_normalises_and_tokenises() -> None:
    result = await parse("  Kubernetes   Deployment  ")
    assert result.query.raw == "  Kubernetes   Deployment  "
    assert result.query.normalised == "kubernetes deployment"
    assert result.query.tokens == ["kubernetes", "deployment"]
    assert result.query.significant_terms == ["kubernetes", "deployment"]
    assert result.query.temporal is None
    assert result.query.distilled is None
    assert result.trace.used_cache is False
    assert result.trace.distilled is False


async def test_parse_drops_stopwords_from_significant_terms() -> None:
    result = await parse("What is the deployment status?")
    assert "the" not in result.query.significant_terms
    assert "is" not in result.query.significant_terms
    assert "deployment" in result.query.significant_terms


async def test_parse_applies_temporal_when_anchor_present() -> None:
    opts = Options(anchor=ANCHOR)
    result = await parse("What did we discuss 2 weeks ago?", opts)
    assert result.query.temporal is not None
    assert result.query.temporal.recogniser == "relative"
    assert result.query.temporal.range_end == ANCHOR
    assert result.query.temporal.range_start == ANCHOR - timedelta(weeks=2)


async def test_parse_skips_temporal_without_anchor() -> None:
    result = await parse("2 weeks ago something happened")
    assert result.query.temporal is None


async def test_parse_last_weekday_when_anchor_present() -> None:
    opts = Options(anchor=ANCHOR)
    result = await parse("What was said last Saturday?", opts)
    assert result.query.temporal is not None
    assert result.query.temporal.recogniser == "last_weekday"
    assert result.query.temporal.range_start.date() == datetime(2023, 4, 8).date()


async def test_parse_no_temporal_when_not_matched() -> None:
    opts = Options(anchor=ANCHOR)
    result = await parse("What is the capital of France?", opts)
    assert result.query.temporal is None


async def test_parse_without_distill_skips_provider() -> None:
    provider = CountingProvider()
    opts = Options(provider=provider, distill=False)
    await parse("whatever input", opts)
    assert provider.calls == 0


async def test_parse_with_distill_calls_provider_once_then_caches() -> None:
    provider = CountingProvider(response="distilled form")
    opts = Options(provider=provider, distill=True, cache=True)

    first = await parse("kubernetes argocd sync failure", opts)
    second = await parse("kubernetes argocd sync failure", opts)

    assert provider.calls == 1
    assert first.query.distilled == "distilled form"
    assert first.trace.distilled is True
    assert first.trace.used_cache is False
    assert second.trace.used_cache is True
    assert second.query.distilled == "distilled form"


async def test_parse_with_distill_cache_disabled_bypasses_cache() -> None:
    provider = CountingProvider(response="distilled")
    opts = Options(provider=provider, distill=True, cache=False)

    await parse("kubernetes argocd sync failure", opts)
    await parse("kubernetes argocd sync failure", opts)
    assert provider.calls == 2


async def test_parse_distill_with_empty_query_is_noop() -> None:
    provider = CountingProvider()
    opts = Options(provider=provider, distill=True)
    result = await parse("   ", opts)
    assert result.query.distilled is None
    assert provider.calls == 0


async def test_parse_distill_propagates_provider_error() -> None:
    provider = CountingProvider(error=RuntimeError("oops"))
    opts = Options(provider=provider, distill=True)
    with pytest.raises(RuntimeError, match="oops"):
        await parse("a longer query string", opts)


async def test_parse_distill_different_models_do_not_collide() -> None:
    provider = CountingProvider(response="x")
    opts_a = Options(provider=provider, distill=True, model="sonnet")
    opts_b = Options(provider=provider, distill=True, model="haiku")

    await parse("some input", opts_a)
    await parse("some input", opts_b)
    assert provider.calls == 2


async def test_parse_normalisation_folds_whitespace_and_case() -> None:
    result = await parse("HELLO\u00a0world\u200bTEST")
    assert result.query.normalised == "hello worldtest"
    assert result.query.tokens == ["hello", "worldtest"]
