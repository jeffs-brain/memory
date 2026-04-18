# SPDX-License-Identifier: Apache-2.0
"""Contextualiser: cache, disable, formatting."""

from __future__ import annotations

import pytest

from jeffs_brain_memory.llm.fake import FakeProvider
from jeffs_brain_memory.memory import (
    Contextualiser,
    ContextualiserConfig,
    apply_contextual_prefix,
    new_contextualiser,
)


@pytest.mark.asyncio
async def test_build_prefix_returns_trimmed(tmp_path):
    p = FakeProvider(
        ["  Context: On Monday the user discussed\n  the blue car they bought.  "]
    )
    c = Contextualiser(
        ContextualiserConfig(provider=p, cache_dir=str(tmp_path))
    )
    out = await c.build_prefix_async("sess-1", "session_date=2024-03-25", "Blue car.")
    assert out
    assert "\n" not in out
    assert not out.lower().startswith("context:")
    assert "blue car" in out


@pytest.mark.asyncio
async def test_caches_by_fingerprint(tmp_path):
    class Counter:
        def __init__(self):
            self.calls = 0

        async def complete(self, req):
            self.calls += 1
            from jeffs_brain_memory.llm.types import CompleteResponse, StopReason

            return CompleteResponse(text="Cached prefix.", stop=StopReason.END_TURN)

        def complete_stream(self, req):
            async def _gen():
                yield
            return _gen()

        async def close(self):
            pass

    counter = Counter()
    c = Contextualiser(
        ContextualiserConfig(provider=counter, cache_dir=str(tmp_path))
    )
    a = await c.build_prefix_async("s", "summary", "body")
    b = await c.build_prefix_async("s", "summary", "body")
    assert a == b
    assert counter.calls == 1


@pytest.mark.asyncio
async def test_fails_open_on_provider_error(tmp_path):
    class Broken:
        async def complete(self, req):
            raise RuntimeError("boom")

        def complete_stream(self, req):
            async def _gen():
                yield
            return _gen()

        async def close(self):
            pass

    c = Contextualiser(
        ContextualiserConfig(provider=Broken(), cache_dir=str(tmp_path))
    )
    assert await c.build_prefix_async("s", "summary", "body") == ""


def test_new_contextualiser_nil_when_no_provider(tmp_path):
    assert new_contextualiser(ContextualiserConfig(provider=None, cache_dir=str(tmp_path))) is None


def test_apply_prefix_shapes():
    assert apply_contextual_prefix("", "fact") == "fact"
    assert apply_contextual_prefix("session", "fact") == "Context: session\n\nfact"
    assert apply_contextual_prefix("   ", "fact") == "fact"


@pytest.mark.asyncio
async def test_cache_busts_on_model_change(tmp_path):
    class Counter:
        def __init__(self, reply):
            self.reply = reply
            self.calls = 0

        async def complete(self, req):
            self.calls += 1
            from jeffs_brain_memory.llm.types import CompleteResponse

            return CompleteResponse(text=self.reply)

        def complete_stream(self, req):
            async def _gen():
                yield
            return _gen()

        async def close(self):
            pass

    p1 = Counter("first-model prefix.")
    c1 = Contextualiser(
        ContextualiserConfig(provider=p1, model="model-a", cache_dir=str(tmp_path))
    )
    await c1.build_prefix_async("s", "", "body")

    p2 = Counter("second-model prefix.")
    c2 = Contextualiser(
        ContextualiserConfig(provider=p2, model="model-b", cache_dir=str(tmp_path))
    )
    out = await c2.build_prefix_async("s", "", "body")
    assert "second-model prefix" in out
    assert p2.calls == 1
