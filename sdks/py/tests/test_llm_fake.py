# SPDX-License-Identifier: Apache-2.0
"""Tests for the deterministic fake provider and embedder."""

from __future__ import annotations

import math

import pytest

from jeffs_brain_memory.llm import (
    CompleteRequest,
    EmptyMessagesError,
    FakeEmbedder,
    FakeProvider,
    Message,
    Role,
    StopReason,
)


async def test_fake_round_robin() -> None:
    provider = FakeProvider(["one", "two", "three"])
    req = CompleteRequest(messages=[Message(role=Role.USER, content="hi")])
    wanted = ["one", "two", "three", "one", "two"]
    for want in wanted:
        resp = await provider.complete(req)
        assert resp.text == want
        assert resp.stop is StopReason.END_TURN
    await provider.close()


async def test_fake_stream() -> None:
    provider = FakeProvider(["abc"])
    req = CompleteRequest(messages=[Message(role=Role.USER, content="hi")])
    gen = await provider.complete_stream(req)
    text = ""
    stop: StopReason | None = None
    async for chunk in gen:
        text += chunk.delta_text
        if chunk.stop is not None:
            stop = chunk.stop
    assert text == "abc"
    assert stop is StopReason.END_TURN
    await provider.close()


async def test_fake_rejects_empty_messages() -> None:
    provider = FakeProvider(["x"])
    with pytest.raises(EmptyMessagesError):
        await provider.complete(CompleteRequest())


async def test_fake_embedder_deterministic() -> None:
    embedder = FakeEmbedder(16)
    a = await embedder.embed(["hello", "world"])
    b = await embedder.embed(["hello", "world"])
    assert len(a) == 2
    assert len(a[0]) == 16
    assert a == b
    assert embedder.dimensions() == 16
    await embedder.close()


async def test_fake_embedder_unit_length() -> None:
    embedder = FakeEmbedder(32)
    vectors = await embedder.embed(["alpha", "beta"])
    for vector in vectors:
        norm = math.sqrt(sum(x * x for x in vector))
        assert abs(norm - 1.0) < 1e-5


async def test_fake_embedder_distinct_inputs() -> None:
    embedder = FakeEmbedder(8)
    vectors = await embedder.embed(["a", "b"])
    assert vectors[0] != vectors[1]
