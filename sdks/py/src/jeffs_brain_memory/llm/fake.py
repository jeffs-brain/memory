# SPDX-License-Identifier: Apache-2.0
"""Deterministic in-memory provider and embedder for tests and dev."""

from __future__ import annotations

import hashlib
import math
import struct
from typing import AsyncIterator

from .types import (
    CompleteRequest,
    CompleteResponse,
    EmptyMessagesError,
    StopReason,
    StreamChunk,
)


class FakeProvider:
    """Hand out responses from a fixed list in round-robin order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses: list[str] = list(responses)
        self._idx = 0
        self._closed = False

    def _next(self) -> str:
        if not self._responses:
            return ""
        value = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return value

    async def complete(self, req: CompleteRequest) -> CompleteResponse:
        if self._closed:
            raise RuntimeError("llm: fake provider closed")
        if not req.messages:
            raise EmptyMessagesError()
        text = self._next()
        tokens_in = sum(len(m.content) for m in req.messages)
        return CompleteResponse(
            text=text,
            stop=StopReason.END_TURN,
            tokens_in=tokens_in,
            tokens_out=len(text),
        )

    async def complete_stream(self, req: CompleteRequest) -> AsyncIterator[StreamChunk]:
        if self._closed:
            raise RuntimeError("llm: fake provider closed")
        if not req.messages:
            raise EmptyMessagesError()
        text = self._next()
        return _fake_stream(text)

    async def close(self) -> None:
        self._closed = True


async def _fake_stream(text: str) -> AsyncIterator[StreamChunk]:
    for ch in text:
        yield StreamChunk(delta_text=ch)
    yield StreamChunk(stop=StopReason.END_TURN)


class FakeEmbedder:
    """Emit deterministic unit-length vectors seeded by SHA-256 of each input."""

    def __init__(self, dims: int) -> None:
        self._dims = dims if dims > 0 else 16

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_seed_vector(t, self._dims) for t in texts]

    def dimensions(self) -> int:
        return self._dims

    async def close(self) -> None:
        return None


def _seed_vector(text: str, dims: int) -> list[float]:
    if dims == 0:
        return []
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    vec: list[float] = []
    sum_sq = 0.0
    for i in range(dims):
        offset = (i * 4) % len(seed)
        end = offset + 4
        if end > len(seed):
            seed = hashlib.sha256(seed + bytes([i & 0xFF])).digest()
            offset = 0
            end = 4
        raw = struct.unpack(">I", seed[offset:end])[0]
        val = (raw / 0xFFFFFFFF) * 2.0 - 1.0
        vec.append(val)
        sum_sq += val * val
    if sum_sq == 0.0:
        return vec
    norm = math.sqrt(sum_sq)
    return [v / norm for v in vec]
