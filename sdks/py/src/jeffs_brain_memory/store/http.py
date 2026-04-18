# SPDX-License-Identifier: Apache-2.0
"""HTTP-backed `Store` matching `spec/PROTOCOL.md`. Stub."""

from __future__ import annotations

from typing import AsyncIterator

from ..path import BrainPath
from . import BatchOp, BatchResult, ChangeEvent, ListEntry, StatEntry, Store


class HttpStore(Store):
    """Store that talks to a remote `memory serve` daemon."""

    def __init__(
        self,
        base_url: str,
        brain_id: str,
        *,
        api_key: str | None = None,
        token: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.brain_id = brain_id
        self.api_key = api_key
        self.token = token
        self.timeout_s = timeout_s

    async def read(self, path: BrainPath) -> bytes:
        raise NotImplementedError("HttpStore.read")

    async def exists(self, path: BrainPath) -> bool:
        raise NotImplementedError("HttpStore.exists")

    async def stat(self, path: BrainPath) -> StatEntry:
        raise NotImplementedError("HttpStore.stat")

    async def list(
        self,
        dir: BrainPath | str = "",
        *,
        recursive: bool = False,
        include_generated: bool = False,
        glob: str | None = None,
    ) -> list[ListEntry]:
        raise NotImplementedError("HttpStore.list")

    async def write(self, path: BrainPath, content: bytes) -> None:
        raise NotImplementedError("HttpStore.write")

    async def append(self, path: BrainPath, content: bytes) -> None:
        raise NotImplementedError("HttpStore.append")

    async def delete(self, path: BrainPath) -> None:
        raise NotImplementedError("HttpStore.delete")

    async def rename(self, from_path: BrainPath, to_path: BrainPath) -> None:
        raise NotImplementedError("HttpStore.rename")

    async def batch(self, ops: list[BatchOp], *, reason: str | None = None) -> BatchResult:
        raise NotImplementedError("HttpStore.batch")

    def events(self) -> AsyncIterator[ChangeEvent]:
        raise NotImplementedError("HttpStore.events")
