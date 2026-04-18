# SPDX-License-Identifier: Apache-2.0
"""In-memory `Store`. Stub. Primarily used by tests."""

from __future__ import annotations

from typing import AsyncIterator

from ..path import BrainPath
from . import BatchOp, BatchResult, ChangeEvent, ListEntry, StatEntry, Store


class MemStore(Store):
    """Store that holds all documents in a dict."""

    def __init__(self) -> None:
        self._docs: dict[str, bytes] = {}

    async def read(self, path: BrainPath) -> bytes:
        raise NotImplementedError("MemStore.read")

    async def exists(self, path: BrainPath) -> bool:
        raise NotImplementedError("MemStore.exists")

    async def stat(self, path: BrainPath) -> StatEntry:
        raise NotImplementedError("MemStore.stat")

    async def list(
        self,
        dir: BrainPath | str = "",
        *,
        recursive: bool = False,
        include_generated: bool = False,
        glob: str | None = None,
    ) -> list[ListEntry]:
        raise NotImplementedError("MemStore.list")

    async def write(self, path: BrainPath, content: bytes) -> None:
        raise NotImplementedError("MemStore.write")

    async def append(self, path: BrainPath, content: bytes) -> None:
        raise NotImplementedError("MemStore.append")

    async def delete(self, path: BrainPath) -> None:
        raise NotImplementedError("MemStore.delete")

    async def rename(self, from_path: BrainPath, to_path: BrainPath) -> None:
        raise NotImplementedError("MemStore.rename")

    async def batch(self, ops: list[BatchOp], *, reason: str | None = None) -> BatchResult:
        raise NotImplementedError("MemStore.batch")

    def events(self) -> AsyncIterator[ChangeEvent]:
        raise NotImplementedError("MemStore.events")
