# SPDX-License-Identifier: Apache-2.0
"""Filesystem-backed `Store`. Stub."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

from ..path import BrainPath
from . import BatchOp, BatchResult, ChangeEvent, ListEntry, StatEntry, Store


class FsStore(Store):
    """Store backed by a directory on disk."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    async def read(self, path: BrainPath) -> bytes:
        raise NotImplementedError("FsStore.read")

    async def exists(self, path: BrainPath) -> bool:
        raise NotImplementedError("FsStore.exists")

    async def stat(self, path: BrainPath) -> StatEntry:
        raise NotImplementedError("FsStore.stat")

    async def list(
        self,
        dir: BrainPath | str = "",
        *,
        recursive: bool = False,
        include_generated: bool = False,
        glob: str | None = None,
    ) -> list[ListEntry]:
        raise NotImplementedError("FsStore.list")

    async def write(self, path: BrainPath, content: bytes) -> None:
        raise NotImplementedError("FsStore.write")

    async def append(self, path: BrainPath, content: bytes) -> None:
        raise NotImplementedError("FsStore.append")

    async def delete(self, path: BrainPath) -> None:
        raise NotImplementedError("FsStore.delete")

    async def rename(self, from_path: BrainPath, to_path: BrainPath) -> None:
        raise NotImplementedError("FsStore.rename")

    async def batch(self, ops: list[BatchOp], *, reason: str | None = None) -> BatchResult:
        raise NotImplementedError("FsStore.batch")

    def events(self) -> AsyncIterator[ChangeEvent]:
        raise NotImplementedError("FsStore.events")
