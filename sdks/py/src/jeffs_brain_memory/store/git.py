# SPDX-License-Identifier: Apache-2.0
"""Git-backed `Store` via `pygit2`. Stub."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

from ..path import BrainPath
from . import BatchOp, BatchResult, ChangeEvent, ListEntry, StatEntry, Store


class GitStore(Store):
    """Store backed by a git repository."""

    def __init__(self, root: Path | str, *, remote: str | None = None) -> None:
        self.root = Path(root)
        self.remote = remote

    async def read(self, path: BrainPath) -> bytes:
        raise NotImplementedError("GitStore.read")

    async def exists(self, path: BrainPath) -> bool:
        raise NotImplementedError("GitStore.exists")

    async def stat(self, path: BrainPath) -> StatEntry:
        raise NotImplementedError("GitStore.stat")

    async def list(
        self,
        dir: BrainPath | str = "",
        *,
        recursive: bool = False,
        include_generated: bool = False,
        glob: str | None = None,
    ) -> list[ListEntry]:
        raise NotImplementedError("GitStore.list")

    async def write(self, path: BrainPath, content: bytes) -> None:
        raise NotImplementedError("GitStore.write")

    async def append(self, path: BrainPath, content: bytes) -> None:
        raise NotImplementedError("GitStore.append")

    async def delete(self, path: BrainPath) -> None:
        raise NotImplementedError("GitStore.delete")

    async def rename(self, from_path: BrainPath, to_path: BrainPath) -> None:
        raise NotImplementedError("GitStore.rename")

    async def batch(self, ops: list[BatchOp], *, reason: str | None = None) -> BatchResult:
        raise NotImplementedError("GitStore.batch")

    def events(self) -> AsyncIterator[ChangeEvent]:
        raise NotImplementedError("GitStore.events")
