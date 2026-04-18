# SPDX-License-Identifier: Apache-2.0
"""Store protocol — document persistence layer.

Concrete implementations:

- `FsStore` — filesystem, the default local store.
- `MemStore` — in-memory, for tests.
- `GitStore` — git-backed via `pygit2`.
- `HttpStore` — remote daemon matching `spec/PROTOCOL.md`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator

from ..path import BrainPath

__all__ = [
    "Store",
    "StatEntry",
    "ListEntry",
    "ChangeEvent",
    "BatchOp",
    "BatchResult",
]


@dataclass(frozen=True, slots=True)
class StatEntry:
    path: BrainPath
    size: int
    mtime: datetime
    is_dir: bool


@dataclass(frozen=True, slots=True)
class ListEntry:
    path: BrainPath
    size: int
    mtime: datetime
    is_dir: bool


@dataclass(frozen=True, slots=True)
class ChangeEvent:
    kind: str  # "created" | "updated" | "deleted" | "renamed"
    path: BrainPath
    when: datetime
    old_path: BrainPath | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class BatchOp:
    type: str  # "write" | "append" | "delete" | "rename"
    path: BrainPath
    content: bytes | None = None
    to: BrainPath | None = None


@dataclass(frozen=True, slots=True)
class BatchResult:
    committed: int


class Store(ABC):
    """Abstract document store. See `spec/STORAGE.md`."""

    @abstractmethod
    async def read(self, path: BrainPath) -> bytes: ...

    @abstractmethod
    async def exists(self, path: BrainPath) -> bool: ...

    @abstractmethod
    async def stat(self, path: BrainPath) -> StatEntry: ...

    @abstractmethod
    async def list(
        self,
        dir: BrainPath | str = "",
        *,
        recursive: bool = False,
        include_generated: bool = False,
        glob: str | None = None,
    ) -> list[ListEntry]: ...

    @abstractmethod
    async def write(self, path: BrainPath, content: bytes) -> None: ...

    @abstractmethod
    async def append(self, path: BrainPath, content: bytes) -> None: ...

    @abstractmethod
    async def delete(self, path: BrainPath) -> None: ...

    @abstractmethod
    async def rename(self, from_path: BrainPath, to_path: BrainPath) -> None: ...

    @abstractmethod
    async def batch(self, ops: list[BatchOp], *, reason: str | None = None) -> BatchResult: ...

    @abstractmethod
    def events(self) -> AsyncIterator[ChangeEvent]: ...
