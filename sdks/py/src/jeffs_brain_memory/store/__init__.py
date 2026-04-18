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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import AsyncIterator, Callable, Iterable

from ..path import BrainPath

__all__ = [
    "ChangeKind",
    "ChangeEvent",
    "FileInfo",
    "StatEntry",
    "ListEntry",
    "BatchOp",
    "BatchOpKind",
    "BatchOptions",
    "ListOpts",
    "Batch",
    "Store",
]


class ChangeKind(str, Enum):
    """Classifies a mutation emitted by a store to its subscribers."""

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass(frozen=True, slots=True)
class ChangeEvent:
    """A single successful mutation observed by a subscriber.

    For renames, `path` holds the destination and `old_path` holds the
    source. `reason` is propagated from the enclosing `BatchOptions` when
    the mutation is part of a batch, and is empty for standalone writes.
    """

    kind: ChangeKind
    path: BrainPath
    when: datetime
    old_path: BrainPath | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class FileInfo:
    """Metadata for a single brain entry returned by `stat` or `list`."""

    path: BrainPath
    size: int = 0
    mtime: datetime | None = None
    is_dir: bool = False


# Aliases — the spec uses several names for the same shape.
StatEntry = FileInfo
ListEntry = FileInfo


class BatchOpKind(str, Enum):
    """The kind of a single batch operation."""

    WRITE = "write"
    APPEND = "append"
    DELETE = "delete"
    RENAME = "rename"


@dataclass(frozen=True, slots=True)
class BatchOp:
    """A single operation queued inside a batch."""

    kind: BatchOpKind
    path: BrainPath
    content: bytes | None = None
    src: BrainPath | None = None


@dataclass(slots=True)
class BatchOptions:
    """Controls how a batch commits."""

    reason: str | None = None
    message: str | None = None
    author: str | None = None
    email: str | None = None


@dataclass(slots=True)
class ListOpts:
    """Tunes a `Store.list` call."""

    recursive: bool = False
    glob: str | None = None
    include_generated: bool = False


class Batch(ABC):
    """Transactional handle — buffers mutations and commits them as one unit."""

    @abstractmethod
    async def read(self, path: BrainPath) -> bytes: ...

    @abstractmethod
    async def exists(self, path: BrainPath) -> bool: ...

    @abstractmethod
    async def stat(self, path: BrainPath) -> FileInfo: ...

    @abstractmethod
    async def list(self, dir: BrainPath | str = "", opts: ListOpts | None = None) -> list[FileInfo]: ...

    @abstractmethod
    async def write(self, path: BrainPath, content: bytes) -> None: ...

    @abstractmethod
    async def append(self, path: BrainPath, content: bytes) -> None: ...

    @abstractmethod
    async def delete(self, path: BrainPath) -> None: ...

    @abstractmethod
    async def rename(self, src: BrainPath, dst: BrainPath) -> None: ...


class Store(ABC):
    """Abstract document store. See `spec/STORAGE.md`."""

    # --- read side -------------------------------------------------------

    @abstractmethod
    async def read(self, path: BrainPath) -> bytes: ...

    @abstractmethod
    async def exists(self, path: BrainPath) -> bool: ...

    @abstractmethod
    async def stat(self, path: BrainPath) -> FileInfo: ...

    @abstractmethod
    async def list(
        self,
        dir: BrainPath | str = "",
        opts: ListOpts | None = None,
    ) -> list[FileInfo]: ...

    # --- write side ------------------------------------------------------

    @abstractmethod
    async def write(self, path: BrainPath, content: bytes) -> None: ...

    @abstractmethod
    async def append(self, path: BrainPath, content: bytes) -> None: ...

    @abstractmethod
    async def delete(self, path: BrainPath) -> None: ...

    @abstractmethod
    async def rename(self, src: BrainPath, dst: BrainPath) -> None: ...

    # --- transactional / subscriber surface ------------------------------

    @abstractmethod
    async def batch(
        self,
        fn: "Callable[[Batch], object]",
        opts: BatchOptions | None = None,
    ) -> None: ...

    @abstractmethod
    def subscribe(self, sink: "Callable[[ChangeEvent], None]") -> Callable[[], None]: ...

    @abstractmethod
    def events(self) -> AsyncIterator[ChangeEvent]: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def local_path(self, path: BrainPath) -> str | None: ...
