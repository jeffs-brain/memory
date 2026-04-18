# SPDX-License-Identifier: Apache-2.0
"""In-memory synchronous store backing the memory layer.

A small, dependency-free store that mirrors the subset of the Go
`brain.Store` API memory code needs: read, write, append, delete,
exists, stat, list, and batch. Async stores elsewhere in the SDK can be
wired in later; the memory layer is written to be storage-agnostic via
the :class:`Store` protocol defined below.
"""

from __future__ import annotations

import posixpath
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Protocol


class NotFoundError(LookupError):
    """Raised when a requested logical path does not exist."""


@dataclass(slots=True)
class FileInfo:
    path: str
    size: int = 0
    mod_time: datetime | None = None
    is_dir: bool = False


@dataclass(slots=True)
class ListOpts:
    recursive: bool = False
    glob: str | None = None
    include_generated: bool = False


@dataclass(slots=True)
class BatchOptions:
    reason: str = ""
    message: str = ""
    author: str = ""
    email: str = ""


class Batch(Protocol):
    def read(self, path: str) -> bytes: ...
    def write(self, path: str, content: bytes) -> None: ...
    def append(self, path: str, content: bytes) -> None: ...
    def delete(self, path: str) -> None: ...
    def exists(self, path: str) -> bool: ...
    def stat(self, path: str) -> FileInfo: ...
    def list(self, prefix: str, opts: ListOpts | None = None) -> list[FileInfo]: ...


class Store(Protocol):
    """Synchronous document store used by the memory layer."""

    def read(self, path: str) -> bytes: ...
    def write(self, path: str, content: bytes) -> None: ...
    def append(self, path: str, content: bytes) -> None: ...
    def delete(self, path: str) -> None: ...
    def exists(self, path: str) -> bool: ...
    def stat(self, path: str) -> FileInfo: ...
    def list(self, prefix: str, opts: ListOpts | None = None) -> list[FileInfo]: ...
    def batch(self, fn: Callable[[Batch], None], opts: BatchOptions | None = None) -> None: ...
    def local_path(self, path: str) -> str | None: ...


@dataclass
class _MemBatch:
    parent: "MemStore"
    pending_writes: dict[str, bytes] = field(default_factory=dict)
    pending_deletes: set[str] = field(default_factory=set)

    def read(self, path: str) -> bytes:
        if path in self.pending_deletes:
            raise NotFoundError(path)
        if path in self.pending_writes:
            return self.pending_writes[path]
        return self.parent.read(path)

    def write(self, path: str, content: bytes) -> None:
        self.pending_deletes.discard(path)
        self.pending_writes[path] = bytes(content)

    def append(self, path: str, content: bytes) -> None:
        existing = b""
        try:
            existing = self.read(path)
        except NotFoundError:
            existing = b""
        self.write(path, existing + bytes(content))

    def delete(self, path: str) -> None:
        self.pending_writes.pop(path, None)
        self.pending_deletes.add(path)

    def exists(self, path: str) -> bool:
        if path in self.pending_deletes:
            return False
        if path in self.pending_writes:
            return True
        return self.parent.exists(path)

    def stat(self, path: str) -> FileInfo:
        if path in self.pending_deletes:
            raise NotFoundError(path)
        if path in self.pending_writes:
            return FileInfo(
                path=path,
                size=len(self.pending_writes[path]),
                mod_time=datetime.now(timezone.utc),
            )
        return self.parent.stat(path)

    def list(self, prefix: str, opts: ListOpts | None = None) -> list[FileInfo]:
        opts = opts or ListOpts()
        snapshot: dict[str, bytes] = dict(self.parent._docs)
        for p in self.pending_deletes:
            snapshot.pop(p, None)
        snapshot.update(self.pending_writes)
        return _list_from(snapshot, prefix, opts, self.parent._mtimes)


class MemStore:
    """Thread-safe in-memory store."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._docs: dict[str, bytes] = {}
        self._mtimes: dict[str, datetime] = {}

    # ---- read ----

    def read(self, path: str) -> bytes:
        with self._lock:
            if path not in self._docs:
                raise NotFoundError(path)
            return self._docs[path]

    def exists(self, path: str) -> bool:
        with self._lock:
            return path in self._docs

    def stat(self, path: str) -> FileInfo:
        with self._lock:
            if path not in self._docs:
                raise NotFoundError(path)
            return FileInfo(
                path=path,
                size=len(self._docs[path]),
                mod_time=self._mtimes.get(path),
            )

    def list(self, prefix: str, opts: ListOpts | None = None) -> list[FileInfo]:
        opts = opts or ListOpts()
        with self._lock:
            snapshot = dict(self._docs)
            mtimes = dict(self._mtimes)
        return _list_from(snapshot, prefix, opts, mtimes)

    # ---- write ----

    def write(self, path: str, content: bytes) -> None:
        with self._lock:
            self._docs[path] = bytes(content)
            self._mtimes[path] = datetime.now(timezone.utc)

    def append(self, path: str, content: bytes) -> None:
        with self._lock:
            existing = self._docs.get(path, b"")
            self._docs[path] = existing + bytes(content)
            self._mtimes[path] = datetime.now(timezone.utc)

    def delete(self, path: str) -> None:
        with self._lock:
            if path not in self._docs:
                raise NotFoundError(path)
            del self._docs[path]
            self._mtimes.pop(path, None)

    def batch(
        self,
        fn: Callable[[Batch], None],
        opts: BatchOptions | None = None,
    ) -> None:
        b = _MemBatch(parent=self)
        fn(b)
        with self._lock:
            now = datetime.now(timezone.utc)
            for p in b.pending_deletes:
                self._docs.pop(p, None)
                self._mtimes.pop(p, None)
            for p, content in b.pending_writes.items():
                self._docs[p] = content
                self._mtimes[p] = now

    def local_path(self, path: str) -> str | None:
        return None

    # ---- introspection for tests ----

    def set_mtime(self, path: str, mod_time: datetime) -> None:
        with self._lock:
            if path in self._docs:
                self._mtimes[path] = mod_time


def _list_from(
    docs: dict[str, bytes],
    prefix: str,
    opts: ListOpts,
    mtimes: dict[str, datetime],
) -> list[FileInfo]:
    prefix = prefix.rstrip("/")
    # top-level "" is treated as listing everything.
    want_prefix = prefix + "/" if prefix else ""

    # collect immediate children when not recursive, else all descendants.
    seen_dirs: set[str] = set()
    files: list[FileInfo] = []

    for path in sorted(docs.keys()):
        if want_prefix:
            if not path.startswith(want_prefix):
                continue
            rel = path[len(want_prefix) :]
        else:
            rel = path
        if not rel:
            continue

        if not opts.recursive:
            head, _, tail = rel.partition("/")
            if tail:
                dir_path = posixpath.join(prefix, head) if prefix else head
                if dir_path not in seen_dirs:
                    seen_dirs.add(dir_path)
                    files.append(FileInfo(path=dir_path, is_dir=True))
                continue

        base = posixpath.basename(path)
        if not opts.include_generated and base.startswith("_"):
            continue

        files.append(
            FileInfo(
                path=path,
                size=len(docs[path]),
                mod_time=mtimes.get(path),
            )
        )

    files.sort(key=lambda f: f.path)
    return files
