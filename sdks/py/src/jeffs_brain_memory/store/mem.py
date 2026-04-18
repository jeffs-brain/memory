# SPDX-License-Identifier: Apache-2.0
"""In-memory `Store` implementation — primarily used by tests.

Safe for concurrent asyncio use via a single lock. Honours the full
store contract including batch read-your-own-writes, event subscription,
and sentinel exceptions.
"""

from __future__ import annotations

import asyncio
import fnmatch
import posixpath
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable

from ..errors import ErrNotFound, ErrReadOnly
from ..path import BrainPath, is_generated, validate_path
from . import (
    Batch,
    BatchOptions,
    ChangeEvent,
    ChangeKind,
    FileInfo,
    ListOpts,
    Store,
)


@dataclass(slots=True)
class _Entry:
    content: bytes
    mtime: datetime


class MemStore(Store):
    """In-memory store backed by a dict keyed by path."""

    def __init__(self) -> None:
        self._files: dict[BrainPath, _Entry] = {}
        self._lock = asyncio.Lock()
        self._sinks: dict[int, Callable[[ChangeEvent], None]] = {}
        self._next_id = 0
        self._closed = False
        self._event_queues: list[asyncio.Queue[ChangeEvent]] = []

    # --- helpers ---------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise ErrReadOnly("memstore: closed")

    def _dispatch(self, evt: ChangeEvent) -> None:
        for sink in list(self._sinks.values()):
            try:
                sink(evt)
            except Exception:
                # Sinks must not break the mutation path.
                pass
        for queue in list(self._event_queues):
            queue.put_nowait(evt)

    # --- read side -------------------------------------------------------

    async def read(self, path: BrainPath) -> bytes:
        self._check_open()
        validate_path(str(path))
        async with self._lock:
            entry = self._files.get(path)
            if entry is None:
                raise ErrNotFound(f"memstore: read {path}: not found")
            return bytes(entry.content)

    async def exists(self, path: BrainPath) -> bool:
        self._check_open()
        validate_path(str(path))
        async with self._lock:
            if path in self._files:
                return True
            prefix = str(path) + "/"
            return any(str(k).startswith(prefix) for k in self._files)

    async def stat(self, path: BrainPath) -> FileInfo:
        self._check_open()
        validate_path(str(path))
        async with self._lock:
            entry = self._files.get(path)
            if entry is not None:
                return FileInfo(
                    path=path,
                    size=len(entry.content),
                    mtime=entry.mtime,
                    is_dir=False,
                )
            prefix = str(path) + "/"
            if any(str(k).startswith(prefix) for k in self._files):
                return FileInfo(path=path, size=0, mtime=None, is_dir=True)
            raise ErrNotFound(f"memstore: stat {path}: not found")

    async def list(
        self,
        dir: BrainPath | str = "",
        opts: ListOpts | None = None,
    ) -> list[FileInfo]:
        self._check_open()
        opts = opts or ListOpts()
        prefix = str(dir)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        async with self._lock:
            return _list_from_map(self._files, prefix, opts)

    # --- write side ------------------------------------------------------

    async def write(self, path: BrainPath, content: bytes) -> None:
        self._check_open()
        validate_path(str(path))
        async with self._lock:
            existed = path in self._files
            self._files[path] = _Entry(content=bytes(content), mtime=_now())
        kind = ChangeKind.UPDATED if existed else ChangeKind.CREATED
        self._dispatch(ChangeEvent(kind=kind, path=path, when=_now()))

    async def append(self, path: BrainPath, content: bytes) -> None:
        self._check_open()
        validate_path(str(path))
        async with self._lock:
            entry = self._files.get(path)
            existed = entry is not None
            if entry is None:
                entry = _Entry(content=b"", mtime=_now())
                self._files[path] = entry
            entry.content = entry.content + bytes(content)
            entry.mtime = _now()
        kind = ChangeKind.UPDATED if existed else ChangeKind.CREATED
        self._dispatch(ChangeEvent(kind=kind, path=path, when=_now()))

    async def delete(self, path: BrainPath) -> None:
        self._check_open()
        validate_path(str(path))
        async with self._lock:
            if path not in self._files:
                raise ErrNotFound(f"memstore: delete {path}: not found")
            del self._files[path]
        self._dispatch(ChangeEvent(kind=ChangeKind.DELETED, path=path, when=_now()))

    async def rename(self, src: BrainPath, dst: BrainPath) -> None:
        self._check_open()
        validate_path(str(src))
        validate_path(str(dst))
        async with self._lock:
            entry = self._files.get(src)
            if entry is None:
                raise ErrNotFound(f"memstore: rename {src}: not found")
            del self._files[src]
            self._files[dst] = _Entry(content=bytes(entry.content), mtime=_now())
        self._dispatch(
            ChangeEvent(kind=ChangeKind.RENAMED, path=dst, old_path=src, when=_now())
        )

    # --- batch / subscribe / close --------------------------------------

    async def batch(
        self,
        fn: Callable[[Batch], Awaitable[None]] | Callable[[Batch], None],
        opts: BatchOptions | None = None,
    ) -> None:
        self._check_open()
        opts = opts or BatchOptions()
        async with self._lock:
            snapshot: dict[BrainPath, _Entry] = {
                k: _Entry(content=bytes(v.content), mtime=v.mtime)
                for k, v in self._files.items()
            }
        b = _MemBatch(snapshot)
        result = fn(b)
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]
        async with self._lock:
            old = self._files
            self._files = b.files
        events = _diff_events(old, b.files, opts.reason)
        for evt in events:
            self._dispatch(evt)

    def subscribe(self, sink: Callable[[ChangeEvent], None]) -> Callable[[], None]:
        self._next_id += 1
        sink_id = self._next_id
        self._sinks[sink_id] = sink

        def unsubscribe() -> None:
            self._sinks.pop(sink_id, None)

        return unsubscribe

    def events(self) -> AsyncIterator[ChangeEvent]:
        queue: asyncio.Queue[ChangeEvent] = asyncio.Queue()
        self._event_queues.append(queue)

        async def iterator() -> AsyncIterator[ChangeEvent]:
            try:
                while True:
                    evt = await queue.get()
                    yield evt
            finally:
                try:
                    self._event_queues.remove(queue)
                except ValueError:
                    pass

        return iterator()

    async def close(self) -> None:
        self._closed = True
        self._sinks.clear()
        for queue in list(self._event_queues):
            try:
                queue.put_nowait(  # type: ignore[arg-type]
                    ChangeEvent(
                        kind=ChangeKind.DELETED,
                        path=BrainPath(""),
                        when=_now(),
                        reason="__closed__",
                    )
                )
            except asyncio.QueueFull:
                pass

    def local_path(self, path: BrainPath) -> str | None:
        return None


# ---------- batch helpers -------------------------------------------------


class _MemBatch(Batch):
    """Operates on a cloned snapshot of the store's map."""

    def __init__(self, files: dict[BrainPath, _Entry]) -> None:
        self.files = files

    async def read(self, path: BrainPath) -> bytes:
        validate_path(str(path))
        entry = self.files.get(path)
        if entry is None:
            raise ErrNotFound(f"memstore: read {path}: not found")
        return bytes(entry.content)

    async def exists(self, path: BrainPath) -> bool:
        validate_path(str(path))
        if path in self.files:
            return True
        prefix = str(path) + "/"
        return any(str(k).startswith(prefix) for k in self.files)

    async def stat(self, path: BrainPath) -> FileInfo:
        validate_path(str(path))
        entry = self.files.get(path)
        if entry is None:
            raise ErrNotFound(f"memstore: stat {path}: not found")
        return FileInfo(path=path, size=len(entry.content), mtime=entry.mtime, is_dir=False)

    async def list(
        self, dir: BrainPath | str = "", opts: ListOpts | None = None
    ) -> list[FileInfo]:
        opts = opts or ListOpts()
        prefix = str(dir)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return _list_from_map(self.files, prefix, opts)

    async def write(self, path: BrainPath, content: bytes) -> None:
        validate_path(str(path))
        self.files[path] = _Entry(content=bytes(content), mtime=_now())

    async def append(self, path: BrainPath, content: bytes) -> None:
        validate_path(str(path))
        entry = self.files.get(path)
        if entry is None:
            entry = _Entry(content=b"", mtime=_now())
            self.files[path] = entry
        entry.content = entry.content + bytes(content)
        entry.mtime = _now()

    async def delete(self, path: BrainPath) -> None:
        validate_path(str(path))
        if path not in self.files:
            raise ErrNotFound(f"memstore: delete {path}: not found")
        del self.files[path]

    async def rename(self, src: BrainPath, dst: BrainPath) -> None:
        validate_path(str(src))
        validate_path(str(dst))
        entry = self.files.get(src)
        if entry is None:
            raise ErrNotFound(f"memstore: rename {src}: not found")
        del self.files[src]
        self.files[dst] = _Entry(content=bytes(entry.content), mtime=_now())


def _list_from_map(
    files: dict[BrainPath, _Entry], prefix: str, opts: ListOpts
) -> list[FileInfo]:
    result: list[FileInfo] = []
    seen_dirs: set[BrainPath] = set()
    for p, entry in files.items():
        ps = str(p)
        if prefix and not ps.startswith(prefix):
            continue
        rest = ps[len(prefix) :] if prefix else ps
        if rest == "":
            continue
        if opts.recursive:
            if not opts.include_generated and is_generated(ps):
                continue
            if opts.glob and not fnmatch.fnmatchcase(_last_segment(rest), opts.glob):
                continue
            result.append(
                FileInfo(path=p, size=len(entry.content), mtime=entry.mtime, is_dir=False)
            )
            continue
        slash = rest.find("/")
        if slash == -1:
            if not opts.include_generated and is_generated(ps):
                continue
            if opts.glob and not fnmatch.fnmatchcase(rest, opts.glob):
                continue
            result.append(
                FileInfo(path=p, size=len(entry.content), mtime=entry.mtime, is_dir=False)
            )
        else:
            child_dir = BrainPath(prefix + rest[:slash])
            if child_dir not in seen_dirs:
                seen_dirs.add(child_dir)
                result.append(FileInfo(path=child_dir, size=0, mtime=None, is_dir=True))
    result.sort(key=lambda fi: fi.path)
    return result


def _last_segment(s: str) -> str:
    idx = s.rfind("/")
    if idx == -1:
        return s
    return s[idx + 1 :]


def _diff_events(
    old: dict[BrainPath, _Entry],
    new: dict[BrainPath, _Entry],
    reason: str | None,
) -> list[ChangeEvent]:
    events: list[ChangeEvent] = []
    now = _now()
    for p, e in new.items():
        prev = old.get(p)
        if prev is None:
            events.append(
                ChangeEvent(kind=ChangeKind.CREATED, path=p, when=now, reason=reason)
            )
        elif prev.content != e.content:
            events.append(
                ChangeEvent(kind=ChangeKind.UPDATED, path=p, when=now, reason=reason)
            )
    for p in old:
        if p not in new:
            events.append(
                ChangeEvent(kind=ChangeKind.DELETED, path=p, when=now, reason=reason)
            )
    return events


def _now() -> datetime:
    return datetime.now(timezone.utc)
