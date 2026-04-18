# SPDX-License-Identifier: Apache-2.0
"""Filesystem-backed `Store`.

Writes are atomic via temp-file + fsync + rename. Batches buffer
mutations in an ordered in-memory journal and replay them on commit;
on error the journal is discarded without touching the working tree.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

from ..errors import ErrNotFound, ErrReadOnly
from ..path import BrainPath, validate_path
from . import (
    Batch,
    BatchOptions,
    ChangeEvent,
    ChangeKind,
    FileInfo,
    ListOpts,
    Store,
)
from ._util import atomic_write, list_dir, path_under, resolve


class FsStore(Store):
    """Store backed by a directory on disk."""

    def __init__(self, root: Path | str) -> None:
        self.root = str(Path(root).resolve())
        os.makedirs(self.root, exist_ok=True)
        self._lock = asyncio.Lock()
        self._closed = False
        self._sinks: dict[int, Callable[[ChangeEvent], None]] = {}
        self._next_id = 0
        self._event_queues: list[asyncio.Queue[ChangeEvent]] = []

    # --- helpers ---------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise ErrReadOnly("fsstore: closed")

    def _resolve(self, path: BrainPath | str) -> str:
        return resolve(self.root, path)

    def _dispatch(self, evt: ChangeEvent) -> None:
        for sink in list(self._sinks.values()):
            try:
                sink(evt)
            except Exception:
                pass
        for queue in list(self._event_queues):
            queue.put_nowait(evt)

    # --- read side -------------------------------------------------------

    async def read(self, path: BrainPath) -> bytes:
        self._check_open()
        abs_path = self._resolve(path)
        try:
            with open(abs_path, "rb") as f:
                return f.read()
        except FileNotFoundError as exc:
            raise ErrNotFound(f"fsstore: read {path}: not found") from exc

    async def exists(self, path: BrainPath) -> bool:
        self._check_open()
        abs_path = self._resolve(path)
        return os.path.exists(abs_path)

    async def stat(self, path: BrainPath) -> FileInfo:
        self._check_open()
        abs_path = self._resolve(path)
        try:
            st = os.stat(abs_path)
        except FileNotFoundError as exc:
            raise ErrNotFound(f"fsstore: stat {path}: not found") from exc
        return FileInfo(
            path=BrainPath(str(path)),
            size=st.st_size,
            mtime=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
            is_dir=os.path.isdir(abs_path),
        )

    async def list(
        self,
        dir: BrainPath | str = "",
        opts: ListOpts | None = None,
    ) -> list[FileInfo]:
        self._check_open()
        opts = opts or ListOpts()
        abs_dir = self.root if not str(dir) else self._resolve(dir)
        return list_dir(self.root, abs_dir, dir, opts)

    # --- write side ------------------------------------------------------

    async def write(self, path: BrainPath, content: bytes) -> None:
        self._check_open()
        abs_path = self._resolve(path)
        existed = os.path.exists(abs_path)
        atomic_write(abs_path, content)
        kind = ChangeKind.UPDATED if existed else ChangeKind.CREATED
        self._dispatch(ChangeEvent(kind=kind, path=BrainPath(str(path)), when=_now()))

    async def append(self, path: BrainPath, content: bytes) -> None:
        self._check_open()
        abs_path = self._resolve(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        existed = os.path.exists(abs_path)
        with open(abs_path, "ab") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        kind = ChangeKind.UPDATED if existed else ChangeKind.CREATED
        self._dispatch(ChangeEvent(kind=kind, path=BrainPath(str(path)), when=_now()))

    async def delete(self, path: BrainPath) -> None:
        self._check_open()
        abs_path = self._resolve(path)
        try:
            os.remove(abs_path)
        except FileNotFoundError as exc:
            raise ErrNotFound(f"fsstore: delete {path}: not found") from exc
        self._dispatch(
            ChangeEvent(kind=ChangeKind.DELETED, path=BrainPath(str(path)), when=_now())
        )

    async def rename(self, src: BrainPath, dst: BrainPath) -> None:
        self._check_open()
        src_abs = self._resolve(src)
        dst_abs = self._resolve(dst)
        if not os.path.exists(src_abs):
            raise ErrNotFound(f"fsstore: rename {src}: not found")
        os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
        os.replace(src_abs, dst_abs)
        self._dispatch(
            ChangeEvent(
                kind=ChangeKind.RENAMED,
                path=BrainPath(str(dst)),
                old_path=BrainPath(str(src)),
                when=_now(),
            )
        )

    # --- batch / subscribe / close --------------------------------------

    async def batch(
        self,
        fn: Callable[[Batch], Awaitable[None]] | Callable[[Batch], None],
        opts: BatchOptions | None = None,
    ) -> None:
        self._check_open()
        opts = opts or BatchOptions()
        b = _FsBatch(self)
        result = fn(b)
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]
        await b.commit()

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

    def local_path(self, path: BrainPath) -> str | None:
        try:
            return self._resolve(path)
        except Exception:
            return None


# ---------- batch ---------------------------------------------------------


@dataclass(slots=True)
class _Op:
    kind: str  # write | append | delete | rename
    path: BrainPath
    content: bytes = b""
    src: BrainPath | None = None


class _FsBatch(Batch):
    """Journalled batch that replays mutations on commit."""

    def __init__(self, store: FsStore) -> None:
        self.store = store
        self.ops: list[_Op] = []

    async def _effective(self, path: BrainPath, upto: int | None = None) -> tuple[bytes | None, bool, bool]:
        """Return (content, present, from_store)."""
        if upto is None:
            upto = len(self.ops)
        have = False
        buf: bytes = b""
        for i in range(upto):
            op = self.ops[i]
            if op.kind == "write" and op.path == path:
                have = True
                buf = op.content
            elif op.kind == "append" and op.path == path:
                if not have:
                    try:
                        existing = await self.store.read(path)
                    except ErrNotFound:
                        existing = b""
                    buf = existing
                    have = True
                buf = buf + op.content
            elif op.kind == "delete" and op.path == path:
                return (None, False, False)
            elif op.kind == "rename":
                if op.src == path:
                    return (None, False, False)
                if op.path == path:
                    sub_content, sub_present, _ = await self._effective(op.src, i)  # type: ignore[arg-type]
                    if sub_present:
                        have = True
                        buf = sub_content or b""
                    else:
                        return (None, False, False)
        if have:
            return (buf, True, False)
        try:
            data = await self.store.read(path)
            return (data, True, True)
        except ErrNotFound:
            return (None, False, True)

    # --- read side ----
    async def read(self, path: BrainPath) -> bytes:
        validate_path(str(path))
        content, present, _ = await self._effective(path)
        if not present:
            raise ErrNotFound(f"fsstore: read {path}: not found")
        assert content is not None
        return content

    async def exists(self, path: BrainPath) -> bool:
        validate_path(str(path))
        _, present, _ = await self._effective(path)
        return present

    async def stat(self, path: BrainPath) -> FileInfo:
        validate_path(str(path))
        content, present, from_store = await self._effective(path)
        if not present:
            raise ErrNotFound(f"fsstore: stat {path}: not found")
        if from_store:
            return await self.store.stat(path)
        assert content is not None
        return FileInfo(path=path, size=len(content), mtime=_now(), is_dir=False)

    async def list(
        self, dir: BrainPath | str = "", opts: ListOpts | None = None
    ) -> list[FileInfo]:
        opts = opts or ListOpts()
        base = await self.store.list(dir, opts)
        by_path: dict[BrainPath, FileInfo] = {fi.path: fi for fi in base}
        touched: set[BrainPath] = set()
        for op in self.ops:
            touched.add(op.path)
            if op.kind == "rename" and op.src is not None:
                touched.add(op.src)
        for p in touched:
            if not path_under(str(p), str(dir), opts.recursive):
                continue
            content, present, _ = await self._effective(p)
            if not present:
                by_path.pop(p, None)
                continue
            assert content is not None
            by_path[p] = FileInfo(path=p, size=len(content), mtime=_now(), is_dir=False)
        from ..path import is_generated  # local import avoids cycle at module top.

        result = [
            fi
            for fi in by_path.values()
            if opts.include_generated or not is_generated(fi.path)
        ]
        result.sort(key=lambda fi: fi.path)
        return result

    # --- write side ---
    async def write(self, path: BrainPath, content: bytes) -> None:
        validate_path(str(path))
        self.ops.append(_Op(kind="write", path=BrainPath(str(path)), content=bytes(content)))

    async def append(self, path: BrainPath, content: bytes) -> None:
        validate_path(str(path))
        self.ops.append(_Op(kind="append", path=BrainPath(str(path)), content=bytes(content)))

    async def delete(self, path: BrainPath) -> None:
        validate_path(str(path))
        _, present, _ = await self._effective(path)
        if not present:
            raise ErrNotFound(f"fsstore: delete {path}: not found")
        self.ops.append(_Op(kind="delete", path=BrainPath(str(path))))

    async def rename(self, src: BrainPath, dst: BrainPath) -> None:
        validate_path(str(src))
        validate_path(str(dst))
        _, present, _ = await self._effective(src)
        if not present:
            raise ErrNotFound(f"fsstore: rename {src}: not found")
        self.ops.append(_Op(kind="rename", path=BrainPath(str(dst)), src=BrainPath(str(src))))

    # --- commit -------
    async def commit(self) -> None:
        if not self.ops:
            return
        touched: list[BrainPath] = []
        seen: set[BrainPath] = set()
        for op in self.ops:
            for p in ([op.path] + ([op.src] if op.kind == "rename" and op.src else [])):
                if p not in seen:
                    seen.add(p)
                    touched.append(p)
        plan: list[tuple[str, BrainPath, bytes]] = []
        for p in touched:
            content, present, from_store = await self._effective(p)
            if present:
                if from_store:
                    continue
                assert content is not None
                plan.append(("write", p, content))
                continue
            if await self.store.exists(p):
                plan.append(("delete", p, b""))
        for kind, p, content in plan:
            if kind == "write":
                try:
                    current = await self.store.read(p)
                    if current == content:
                        continue
                except ErrNotFound:
                    pass
                await self.store.write(p, content)
            else:
                try:
                    await self.store.delete(p)
                except ErrNotFound:
                    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)
