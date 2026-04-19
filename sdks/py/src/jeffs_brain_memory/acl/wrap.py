# SPDX-License-Identifier: Apache-2.0
"""`wrap_store(store, provider, subject, ...)` - guard a `Store` with ACL.

Each `Store` method maps to an `Action`:

  read, exists, stat, list, batch-read         -> 'read'
  write, append, batch-write, batch-append     -> 'write'
  delete, batch-delete                         -> 'delete'
  rename, batch-rename                         -> 'write' (both sides)

The resource targeted by a path is resolved via `resolve_resource` if
provided; otherwise the wrapper falls back to a fixed `resource`. When
neither is configured the wrapper raises `ForbiddenError` rather than
let the call through unchecked.
"""

from __future__ import annotations

from typing import AsyncIterator, Callable

from ..path import BrainPath
from ..store import (
    Batch,
    BatchOptions,
    ChangeEvent,
    FileInfo,
    ListOpts,
    Store,
)
from .types import Action, ForbiddenError, Provider, Resource, Subject

__all__ = ["wrap_store"]


def wrap_store(
    store: Store,
    provider: Provider,
    subject: Subject,
    *,
    resource: Resource | None = None,
    resolve_resource: Callable[[BrainPath], Resource] | None = None,
) -> Store:
    """Return a `Store` that runs an ACL check before every delegated call."""
    return _WrappedStore(
        store=store,
        provider=provider,
        subject=subject,
        resource=resource,
        resolve_resource=resolve_resource,
    )


class _WrappedStore(Store):
    def __init__(
        self,
        *,
        store: Store,
        provider: Provider,
        subject: Subject,
        resource: Resource | None,
        resolve_resource: Callable[[BrainPath], Resource] | None,
    ) -> None:
        self._store = store
        self._provider = provider
        self._subject = subject
        self._resource = resource
        self._resolve_resource = resolve_resource

    def _resolve(self, path: BrainPath | str) -> Resource:
        bp = BrainPath(str(path))
        if self._resolve_resource is not None:
            return self._resolve_resource(bp)
        if self._resource is not None:
            return self._resource
        raise ForbiddenError(
            self._subject,
            "read",
            Resource(type="document", id=str(path)),
            "no resource resolver configured",
        )

    async def _guard(self, action: Action, path: BrainPath | str) -> None:
        resource = self._resolve(path)
        result = await self._provider.check(self._subject, action, resource)
        if not result.allowed:
            raise ForbiddenError(self._subject, action, resource, result.reason)

    # --- read side -------------------------------------------------------

    async def read(self, path: BrainPath) -> bytes:
        await self._guard("read", path)
        return await self._store.read(path)

    async def exists(self, path: BrainPath) -> bool:
        await self._guard("read", path)
        return await self._store.exists(path)

    async def stat(self, path: BrainPath) -> FileInfo:
        await self._guard("read", path)
        return await self._store.stat(path)

    async def list(
        self,
        dir: BrainPath | str = "",
        opts: ListOpts | None = None,
    ) -> list[FileInfo]:
        if str(dir) != "":
            await self._guard("read", dir)
        return await self._store.list(dir, opts)

    # --- write side ------------------------------------------------------

    async def write(self, path: BrainPath, content: bytes) -> None:
        await self._guard("write", path)
        await self._store.write(path, content)

    async def append(self, path: BrainPath, content: bytes) -> None:
        await self._guard("write", path)
        await self._store.append(path, content)

    async def delete(self, path: BrainPath) -> None:
        await self._guard("delete", path)
        await self._store.delete(path)

    async def rename(self, src: BrainPath, dst: BrainPath) -> None:
        await self._guard("write", src)
        await self._guard("write", dst)
        await self._store.rename(src, dst)

    # --- transactional / subscriber surface ------------------------------

    async def batch(
        self,
        fn: Callable[[Batch], object],
        opts: BatchOptions | None = None,
    ) -> None:
        async def runner(inner: Batch) -> None:
            wrapped = _WrappedBatch(inner, self)
            result = fn(wrapped)
            if hasattr(result, "__await__"):
                await result

        await self._store.batch(runner, opts)

    def subscribe(self, sink: Callable[[ChangeEvent], None]) -> Callable[[], None]:
        return self._store.subscribe(sink)

    def events(self) -> AsyncIterator[ChangeEvent]:
        return self._store.events()

    async def close(self) -> None:
        await self._store.close()

    def local_path(self, path: BrainPath) -> str | None:
        # `local_path` cannot enforce ACL safely because the provider contract
        # is async. Return no path at all rather than leak a filesystem location.
        return None


class _WrappedBatch(Batch):
    """Per-op guarded view onto an underlying `Batch`."""

    def __init__(self, inner: Batch, parent: _WrappedStore) -> None:
        self._inner = inner
        self._parent = parent

    async def read(self, path: BrainPath) -> bytes:
        await self._parent._guard("read", path)
        return await self._inner.read(path)

    async def exists(self, path: BrainPath) -> bool:
        await self._parent._guard("read", path)
        return await self._inner.exists(path)

    async def stat(self, path: BrainPath) -> FileInfo:
        await self._parent._guard("read", path)
        return await self._inner.stat(path)

    async def list(
        self,
        dir: BrainPath | str = "",
        opts: ListOpts | None = None,
    ) -> list[FileInfo]:
        if str(dir) != "":
            await self._parent._guard("read", dir)
        return await self._inner.list(dir, opts)

    async def write(self, path: BrainPath, content: bytes) -> None:
        await self._parent._guard("write", path)
        await self._inner.write(path, content)

    async def append(self, path: BrainPath, content: bytes) -> None:
        await self._parent._guard("write", path)
        await self._inner.append(path, content)

    async def delete(self, path: BrainPath) -> None:
        await self._parent._guard("delete", path)
        await self._inner.delete(path)

    async def rename(self, src: BrainPath, dst: BrainPath) -> None:
        await self._parent._guard("write", src)
        await self._parent._guard("write", dst)
        await self._inner.rename(src, dst)
