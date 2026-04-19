# SPDX-License-Identifier: Apache-2.0
"""Behavioural specs for `wrap_store`."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable

import pytest

from jeffs_brain_memory.acl import (
    Action,
    CheckResult,
    ForbiddenError,
    Resource,
    Subject,
    allow,
    deny,
    wrap_store,
)
from jeffs_brain_memory.acl.types import Provider, ReadTuplesQuery, Tuple, WriteTuplesRequest
from jeffs_brain_memory.errors import ErrForbidden
from jeffs_brain_memory.path import BrainPath
from jeffs_brain_memory.store import (
    Batch,
    BatchOptions,
    ChangeEvent,
    FileInfo,
    ListOpts,
    Store,
)

pytestmark = pytest.mark.asyncio


@dataclass(slots=True)
class _Call:
    method: str
    args: tuple[object, ...]


class _StubStore(Store):
    """Records every call without performing any side effects."""

    def __init__(self) -> None:
        self.calls: list[_Call] = []

    def _record(self, method: str, *args: object) -> None:
        self.calls.append(_Call(method=method, args=args))

    async def read(self, path: BrainPath) -> bytes:
        self._record("read", path)
        return b"hello"

    async def exists(self, path: BrainPath) -> bool:
        self._record("exists", path)
        return True

    async def stat(self, path: BrainPath) -> FileInfo:
        self._record("stat", path)
        return FileInfo(path=path, size=1, mtime=datetime.now(timezone.utc))

    async def list(
        self,
        dir: BrainPath | str = "",
        opts: ListOpts | None = None,
    ) -> list[FileInfo]:
        self._record("list", dir)
        return []

    async def write(self, path: BrainPath, content: bytes) -> None:
        self._record("write", path, content)

    async def append(self, path: BrainPath, content: bytes) -> None:
        self._record("append", path, content)

    async def delete(self, path: BrainPath) -> None:
        self._record("delete", path)

    async def rename(self, src: BrainPath, dst: BrainPath) -> None:
        self._record("rename", src, dst)

    async def batch(
        self,
        fn: Callable[[Batch], Awaitable[None]] | Callable[[Batch], None],
        opts: BatchOptions | None = None,
    ) -> None:
        self._record("batch")
        inner = _StubBatch(self)
        result = fn(inner)
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]

    def subscribe(self, sink: Callable[[ChangeEvent], None]) -> Callable[[], None]:
        self._record("subscribe")
        return lambda: None

    def events(self) -> AsyncIterator[ChangeEvent]:
        async def empty() -> AsyncIterator[ChangeEvent]:
            return
            yield  # pragma: no cover

        return empty()

    async def close(self) -> None:
        self._record("close")

    def local_path(self, path: BrainPath) -> str | None:
        return f"/tmp/{path}"


class _StubBatch(Batch):
    def __init__(self, parent: _StubStore) -> None:
        self._parent = parent

    async def read(self, path: BrainPath) -> bytes:
        self._parent._record("batch.read", path)
        return b"x"

    async def exists(self, path: BrainPath) -> bool:
        self._parent._record("batch.exists", path)
        return True

    async def stat(self, path: BrainPath) -> FileInfo:
        self._parent._record("batch.stat", path)
        return FileInfo(path=path, size=1, mtime=datetime.now(timezone.utc))

    async def list(
        self,
        dir: BrainPath | str = "",
        opts: ListOpts | None = None,
    ) -> list[FileInfo]:
        self._parent._record("batch.list", dir)
        return []

    async def write(self, path: BrainPath, content: bytes) -> None:
        self._parent._record("batch.write", path, content)

    async def append(self, path: BrainPath, content: bytes) -> None:
        self._parent._record("batch.append", path, content)

    async def delete(self, path: BrainPath) -> None:
        self._parent._record("batch.delete", path)

    async def rename(self, src: BrainPath, dst: BrainPath) -> None:
        self._parent._record("batch.rename", src, dst)


@dataclass(slots=True)
class _SpyProvider:
    name: str = "spy"
    result: CheckResult = field(default_factory=lambda: allow())
    calls: list[tuple[Subject, Action, Resource]] = field(default_factory=list)

    async def check(
        self, subject: Subject, action: Action, resource: Resource
    ) -> CheckResult:
        self.calls.append((subject, action, resource))
        return self.result

    async def write(self, request: WriteTuplesRequest) -> None:
        return None

    async def read(self, query: ReadTuplesQuery) -> list[Tuple]:
        return []

    async def close(self) -> None:
        return None


_DENY_ALL: Provider = _SpyProvider(name="deny-all", result=deny("deny-all"))
_ALLOW_ALL: Provider = _SpyProvider(name="allow-all", result=allow())


def _alice() -> Subject:
    return Subject(kind="user", id="alice")


def _brain_resource() -> Resource:
    return Resource(type="brain", id="notes")


async def test_denied_read_raises_forbidden_and_skips_store() -> None:
    inner = _StubStore()
    wrapped = wrap_store(
        inner, _SpyProvider(result=deny("nope")), _alice(), resource=_brain_resource()
    )
    with pytest.raises(ForbiddenError):
        await wrapped.read(BrainPath("foo"))
    assert all(c.method != "read" for c in inner.calls)


async def test_denied_write_raises_forbidden_and_skips_store() -> None:
    inner = _StubStore()
    wrapped = wrap_store(
        inner, _SpyProvider(result=deny("nope")), _alice(), resource=_brain_resource()
    )
    with pytest.raises(ForbiddenError):
        await wrapped.write(BrainPath("foo"), b"x")
    assert all(c.method != "write" for c in inner.calls)


async def test_denied_delete_raises_forbidden_and_skips_store() -> None:
    inner = _StubStore()
    wrapped = wrap_store(
        inner, _SpyProvider(result=deny("nope")), _alice(), resource=_brain_resource()
    )
    with pytest.raises(ForbiddenError):
        await wrapped.delete(BrainPath("foo"))
    assert all(c.method != "delete" for c in inner.calls)


async def test_allowed_call_delegates_and_returns_inner_result() -> None:
    inner = _StubStore()
    wrapped = wrap_store(inner, _SpyProvider(), _alice(), resource=_brain_resource())
    out = await wrapped.read(BrainPath("foo"))
    assert out == b"hello"
    assert [(c.method, c.args) for c in inner.calls] == [("read", (BrainPath("foo"),))]


async def test_check_called_with_correct_subject_action_resource() -> None:
    spy = _SpyProvider()
    inner = _StubStore()
    wrapped = wrap_store(
        inner,
        spy,
        _alice(),
        resolve_resource=lambda p: Resource(type="document", id=str(p)),
    )
    await wrapped.delete(BrainPath("doc-1"))
    assert spy.calls == [
        (_alice(), "delete", Resource(type="document", id="doc-1"))
    ]


async def test_rename_guards_both_src_and_dst() -> None:
    spy = _SpyProvider()
    inner = _StubStore()
    wrapped = wrap_store(
        inner,
        spy,
        _alice(),
        resolve_resource=lambda p: Resource(type="document", id=str(p)),
    )
    await wrapped.rename(BrainPath("src"), BrainPath("dst"))
    assert len(spy.calls) == 2
    assert spy.calls[0] == (_alice(), "write", Resource(type="document", id="src"))
    assert spy.calls[1] == (_alice(), "write", Resource(type="document", id="dst"))


async def test_every_op_inside_a_batch_is_guarded() -> None:
    inner = _StubStore()
    wrapped = wrap_store(
        inner,
        _SpyProvider(result=deny("nope")),
        _alice(),
        resource=_brain_resource(),
    )

    async def work(b: Batch) -> None:
        await b.read(BrainPath("foo"))

    with pytest.raises(ForbiddenError):
        await wrapped.batch(work, BatchOptions(reason="test"))
    # The batch wrapper itself runs, but the inner read must be blocked
    # before the stub's batch.read fires.
    assert all(c.method != "batch.read" for c in inner.calls)


async def test_local_path_always_returns_none() -> None:
    inner = _StubStore()
    wrapped = wrap_store(inner, _SpyProvider(), _alice(), resource=_brain_resource())
    assert wrapped.local_path(BrainPath("foo")) is None


async def test_wrapped_store_passes_isinstance_check() -> None:
    inner = _StubStore()
    wrapped = wrap_store(inner, _SpyProvider(), _alice(), resource=_brain_resource())
    assert isinstance(wrapped, Store)


async def test_forbidden_error_is_caught_by_err_forbidden_handler() -> None:
    inner = _StubStore()
    wrapped = wrap_store(
        inner, _SpyProvider(result=deny("nope")), _alice(), resource=_brain_resource()
    )
    caught: bool = False
    try:
        await wrapped.read(BrainPath("foo"))
    except ErrForbidden:
        caught = True
    assert caught is True
