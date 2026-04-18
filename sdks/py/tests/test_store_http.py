# SPDX-License-Identifier: Apache-2.0
"""Contract tests for `HttpStore` using an in-process Starlette app."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import httpx
import pytest

from jeffs_brain_memory.errors import (
    ErrInvalidPath,
    ErrNotFound,
    ErrPayloadTooLarge,
    ErrReadOnly,
    ErrValidation,
)
from jeffs_brain_memory.path import BrainPath
from jeffs_brain_memory.store import BatchOptions, ListOpts
from jeffs_brain_memory.store.http import HttpStore

from ._fake_server import build_app

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=build_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def store(client: httpx.AsyncClient) -> AsyncIterator[HttpStore]:
    s = HttpStore("http://test", "brain-a", client=client)
    try:
        yield s
    finally:
        await s.close()


async def test_write_then_read(store: HttpStore) -> None:
    await store.write(BrainPath("memory/a.md"), b"hello")
    assert await store.read(BrainPath("memory/a.md")) == b"hello"


async def test_read_missing_raises_not_found(store: HttpStore) -> None:
    with pytest.raises(ErrNotFound):
        await store.read(BrainPath("memory/nope.md"))


async def test_write_overwrites(store: HttpStore) -> None:
    await store.write(BrainPath("wiki/x.md"), b"first")
    await store.write(BrainPath("wiki/x.md"), b"second")
    assert await store.read(BrainPath("wiki/x.md")) == b"second"


async def test_append_creates_then_extends(store: HttpStore) -> None:
    await store.append(BrainPath("wiki/log.md"), b"a\n")
    await store.append(BrainPath("wiki/log.md"), b"b\n")
    assert await store.read(BrainPath("wiki/log.md")) == b"a\nb\n"


async def test_delete_removes_file(store: HttpStore) -> None:
    await store.write(BrainPath("memory/rm.md"), b"x")
    await store.delete(BrainPath("memory/rm.md"))
    assert await store.exists(BrainPath("memory/rm.md")) is False


async def test_delete_missing_raises(store: HttpStore) -> None:
    with pytest.raises(ErrNotFound):
        await store.delete(BrainPath("memory/never.md"))


async def test_rename_moves_content(store: HttpStore) -> None:
    await store.write(BrainPath("raw/old.md"), b"content")
    await store.rename(BrainPath("raw/old.md"), BrainPath("raw/new.md"))
    assert await store.read(BrainPath("raw/new.md")) == b"content"
    assert await store.exists(BrainPath("raw/old.md")) is False


async def test_rename_missing_raises(store: HttpStore) -> None:
    with pytest.raises(ErrNotFound):
        await store.rename(BrainPath("memory/nope.md"), BrainPath("memory/other.md"))


async def test_stat_reports_size(store: HttpStore) -> None:
    await store.write(BrainPath("memory/s.md"), b"twelve bytes")
    stat = await store.stat(BrainPath("memory/s.md"))
    assert stat.size == 12


async def test_stat_missing_raises(store: HttpStore) -> None:
    with pytest.raises(ErrNotFound):
        await store.stat(BrainPath("memory/nope.md"))


async def test_list_sorted(store: HttpStore) -> None:
    await store.write(BrainPath("memory/c.md"), b"c")
    await store.write(BrainPath("memory/a.md"), b"a")
    await store.write(BrainPath("memory/b.md"), b"b")
    out = await store.list("memory")
    names = [str(fi.path) for fi in out]
    assert names == ["memory/a.md", "memory/b.md", "memory/c.md"]


async def test_list_recursive(store: HttpStore) -> None:
    await store.write(BrainPath("wiki/go/a.md"), b"a")
    await store.write(BrainPath("wiki/rust/b.md"), b"b")
    out = await store.list("wiki", ListOpts(recursive=True))
    files = sorted(str(fi.path) for fi in out if not fi.is_dir)
    assert "wiki/go/a.md" in files and "wiki/rust/b.md" in files


async def test_list_hides_generated(store: HttpStore) -> None:
    await store.write(BrainPath("wiki/_index.md"), b"idx")
    await store.write(BrainPath("wiki/article.md"), b"body")
    out = await store.list("wiki")
    names = [str(fi.path) for fi in out]
    assert "wiki/article.md" in names
    assert "wiki/_index.md" not in names


async def test_list_glob(store: HttpStore) -> None:
    await store.write(BrainPath("wiki/a.md"), b"a")
    await store.write(BrainPath("wiki/b.txt"), b"b")
    await store.write(BrainPath("wiki/c.md"), b"c")
    out = await store.list("wiki", ListOpts(glob="*.md"))
    names = sorted(str(fi.path) for fi in out)
    assert names == ["wiki/a.md", "wiki/c.md"]


async def test_exists_true_for_present(store: HttpStore) -> None:
    await store.write(BrainPath("memory/hit.md"), b"x")
    assert await store.exists(BrainPath("memory/hit.md")) is True


async def test_exists_false_for_missing(store: HttpStore) -> None:
    assert await store.exists(BrainPath("memory/miss.md")) is False


async def test_invalid_path_rejected_client_side(store: HttpStore) -> None:
    with pytest.raises(ErrInvalidPath):
        await store.write(BrainPath("../bad"), b"x")


async def test_batch_commits(store: HttpStore) -> None:
    async def do_work(b):
        await b.write(BrainPath("memory/x.md"), b"one")
        await b.write(BrainPath("memory/y.md"), b"two")

    await store.batch(do_work, BatchOptions(reason="test"))
    assert await store.read(BrainPath("memory/x.md")) == b"one"
    assert await store.read(BrainPath("memory/y.md")) == b"two"


async def test_batch_write_then_delete_cancels(store: HttpStore) -> None:
    async def do_work(b):
        await b.write(BrainPath("memory/ephem.md"), b"a")
        await b.delete(BrainPath("memory/ephem.md"))

    await store.batch(do_work)
    assert await store.exists(BrainPath("memory/ephem.md")) is False


async def test_single_body_cap_enforced(store: HttpStore) -> None:
    big = b"x" * (2 * 1024 * 1024 + 1)
    with pytest.raises(ErrPayloadTooLarge):
        await store.write(BrainPath("memory/big.md"), big)


async def test_bearer_token_forwarded(client: httpx.AsyncClient) -> None:
    seen_headers: list[dict[str, str]] = []

    original = client.send

    async def spy_send(request, *args, **kwargs):
        seen_headers.append(dict(request.headers))
        return await original(request, *args, **kwargs)

    client.send = spy_send  # type: ignore[assignment]
    s = HttpStore("http://test", "brain-auth", client=client, token="jbk_secret")
    await s.write(BrainPath("memory/t.md"), b"x")
    assert any(h.get("authorization") == "Bearer jbk_secret" for h in seen_headers)
    await s.close()


async def test_problem_json_mapped_to_error(client: httpx.AsyncClient) -> None:
    s = HttpStore("http://test", "brain-err", client=client)
    with pytest.raises(ErrNotFound):
        await s.read(BrainPath("memory/never.md"))
    await s.close()


async def test_sse_dispatch_parses_change_event(client: httpx.AsyncClient) -> None:
    """Unit-level check that `_dispatch_sse` routes frames to sinks.

    The full SSE stream over ASGITransport is not exercised here because
    ASGITransport in-process buffering defeats event-stream semantics.
    The production wire path is covered by the conformance harness's
    `SSE change event fires after a commit` case.
    """
    s = HttpStore("http://test", "brain-sse", client=client)
    received: list = []

    def sink(evt):
        received.append(evt)

    s.subscribe(sink)
    frame = (
        '{"kind":"created","path":"memory/e.md","when":"2025-01-01T00:00:00Z"}'
    )
    s._dispatch_sse("change", frame)
    assert len(received) == 1
    assert str(received[0].path) == "memory/e.md"
    await s.close()


async def test_close_blocks_ops(store: HttpStore) -> None:
    await store.write(BrainPath("memory/a.md"), b"x")
    await store.close()
    with pytest.raises(ErrReadOnly):
        await store.write(BrainPath("memory/b.md"), b"y")
