# SPDX-License-Identifier: Apache-2.0
"""Contract tests for `MemStore`."""

from __future__ import annotations

import asyncio

import pytest

from jeffs_brain_memory.errors import ErrInvalidPath, ErrNotFound, ErrReadOnly
from jeffs_brain_memory.path import BrainPath
from jeffs_brain_memory.store import (
    BatchOptions,
    ChangeEvent,
    ChangeKind,
    ListOpts,
)
from jeffs_brain_memory.store.mem import MemStore

pytestmark = pytest.mark.asyncio


async def test_write_then_read_round_trip() -> None:
    s = MemStore()
    await s.write(BrainPath("memory/a.md"), b"hello")
    assert await s.read(BrainPath("memory/a.md")) == b"hello"


async def test_read_missing_raises_not_found() -> None:
    s = MemStore()
    with pytest.raises(ErrNotFound):
        await s.read(BrainPath("memory/nope.md"))


async def test_write_overwrites() -> None:
    s = MemStore()
    await s.write(BrainPath("wiki/x.md"), b"first")
    await s.write(BrainPath("wiki/x.md"), b"second")
    assert await s.read(BrainPath("wiki/x.md")) == b"second"


async def test_append_creates_then_extends() -> None:
    s = MemStore()
    await s.append(BrainPath("wiki/log.md"), b"a\n")
    await s.append(BrainPath("wiki/log.md"), b"b\n")
    assert await s.read(BrainPath("wiki/log.md")) == b"a\nb\n"


async def test_delete_removes_file() -> None:
    s = MemStore()
    await s.write(BrainPath("memory/rm.md"), b"x")
    await s.delete(BrainPath("memory/rm.md"))
    assert await s.exists(BrainPath("memory/rm.md")) is False


async def test_delete_missing_raises_not_found() -> None:
    s = MemStore()
    with pytest.raises(ErrNotFound):
        await s.delete(BrainPath("memory/never.md"))


async def test_rename_moves_content() -> None:
    s = MemStore()
    await s.write(BrainPath("raw/old.md"), b"content")
    await s.rename(BrainPath("raw/old.md"), BrainPath("raw/new.md"))
    assert await s.read(BrainPath("raw/new.md")) == b"content"
    assert await s.exists(BrainPath("raw/old.md")) is False


async def test_rename_missing_raises_not_found() -> None:
    s = MemStore()
    with pytest.raises(ErrNotFound):
        await s.rename(BrainPath("memory/nope.md"), BrainPath("memory/other.md"))


async def test_stat_reports_size_and_mtime() -> None:
    s = MemStore()
    await s.write(BrainPath("memory/s.md"), b"twelve bytes")
    stat = await s.stat(BrainPath("memory/s.md"))
    assert stat.size == 12
    assert stat.mtime is not None


async def test_stat_missing_raises_not_found() -> None:
    s = MemStore()
    with pytest.raises(ErrNotFound):
        await s.stat(BrainPath("memory/nope.md"))


async def test_list_flat_returns_sorted_children() -> None:
    s = MemStore()
    await s.write(BrainPath("memory/c.md"), b"c")
    await s.write(BrainPath("memory/a.md"), b"a")
    await s.write(BrainPath("memory/b.md"), b"b")
    out = await s.list("memory")
    names = [fi.path for fi in out]
    assert names == [BrainPath("memory/a.md"), BrainPath("memory/b.md"), BrainPath("memory/c.md")]


async def test_list_hides_generated_by_default() -> None:
    s = MemStore()
    await s.write(BrainPath("wiki/_index.md"), b"idx")
    await s.write(BrainPath("wiki/article.md"), b"body")
    out = await s.list("wiki")
    names = [fi.path for fi in out]
    assert BrainPath("wiki/article.md") in names
    assert BrainPath("wiki/_index.md") not in names


async def test_list_includes_generated_when_requested() -> None:
    s = MemStore()
    await s.write(BrainPath("wiki/_index.md"), b"idx")
    await s.write(BrainPath("wiki/article.md"), b"body")
    out = await s.list("wiki", ListOpts(include_generated=True))
    names = [fi.path for fi in out]
    assert BrainPath("wiki/_index.md") in names


async def test_list_recursive_walks_subtree() -> None:
    s = MemStore()
    await s.write(BrainPath("wiki/go/a.md"), b"a")
    await s.write(BrainPath("wiki/go/b.md"), b"b")
    await s.write(BrainPath("wiki/rust/c.md"), b"c")
    out = await s.list("wiki", ListOpts(recursive=True))
    names = sorted(str(fi.path) for fi in out if not fi.is_dir)
    assert names == ["wiki/go/a.md", "wiki/go/b.md", "wiki/rust/c.md"]


async def test_list_non_recursive_returns_dir_entries() -> None:
    s = MemStore()
    await s.write(BrainPath("wiki/go/a.md"), b"a")
    await s.write(BrainPath("wiki/top.md"), b"t")
    out = await s.list("wiki")
    dirs = [fi.path for fi in out if fi.is_dir]
    files = [fi.path for fi in out if not fi.is_dir]
    assert BrainPath("wiki/go") in dirs
    assert BrainPath("wiki/top.md") in files


async def test_list_glob_filters_base_names() -> None:
    s = MemStore()
    await s.write(BrainPath("wiki/a.md"), b"a")
    await s.write(BrainPath("wiki/b.txt"), b"b")
    await s.write(BrainPath("wiki/c.md"), b"c")
    out = await s.list("wiki", ListOpts(glob="*.md"))
    names = sorted(str(fi.path) for fi in out)
    assert names == ["wiki/a.md", "wiki/c.md"]


async def test_path_validation_traversal_rejected() -> None:
    s = MemStore()
    with pytest.raises(ErrInvalidPath):
        await s.write(BrainPath("../etc/passwd"), b"x")


async def test_path_validation_leading_slash_rejected() -> None:
    s = MemStore()
    with pytest.raises(ErrInvalidPath):
        await s.write(BrainPath("/absolute.md"), b"x")


async def test_path_validation_trailing_slash_rejected() -> None:
    s = MemStore()
    with pytest.raises(ErrInvalidPath):
        await s.write(BrainPath("memory/dir/"), b"x")


async def test_path_validation_null_byte_rejected() -> None:
    s = MemStore()
    with pytest.raises(ErrInvalidPath):
        await s.write(BrainPath("memory/a\x00b.md"), b"x")


async def test_batch_commits_on_success() -> None:
    s = MemStore()

    async def do_work(b):
        await b.write(BrainPath("memory/x.md"), b"one")
        await b.write(BrainPath("memory/y.md"), b"two")

    await s.batch(do_work, BatchOptions(reason="test"))
    assert await s.read(BrainPath("memory/x.md")) == b"one"
    assert await s.read(BrainPath("memory/y.md")) == b"two"


async def test_batch_write_then_delete_cancels_both() -> None:
    s = MemStore()

    async def do_work(b):
        await b.write(BrainPath("memory/ephem.md"), b"a")
        await b.delete(BrainPath("memory/ephem.md"))

    await s.batch(do_work, BatchOptions(reason="test"))
    assert await s.exists(BrainPath("memory/ephem.md")) is False


async def test_batch_write_then_write_keeps_latter() -> None:
    s = MemStore()

    async def do_work(b):
        await b.write(BrainPath("memory/ow.md"), b"first")
        await b.write(BrainPath("memory/ow.md"), b"second")

    await s.batch(do_work)
    assert await s.read(BrainPath("memory/ow.md")) == b"second"


async def test_subscribe_receives_events() -> None:
    s = MemStore()
    seen: list[ChangeEvent] = []

    def sink(evt: ChangeEvent) -> None:
        seen.append(evt)

    unsub = s.subscribe(sink)
    await s.write(BrainPath("memory/a.md"), b"x")
    await s.delete(BrainPath("memory/a.md"))
    unsub()
    await s.write(BrainPath("memory/b.md"), b"y")
    assert [e.kind for e in seen] == [ChangeKind.CREATED, ChangeKind.DELETED]


async def test_close_blocks_further_operations() -> None:
    s = MemStore()
    await s.write(BrainPath("memory/a.md"), b"x")
    await s.close()
    with pytest.raises(ErrReadOnly):
        await s.write(BrainPath("memory/b.md"), b"y")
