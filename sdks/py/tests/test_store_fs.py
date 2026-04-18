# SPDX-License-Identifier: Apache-2.0
"""Contract tests for `FsStore`."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from jeffs_brain_memory.errors import ErrInvalidPath, ErrNotFound, ErrReadOnly
from jeffs_brain_memory.path import BrainPath
from jeffs_brain_memory.store import (
    BatchOptions,
    ChangeEvent,
    ChangeKind,
    ListOpts,
)
from jeffs_brain_memory.store.fs import FsStore

pytestmark = pytest.mark.asyncio


@pytest.fixture
def store(tmp_path: Path) -> FsStore:
    return FsStore(tmp_path)


async def test_write_then_read_round_trip(store: FsStore) -> None:
    await store.write(BrainPath("memory/a.md"), b"hello")
    assert await store.read(BrainPath("memory/a.md")) == b"hello"


async def test_read_missing_raises_not_found(store: FsStore) -> None:
    with pytest.raises(ErrNotFound):
        await store.read(BrainPath("memory/nope.md"))


async def test_write_overwrites(store: FsStore) -> None:
    await store.write(BrainPath("wiki/x.md"), b"first")
    await store.write(BrainPath("wiki/x.md"), b"second")
    assert await store.read(BrainPath("wiki/x.md")) == b"second"


async def test_write_atomic_cleans_up_temp_files(store: FsStore, tmp_path: Path) -> None:
    await store.write(BrainPath("memory/a.md"), b"hello")
    # No leftover .brain-tmp-* files after a successful write.
    tmp_files = [p for p in tmp_path.rglob(".brain-tmp-*")]
    assert tmp_files == []


async def test_append_creates_then_extends(store: FsStore) -> None:
    await store.append(BrainPath("wiki/log.md"), b"a\n")
    await store.append(BrainPath("wiki/log.md"), b"b\n")
    assert await store.read(BrainPath("wiki/log.md")) == b"a\nb\n"


async def test_delete_removes_file(store: FsStore) -> None:
    await store.write(BrainPath("memory/rm.md"), b"x")
    await store.delete(BrainPath("memory/rm.md"))
    assert await store.exists(BrainPath("memory/rm.md")) is False


async def test_delete_missing_raises_not_found(store: FsStore) -> None:
    with pytest.raises(ErrNotFound):
        await store.delete(BrainPath("memory/never.md"))


async def test_rename_moves_content(store: FsStore) -> None:
    await store.write(BrainPath("raw/old.md"), b"content")
    await store.rename(BrainPath("raw/old.md"), BrainPath("raw/new.md"))
    assert await store.read(BrainPath("raw/new.md")) == b"content"
    assert await store.exists(BrainPath("raw/old.md")) is False


async def test_rename_missing_raises_not_found(store: FsStore) -> None:
    with pytest.raises(ErrNotFound):
        await store.rename(BrainPath("memory/nope.md"), BrainPath("memory/other.md"))


async def test_stat_returns_size_and_mtime(store: FsStore) -> None:
    await store.write(BrainPath("memory/s.md"), b"twelve bytes")
    stat = await store.stat(BrainPath("memory/s.md"))
    assert stat.size == 12
    assert stat.mtime is not None


async def test_stat_missing_raises_not_found(store: FsStore) -> None:
    with pytest.raises(ErrNotFound):
        await store.stat(BrainPath("memory/nope.md"))


async def test_list_flat_returns_sorted_children(store: FsStore) -> None:
    await store.write(BrainPath("memory/c.md"), b"c")
    await store.write(BrainPath("memory/a.md"), b"a")
    await store.write(BrainPath("memory/b.md"), b"b")
    out = await store.list(BrainPath("memory/global"))
    names = [str(fi.path) for fi in out]
    assert names == ["memory/global/a.md", "memory/global/b.md", "memory/global/c.md"]


async def test_list_hides_generated_by_default(store: FsStore) -> None:
    await store.write(BrainPath("wiki/_index.md"), b"idx")
    await store.write(BrainPath("wiki/article.md"), b"body")
    out = await store.list("wiki")
    names = [str(fi.path) for fi in out]
    assert "wiki/article.md" in names
    assert "wiki/_index.md" not in names


async def test_list_includes_generated_when_requested(store: FsStore) -> None:
    await store.write(BrainPath("wiki/_index.md"), b"idx")
    await store.write(BrainPath("wiki/article.md"), b"body")
    out = await store.list("wiki", ListOpts(include_generated=True))
    names = [str(fi.path) for fi in out]
    assert "wiki/_index.md" in names


async def test_list_recursive_walks_subtree(store: FsStore) -> None:
    await store.write(BrainPath("wiki/go/a.md"), b"a")
    await store.write(BrainPath("wiki/go/b.md"), b"b")
    await store.write(BrainPath("wiki/rust/c.md"), b"c")
    out = await store.list("wiki", ListOpts(recursive=True))
    names = sorted(str(fi.path) for fi in out if not fi.is_dir)
    assert names == ["wiki/go/a.md", "wiki/go/b.md", "wiki/rust/c.md"]


async def test_list_non_recursive_returns_dir_entries(store: FsStore) -> None:
    await store.write(BrainPath("wiki/go/a.md"), b"a")
    await store.write(BrainPath("wiki/top.md"), b"t")
    out = await store.list("wiki")
    dirs = [str(fi.path) for fi in out if fi.is_dir]
    files = [str(fi.path) for fi in out if not fi.is_dir]
    assert "wiki/go" in dirs
    assert "wiki/top.md" in files


async def test_list_glob_filters_base_names(store: FsStore) -> None:
    await store.write(BrainPath("wiki/a.md"), b"a")
    await store.write(BrainPath("wiki/b.txt"), b"b")
    await store.write(BrainPath("wiki/c.md"), b"c")
    out = await store.list("wiki", ListOpts(glob="*.md"))
    names = sorted(str(fi.path) for fi in out)
    assert names == ["wiki/a.md", "wiki/c.md"]


async def test_path_validation_traversal(store: FsStore) -> None:
    with pytest.raises(ErrInvalidPath):
        await store.write(BrainPath("../etc/passwd"), b"x")


async def test_path_validation_leading_slash(store: FsStore) -> None:
    with pytest.raises(ErrInvalidPath):
        await store.write(BrainPath("/absolute.md"), b"x")


async def test_path_validation_trailing_slash(store: FsStore) -> None:
    with pytest.raises(ErrInvalidPath):
        await store.write(BrainPath("memory/dir/"), b"x")


async def test_path_validation_null_byte(store: FsStore) -> None:
    with pytest.raises(ErrInvalidPath):
        await store.write(BrainPath("memory/a\x00b.md"), b"x")


async def test_batch_commits_on_success(store: FsStore) -> None:
    async def do_work(b):
        await b.write(BrainPath("memory/x.md"), b"one")
        await b.write(BrainPath("memory/y.md"), b"two")

    await store.batch(do_work, BatchOptions(reason="test"))
    assert await store.read(BrainPath("memory/x.md")) == b"one"
    assert await store.read(BrainPath("memory/y.md")) == b"two"


async def test_batch_write_then_delete_cancels(store: FsStore) -> None:
    async def do_work(b):
        await b.write(BrainPath("memory/ephem.md"), b"a")
        await b.delete(BrainPath("memory/ephem.md"))

    await store.batch(do_work)
    assert await store.exists(BrainPath("memory/ephem.md")) is False


async def test_subscribe_receives_events(store: FsStore) -> None:
    seen: list[ChangeEvent] = []

    def sink(evt: ChangeEvent) -> None:
        seen.append(evt)

    store.subscribe(sink)
    await store.write(BrainPath("memory/a.md"), b"x")
    await store.delete(BrainPath("memory/a.md"))
    kinds = [e.kind for e in seen]
    assert kinds[0] == ChangeKind.CREATED
    assert kinds[-1] == ChangeKind.DELETED


async def test_close_blocks_further_operations(store: FsStore) -> None:
    await store.write(BrainPath("memory/a.md"), b"x")
    await store.close()
    with pytest.raises(ErrReadOnly):
        await store.write(BrainPath("memory/b.md"), b"y")
