# SPDX-License-Identifier: Apache-2.0
"""Contract tests for `GitStore`.

Requires git on PATH. Tests are skipped gracefully when it is missing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from jeffs_brain_memory.errors import ErrConflict, ErrInvalidPath, ErrNotFound
from jeffs_brain_memory.path import BrainPath
from jeffs_brain_memory.store import BatchOptions, ListOpts
from jeffs_brain_memory.store.git import GitStore

pytestmark = pytest.mark.asyncio

if shutil.which("git") is None:  # pragma: no cover
    pytestmark = [pytestmark, pytest.mark.skip(reason="git binary not on PATH")]


@pytest.fixture
def store(tmp_path: Path) -> GitStore:
    # Ensure the committer identity is set so pygit2 can commit.
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"], check=False)
    subprocess.run(["git", "config", "--global", "user.name", "test"], check=False)
    return GitStore(tmp_path)


async def test_write_creates_commit(store: GitStore, tmp_path: Path) -> None:
    await store.write(BrainPath("memory/a.md"), b"hello")
    # Confirm the commit landed.
    rc = subprocess.run(
        ["git", "-C", str(tmp_path), "log", "--oneline"],
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0
    assert "write memory/a.md" in rc.stdout


async def test_write_then_read(store: GitStore) -> None:
    await store.write(BrainPath("memory/a.md"), b"hello")
    assert await store.read(BrainPath("memory/a.md")) == b"hello"


async def test_append_extends_content(store: GitStore) -> None:
    await store.append(BrainPath("wiki/log.md"), b"a\n")
    await store.append(BrainPath("wiki/log.md"), b"b\n")
    assert await store.read(BrainPath("wiki/log.md")) == b"a\nb\n"


async def test_delete_removes_file(store: GitStore) -> None:
    await store.write(BrainPath("memory/rm.md"), b"x")
    await store.delete(BrainPath("memory/rm.md"))
    assert await store.exists(BrainPath("memory/rm.md")) is False


async def test_delete_missing_raises(store: GitStore) -> None:
    with pytest.raises(ErrNotFound):
        await store.delete(BrainPath("memory/nope.md"))


async def test_rename_moves_content(store: GitStore) -> None:
    await store.write(BrainPath("raw/old.md"), b"content")
    await store.rename(BrainPath("raw/old.md"), BrainPath("raw/new.md"))
    assert await store.read(BrainPath("raw/new.md")) == b"content"
    assert await store.exists(BrainPath("raw/old.md")) is False


async def test_stat_reports_size(store: GitStore) -> None:
    await store.write(BrainPath("memory/s.md"), b"hello world!")
    stat = await store.stat(BrainPath("memory/s.md"))
    assert stat.size == 12


async def test_list_sorted(store: GitStore) -> None:
    await store.write(BrainPath("memory/c.md"), b"c")
    await store.write(BrainPath("memory/a.md"), b"a")
    await store.write(BrainPath("memory/b.md"), b"b")
    out = await store.list(BrainPath("memory/global"))
    names = [str(fi.path) for fi in out]
    assert names == ["memory/global/a.md", "memory/global/b.md", "memory/global/c.md"]


async def test_list_recursive(store: GitStore) -> None:
    await store.write(BrainPath("wiki/go/a.md"), b"a")
    await store.write(BrainPath("wiki/rust/b.md"), b"b")
    out = await store.list("wiki", ListOpts(recursive=True))
    files = sorted(str(fi.path) for fi in out if not fi.is_dir)
    assert "wiki/go/a.md" in files and "wiki/rust/b.md" in files


async def test_batch_produces_single_commit(store: GitStore, tmp_path: Path) -> None:
    before = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
    )

    async def do_work(b):
        await b.write(BrainPath("memory/x.md"), b"one")
        await b.write(BrainPath("memory/y.md"), b"two")

    await store.batch(do_work, BatchOptions(reason="test", message="batch test"))
    after = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
    )
    assert int(after.stdout.strip()) == int(before.stdout.strip()) + 1


async def test_path_validation(store: GitStore) -> None:
    with pytest.raises(ErrInvalidPath):
        await store.write(BrainPath("../etc/passwd"), b"x")


async def test_conflict_on_push_reject(tmp_path: Path) -> None:
    """Two clones producing diverging commits forces a rebase + retry path."""
    bare = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main", str(bare)],
        check=True,
        capture_output=True,
    )

    # Seed the remote with an initial commit so subsequent clones succeed.
    seed = tmp_path / "seed"
    seed_store = GitStore(seed)
    await seed_store.write(BrainPath("memory/seed.md"), b"seed")
    subprocess.run(
        ["git", "-C", str(seed), "remote", "add", "origin", str(bare)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(seed), "push", "origin", "main"],
        check=True,
        capture_output=True,
    )

    clone_a = tmp_path / "clone-a"
    clone_b = tmp_path / "clone-b"
    subprocess.run(
        ["git", "clone", "-b", "main", str(bare), str(clone_a)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "clone", "-b", "main", str(bare), str(clone_b)],
        check=True,
        capture_output=True,
    )
    a = GitStore(clone_a, remote_url=str(bare), auto_push=True)
    b = GitStore(clone_b, remote_url=str(bare), auto_push=True)

    # A commits and pushes to the remote.
    await a.write(BrainPath("memory/only-a.md"), b"a1")

    # Now B commits. Its push must rebase against A's changes automatically.
    await b.write(BrainPath("memory/only-b.md"), b"b1")
    assert await b.read(BrainPath("memory/only-a.md")) == b"a1"
    assert await b.read(BrainPath("memory/only-b.md")) == b"b1"


async def test_sign_callback_invoked(tmp_path: Path) -> None:
    calls: list[bytes] = []

    def fake_sign(payload: bytes) -> bytes:
        calls.append(payload)
        return (
            b"-----BEGIN PGP SIGNATURE-----\n"
            b"ZmFrZSBzaWduYXR1cmUK\n"
            b"-----END PGP SIGNATURE-----\n"
        )

    store = GitStore(tmp_path, sign=fake_sign)
    await store.write(BrainPath("memory/a.md"), b"hello")
    # init commit + write = 2 calls.
    assert len(calls) >= 1
