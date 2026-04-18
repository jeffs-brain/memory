# SPDX-License-Identifier: Apache-2.0
"""Git-backed `Store` via `pygit2`.

Writes go to the working tree; each batch is exactly one commit. On push
rejection the store rebases (via the git CLI, since pygit2 has no rebase
plumbing) and retries once; on retry failure `ErrConflict` is raised.

Signing is delegated to an optional callable with the shape
`(payload: bytes) -> bytes` returning an ASCII-armored detached
signature. The SDK never sees the private key material.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

import pygit2

from ..errors import ErrConflict, ErrNotFound, ErrReadOnly
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
from ._util import atomic_write, list_dir, path_under, resolve

log = logging.getLogger(__name__)

_DEFAULT_BRANCH = "main"
_DEFAULT_AUTHOR = "jeffs-brain"
_DEFAULT_EMAIL = "noreply@jeffsbrain.com"
_INIT_README = "# memory\n\nGitstore-initialised brain.\n"
_AUTOSTASH_MARKER = "memory-sync-autostash"
_REMOTE_NAME = "origin"

# GitSignFn signs a serialised commit payload and returns an ASCII-armored
# detached signature. Invoked on every batch commit and on the init commit.
GitSignFn = Callable[[bytes], bytes]


class GitStore(Store):
    """Store backed by a git repository on disk."""

    def __init__(
        self,
        root: Path | str,
        *,
        remote_url: str | None = None,
        branch: str = _DEFAULT_BRANCH,
        author: str = _DEFAULT_AUTHOR,
        email: str = _DEFAULT_EMAIL,
        auto_push: bool = False,
        sign: GitSignFn | None = None,
    ) -> None:
        self.root = str(Path(root).resolve())
        self.remote_url = remote_url
        self.branch = branch
        self.author_name = author
        self.author_email = email
        self.auto_push = auto_push
        self.sign = sign
        self._lock = asyncio.Lock()
        self._closed = False
        self._sinks: dict[int, Callable[[ChangeEvent], None]] = {}
        self._next_id = 0
        self._event_queues: list[asyncio.Queue[ChangeEvent]] = []
        os.makedirs(self.root, exist_ok=True)
        self._repo = self._open_or_init()
        if self._created_fresh:
            self._init_commit()

    # --- open / init ----------------------------------------------------

    def _open_or_init(self) -> pygit2.Repository:
        git_dir = Path(self.root) / ".git"
        self._created_fresh = False
        if git_dir.exists():
            return pygit2.Repository(self.root)
        entries = [e for e in Path(self.root).iterdir() if e.name != ".git"]
        if not entries and self.remote_url:
            # Clone into the existing empty directory.
            repo = pygit2.clone_repository(self.remote_url, self.root, checkout_branch=self.branch)
            return repo
        repo = pygit2.init_repository(self.root, initial_head=self.branch)
        if self.remote_url:
            repo.remotes.create(_REMOTE_NAME, self.remote_url)
        self._created_fresh = True
        return repo

    def _init_commit(self) -> None:
        readme = Path(self.root) / "README.md"
        if not readme.exists():
            readme.write_text(_INIT_README)
        self._commit(["README.md"], "[init] memory gitstore initialised")

    def _commit(self, rel_paths: list[str], message: str) -> str:
        """Stage `rel_paths` and produce a commit. Returns new HEAD SHA."""
        index = self._repo.index
        index.read()
        for rel in rel_paths:
            abs_p = Path(self.root) / rel
            if abs_p.exists():
                index.add(rel)
            else:
                try:
                    index.remove(rel)
                except (KeyError, pygit2.GitError):
                    pass
        index.write()
        tree = index.write_tree()
        sig = pygit2.Signature(self.author_name, self.author_email)
        parents: list[pygit2.Oid] = []
        if not self._repo.head_is_unborn:
            parents = [self._repo.head.target]
        if self.sign is not None:
            commit_hex = self._repo.create_commit_string(sig, sig, message, tree, parents)
            signature = self.sign(commit_hex.encode())
            new_oid = self._repo.create_commit_with_signature(
                commit_hex, signature.decode()
            )
            # Update the branch ref to the new commit.
            branch_ref = f"refs/heads/{self.branch}"
            try:
                self._repo.references[branch_ref].set_target(new_oid)
            except KeyError:
                self._repo.references.create(branch_ref, new_oid)
            # Ensure HEAD points at the branch.
            self._repo.set_head(branch_ref)
            return str(new_oid)
        oid = self._repo.create_commit(
            f"refs/heads/{self.branch}",
            sig,
            sig,
            message,
            tree,
            parents,
        )
        return str(oid)

    # --- helpers --------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise ErrReadOnly("gitstore: closed")

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

    def _run_git(self, *args: str) -> tuple[int, str, str]:
        proc = subprocess.run(
            ["git", "-C", self.root, *args],
            capture_output=True,
            text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr

    # --- read side ------------------------------------------------------

    async def read(self, path: BrainPath) -> bytes:
        self._check_open()
        abs_p = self._resolve(path)
        try:
            with open(abs_p, "rb") as f:
                return f.read()
        except FileNotFoundError as exc:
            raise ErrNotFound(f"gitstore: read {path}: not found") from exc

    async def exists(self, path: BrainPath) -> bool:
        self._check_open()
        return os.path.exists(self._resolve(path))

    async def stat(self, path: BrainPath) -> FileInfo:
        self._check_open()
        abs_p = self._resolve(path)
        try:
            st = os.stat(abs_p)
        except FileNotFoundError as exc:
            raise ErrNotFound(f"gitstore: stat {path}: not found") from exc
        return FileInfo(
            path=BrainPath(str(path)),
            size=st.st_size,
            mtime=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
            is_dir=os.path.isdir(abs_p),
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

    # --- write side (each op is its own commit) ------------------------

    async def write(self, path: BrainPath, content: bytes) -> None:
        await self.batch(
            lambda b: b.write(path, content),
            BatchOptions(reason="write", message=f"write {path}"),
        )

    async def append(self, path: BrainPath, content: bytes) -> None:
        await self.batch(
            lambda b: b.append(path, content),
            BatchOptions(reason="append", message=f"append {path}"),
        )

    async def delete(self, path: BrainPath) -> None:
        await self.batch(
            lambda b: b.delete(path),
            BatchOptions(reason="delete", message=f"delete {path}"),
        )

    async def rename(self, src: BrainPath, dst: BrainPath) -> None:
        await self.batch(
            lambda b: b.rename(src, dst),
            BatchOptions(reason="rename", message=f"rename {src} -> {dst}"),
        )

    # --- batch ----------------------------------------------------------

    async def batch(
        self,
        fn: Callable[[Batch], Awaitable[None]] | Callable[[Batch], None],
        opts: BatchOptions | None = None,
    ) -> None:
        self._check_open()
        opts = opts or BatchOptions()
        async with self._lock:
            b = _GitBatch(self)
            result = fn(b)
            if hasattr(result, "__await__"):
                await result  # type: ignore[misc]
            events = await b.commit(opts)
        for evt in events:
            self._dispatch(evt)
        if self.auto_push and self.remote_url:
            await self._push_with_retry()

    # --- subscribe / close / push ---------------------------------------

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

    async def freshen(self) -> None:
        """Fetch and rebase against the configured remote.

        Mirrors `Store.Freshen` in Go. Raises `ErrConflict` on a merge
        conflict or stash-pop conflict.
        """
        if not self.remote_url:
            return
        rc, _, _ = self._run_git("fetch", _REMOTE_NAME, self.branch)
        if rc != 0:
            raise RuntimeError(f"gitstore: fetch failed rc={rc}")
        stashed = await self._autostash_if_dirty()
        rc, _, err = self._run_git("rebase", f"{_REMOTE_NAME}/{self.branch}")
        if rc != 0:
            self._run_git("rebase", "--abort")
            if stashed:
                self._run_git("stash", "pop")
            raise ErrConflict(
                f"gitstore: rebase conflicted on {_REMOTE_NAME}/{self.branch}: {err}"
            )
        if stashed:
            rc, _, err = self._run_git("stash", "pop")
            if rc != 0:
                raise ErrConflict(
                    f"gitstore: stash pop conflicted after rebase: {err}"
                )

    async def push(self) -> None:
        if not self.remote_url:
            raise RuntimeError("gitstore: no remote configured")
        rc, _, err = self._run_git("push", _REMOTE_NAME, self.branch)
        if rc != 0:
            raise RuntimeError(f"gitstore: push failed: {err}")

    async def _autostash_if_dirty(self) -> bool:
        rc, out, _ = self._run_git("status", "--porcelain")
        if rc != 0 or not out.strip():
            return False
        rc, _, _ = self._run_git(
            "stash", "push", "--include-untracked", "-m", _AUTOSTASH_MARKER
        )
        return rc == 0

    async def _push_with_retry(self) -> None:
        rc, _, err = self._run_git("push", _REMOTE_NAME, self.branch)
        if rc == 0:
            return
        if not _is_non_fast_forward(err):
            log.warning("gitstore: push failed, commit remains local: %s", err)
            return
        try:
            await self.freshen()
        except ErrConflict:
            raise ErrConflict(
                f"gitstore: push rejected by {_REMOTE_NAME}/{self.branch}: rebase conflict"
            )
        rc, _, err = self._run_git("push", _REMOTE_NAME, self.branch)
        if rc != 0:
            if _is_non_fast_forward(err):
                raise ErrConflict(
                    f"gitstore: push still rejected after rebase: {err}"
                )
            log.warning("gitstore: push retry failed, commit remains local: %s", err)


def _is_non_fast_forward(err: str) -> bool:
    return (
        "non-fast-forward" in err
        or "stale info" in err
        or "rejected" in err
    )


# ---------- batch ---------------------------------------------------------


@dataclass(slots=True)
class _Op:
    kind: str
    path: BrainPath
    content: bytes = b""
    src: BrainPath | None = None


class _GitBatch(Batch):
    """Collects mutations into a single commit."""

    def __init__(self, store: GitStore) -> None:
        self.store = store
        self.ops: list[_Op] = []

    async def _effective(
        self, path: BrainPath, upto: int | None = None
    ) -> tuple[bytes | None, bool, bool]:
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

    async def read(self, path: BrainPath) -> bytes:
        validate_path(str(path))
        content, present, _ = await self._effective(path)
        if not present:
            raise ErrNotFound(f"gitstore: read {path}: not found")
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
            raise ErrNotFound(f"gitstore: stat {path}: not found")
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
        result = [
            fi
            for fi in by_path.values()
            if opts.include_generated or not is_generated(fi.path)
        ]
        result.sort(key=lambda fi: fi.path)
        return result

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
            raise ErrNotFound(f"gitstore: delete {path}: not found")
        self.ops.append(_Op(kind="delete", path=BrainPath(str(path))))

    async def rename(self, src: BrainPath, dst: BrainPath) -> None:
        validate_path(str(src))
        validate_path(str(dst))
        _, present, _ = await self._effective(src)
        if not present:
            raise ErrNotFound(f"gitstore: rename {src}: not found")
        self.ops.append(_Op(kind="rename", path=BrainPath(str(dst)), src=BrainPath(str(src))))

    async def commit(self, opts: BatchOptions) -> list[ChangeEvent]:
        if not self.ops:
            return []
        touched: list[BrainPath] = []
        seen: set[BrainPath] = set()
        for op in self.ops:
            for p in ([op.path] + ([op.src] if op.kind == "rename" and op.src else [])):
                if p not in seen:
                    seen.add(p)
                    touched.append(p)

        existed_before: dict[BrainPath, bool] = {}
        for op in self.ops:
            if op.kind in ("write", "append", "rename"):
                if op.path not in existed_before:
                    existed_before[op.path] = await self.store.exists(op.path)

        plan: list[tuple[str, BrainPath, bytes]] = []
        for p in touched:
            content, present, from_store = await self._effective(p)
            if present:
                if from_store:
                    continue
                assert content is not None
                plan.append(("write", p, content))
            elif await self.store.exists(p):
                plan.append(("delete", p, b""))

        # Apply the plan against the working tree.
        for kind, p, content in plan:
            abs_p = self.store._resolve(p)
            if kind == "write":
                try:
                    current = await self.store.read(p)
                    if current == content:
                        continue
                except ErrNotFound:
                    pass
                atomic_write(abs_p, content)
            else:
                try:
                    os.remove(abs_p)
                except FileNotFoundError:
                    pass

        if not plan:
            return []

        # Compute the commit message.
        reason = opts.reason or "write"
        if opts.message:
            subject = f"[{reason}] {opts.message}"
        else:
            summary = ", ".join(str(p) for _, p, _ in plan[:3]) or "batch"
            if len(plan) > 3:
                summary += f" (+{len(plan) - 3} more)"
            subject = f"[{reason}] {summary}"

        # Resolve the relative on-disk paths to stage.
        from ._util import relative

        rel_paths = [relative(p) for _, p, _ in plan]

        try:
            self.store._commit(rel_paths, subject)
        except Exception as exc:
            raise exc

        # Synthesise events in journal order.
        events: list[ChangeEvent] = []
        for op in self.ops:
            if op.kind in ("write", "append"):
                kind = (
                    ChangeKind.CREATED
                    if not existed_before.get(op.path, False)
                    else ChangeKind.UPDATED
                )
                existed_before[op.path] = True
                events.append(
                    ChangeEvent(
                        kind=kind, path=op.path, when=_now(), reason=opts.reason
                    )
                )
            elif op.kind == "delete":
                events.append(
                    ChangeEvent(
                        kind=ChangeKind.DELETED, path=op.path, when=_now(), reason=opts.reason
                    )
                )
            elif op.kind == "rename":
                events.append(
                    ChangeEvent(
                        kind=ChangeKind.RENAMED,
                        path=op.path,
                        old_path=op.src,
                        when=_now(),
                        reason=opts.reason,
                    )
                )
        return events


def _now() -> datetime:
    return datetime.now(timezone.utc)
