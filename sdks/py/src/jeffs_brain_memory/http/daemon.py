# SPDX-License-Identifier: Apache-2.0
"""HTTP daemon state: `Daemon`, `BrainManager`, `BrainResources`.

Mirrors the Go reference in `sdks/go/cmd/memory/daemon.go`. Per-brain
resources are constructed lazily on first access and cached for the
daemon's lifetime. A per-daemon `asyncio.Lock` guards the cache dict
so concurrent handlers cannot race construction.

The store used here is a self-contained filesystem passthrough that
mirrors Go's `passthroughStore`: every canonical POSIX path in the
wire protocol maps directly onto disk. Production deployments swap
this for a git or remote backend via the SDK's regular Store surface;
the daemon only needs the wire-equivalent behaviour.
"""

from __future__ import annotations

import asyncio
import errno
import fnmatch
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from .. import knowledge, memory, retrieval, search
from ..knowledge import Options as KnowledgeOptions
from ..llm import (
    FakeProvider,
    embedder_from_env,
    provider_from_env,
    resolve_embed_model,
)
from ..llm.provider import Embedder, Provider
from ..memory._memstore import FileInfo as MemFileInfo
from ..memory._memstore import ListOpts as MemListOpts
from ..memory._memstore import NotFoundError as MemNotFound
from ..path import validate_path
from ..retrieval.index_source import IndexedRow
from .daemon_vectors import backfill_vectors

_log = logging.getLogger(__name__)


class StoreError(Exception):
    """Base class for passthrough store errors."""


class StoreNotFound(StoreError):
    """Target path missing."""


class StoreReadOnly(StoreError):
    """Store has been closed."""


@dataclass(frozen=True, slots=True)
class FileInfo:
    path: str
    size: int
    mtime: datetime
    is_dir: bool


@dataclass(frozen=True, slots=True)
class ChangeEvent:
    kind: str  # created | updated | deleted | renamed
    path: str
    when: datetime
    old_path: str | None = None
    reason: str | None = None


class PassthroughStore:
    """Filesystem-backed store that maps wire paths directly to disk.

    Kept deliberately minimal: atomic writes via rename, a mutex-guarded
    subscriber map, and a simple batch journal that matches the Go
    daemon's semantics (write+delete cancels, write+write keeps last,
    rename is materialised).
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._sinks: dict[int, Callable[[ChangeEvent], None]] = {}
        self._next_sink_id = 0
        self._sink_lock = asyncio.Lock()
        self._closed = False

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, path: str) -> Path:
        validate_path(path)
        return self._root / path

    async def read(self, path: str) -> bytes:
        if self._closed:
            raise StoreReadOnly("store closed")
        abs_path = self._resolve(path)
        try:
            return abs_path.read_bytes()
        except FileNotFoundError as exc:
            raise StoreNotFound(f"passthrough: read {path}") from exc
        except IsADirectoryError as exc:
            raise StoreNotFound(f"passthrough: read {path}") from exc

    async def exists(self, path: str) -> bool:
        if self._closed:
            raise StoreReadOnly("store closed")
        abs_path = self._resolve(path)
        return abs_path.exists() and abs_path.is_file()

    async def stat(self, path: str) -> FileInfo:
        if self._closed:
            raise StoreReadOnly("store closed")
        abs_path = self._resolve(path)
        try:
            st = abs_path.stat()
        except FileNotFoundError as exc:
            raise StoreNotFound(f"passthrough: stat {path}") from exc
        return FileInfo(
            path=path,
            size=st.st_size,
            mtime=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
            is_dir=abs_path.is_dir(),
        )

    async def list(
        self,
        directory: str = "",
        *,
        recursive: bool = False,
        glob: str | None = None,
        include_generated: bool = False,
    ) -> list[FileInfo]:
        if self._closed:
            raise StoreReadOnly("store closed")
        base = self._root
        if directory:
            validate_path(directory)
            base = self._root / directory
        if not base.exists() or not base.is_dir():
            return []

        out: list[FileInfo] = []
        if recursive:
            for p in sorted(base.rglob("*")):
                rel = str(p.relative_to(self._root).as_posix())
                name = p.name
                if not include_generated and name.startswith("_"):
                    continue
                if glob and not p.is_dir() and not fnmatch.fnmatch(name, glob):
                    continue
                st = p.stat()
                out.append(
                    FileInfo(
                        path=rel,
                        size=st.st_size,
                        mtime=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
                        is_dir=p.is_dir(),
                    )
                )
        else:
            for p in sorted(base.iterdir()):
                name = p.name
                if not include_generated and name.startswith("_"):
                    continue
                if glob and not p.is_dir() and not fnmatch.fnmatch(name, glob):
                    continue
                rel = name if not directory else f"{directory}/{name}"
                st = p.stat()
                out.append(
                    FileInfo(
                        path=rel,
                        size=st.st_size,
                        mtime=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
                        is_dir=p.is_dir(),
                    )
                )
        out.sort(key=lambda fi: fi.path)
        return out

    async def write(self, path: str, content: bytes) -> None:
        if self._closed:
            raise StoreReadOnly("store closed")
        abs_path = self._resolve(path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        existed = abs_path.exists()
        tmp = abs_path.with_suffix(abs_path.suffix + ".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, abs_path)
        kind = "updated" if existed else "created"
        await self._emit(ChangeEvent(kind=kind, path=path, when=_utcnow()))

    async def append(self, path: str, content: bytes) -> None:
        if self._closed:
            raise StoreReadOnly("store closed")
        abs_path = self._resolve(path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        existed = abs_path.exists()
        with abs_path.open("ab") as fh:
            fh.write(content)
        kind = "updated" if existed else "created"
        await self._emit(ChangeEvent(kind=kind, path=path, when=_utcnow()))

    async def delete(self, path: str) -> None:
        if self._closed:
            raise StoreReadOnly("store closed")
        abs_path = self._resolve(path)
        try:
            abs_path.unlink()
        except FileNotFoundError as exc:
            raise StoreNotFound(f"passthrough: delete {path}") from exc
        except OSError as exc:
            if exc.errno == errno.EISDIR:
                raise StoreNotFound(f"passthrough: delete {path}") from exc
            raise
        await self._emit(ChangeEvent(kind="deleted", path=path, when=_utcnow()))

    async def rename(self, src: str, dst: str) -> None:
        if self._closed:
            raise StoreReadOnly("store closed")
        src_abs = self._resolve(src)
        dst_abs = self._resolve(dst)
        if not src_abs.exists():
            raise StoreNotFound(f"passthrough: rename {src}")
        dst_abs.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src_abs, dst_abs)
        await self._emit(
            ChangeEvent(
                kind="renamed",
                path=dst,
                old_path=src,
                when=_utcnow(),
            )
        )

    async def batch(
        self,
        ops: list[dict[str, Any]],
        *,
        reason: str | None = None,
    ) -> int:
        """Run a list of batch operations atomically enough for the wire
        contract. On any failure raises and leaves prior applied ops as-is
        (matching the Go daemon's passthrough behaviour)."""
        if self._closed:
            raise StoreReadOnly("store closed")
        applied = 0
        for i, op in enumerate(ops):
            kind = op.get("type")
            path = op.get("path", "")
            if kind == "write":
                content = op.get("_decoded", b"")
                await self.write(path, content)
            elif kind == "append":
                content = op.get("_decoded", b"")
                await self.append(path, content)
            elif kind == "delete":
                # delete is idempotent in batch contexts to match the
                # client's materialised semantics (write+delete cancels).
                try:
                    await self.delete(path)
                except StoreNotFound:
                    pass
            elif kind == "rename":
                to = op.get("to", "")
                await self.rename(path, to)
            else:
                raise ValueError(f"unknown op type at index {i}: {kind!r}")
            applied += 1
        return applied

    async def subscribe(
        self,
        sink: Callable[[ChangeEvent], Awaitable[None] | None],
    ) -> Callable[[], None]:
        async with self._sink_lock:
            sink_id = self._next_sink_id
            self._next_sink_id += 1
            self._sinks[sink_id] = sink

        def unsubscribe() -> None:
            self._sinks.pop(sink_id, None)

        return unsubscribe

    async def _emit(self, event: ChangeEvent) -> None:
        async with self._sink_lock:
            sinks = list(self._sinks.values())
        for sink in sinks:
            try:
                result = sink(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001 - subscriber isolation
                continue

    async def close(self) -> None:
        self._closed = True
        async with self._sink_lock:
            self._sinks.clear()


@dataclass(slots=True)
class BrainResources:
    """Per-brain runtime objects, matching Go's `BrainResources`.

    Each brain owns a filesystem passthrough store (wire side), a
    synchronous ``memory.MemoryManager`` that reads/writes the same
    tree, a SQLite-backed ``search.Index`` for BM25, a ``Retriever``
    wrapping it, and a ``knowledge.Base`` for ingest/search delegation.
    The daemon-level LLM provider and embedder are injected here so
    handlers pick them up without a detour through ``app.state``.
    """

    id: str
    root: Path
    store: PassthroughStore
    memory_store: "_FsMemoryStore"
    memory_manager: memory.MemoryManager
    search_index: search.Index
    retriever: retrieval.Retriever
    knowledge_base: knowledge.Base
    embed_model: str = ""
    _unsubscribe: Callable[[], None] | None = None
    _backfill_task: asyncio.Task[None] | None = None

    async def close(self) -> None:
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:  # noqa: BLE001
                pass
            self._unsubscribe = None
        if self._backfill_task is not None and not self._backfill_task.done():
            self._backfill_task.cancel()
            try:
                await self._backfill_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._backfill_task = None
        try:
            await self.knowledge_base.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.search_index.close()
        except Exception:  # noqa: BLE001
            pass
        await self.store.close()


class BrainNotFound(Exception):
    """Raised when the daemon is asked for a brain that does not exist."""


class BrainConflict(Exception):
    """Raised when the daemon is asked to create a brain that already exists."""


class BrainManager:
    """Lazy per-brain resource manager, guarded by an asyncio lock."""

    def __init__(self, daemon: "Daemon") -> None:
        self._daemon = daemon
        self._lock = asyncio.Lock()
        self._cache: dict[str, BrainResources] = {}

    def _root_for(self, brain_id: str) -> Path:
        return self._daemon.root / "brains" / brain_id

    def _exists_on_disk(self, brain_id: str) -> bool:
        return self._root_for(brain_id).is_dir()

    async def list(self) -> list[str]:
        brains_dir = self._daemon.root / "brains"
        if not brains_dir.exists():
            return []
        return sorted(p.name for p in brains_dir.iterdir() if p.is_dir())

    async def get(self, brain_id: str) -> BrainResources:
        if not brain_id:
            raise ValueError("brainId required")
        async with self._lock:
            cached = self._cache.get(brain_id)
            if cached is not None:
                return cached
            if not self._exists_on_disk(brain_id):
                raise BrainNotFound(brain_id)
            resources = await self._build(brain_id)
            self._cache[brain_id] = resources
            return resources

    async def create(self, brain_id: str) -> BrainResources:
        if not brain_id:
            raise ValueError("brainId required")
        async with self._lock:
            if self._exists_on_disk(brain_id) or brain_id in self._cache:
                raise BrainConflict(brain_id)
            self._root_for(brain_id).mkdir(parents=True, exist_ok=False)
            resources = await self._build(brain_id)
            self._cache[brain_id] = resources
            return resources

    async def delete(self, brain_id: str) -> None:
        async with self._lock:
            existing = self._cache.pop(brain_id, None)
        if existing is not None:
            await existing.close()
        root = self._root_for(brain_id)
        if not root.is_dir():
            raise BrainNotFound(brain_id)
        shutil.rmtree(root)

    async def close(self) -> None:
        async with self._lock:
            cache = self._cache
            self._cache = {}
        for entry in cache.values():
            await entry.close()

    async def _build(self, brain_id: str) -> BrainResources:
        root = self._root_for(brain_id)
        store = PassthroughStore(root)

        mem_store = _FsMemoryStore(root)
        manager = memory.MemoryManager(mem_store)

        db_path = self._daemon.root / "indices" / brain_id / "search.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        index = search.Index(str(db_path))

        embed_model = self._daemon.embed_model
        retrieval_adapter = _IndexForRetrieval(index)
        vector_adapter: _IndexVectorStore | None = None
        if self._daemon.embedder is not None and embed_model:
            vector_adapter = _IndexVectorStore(index)
        source = retrieval.IndexSource(
            search_index=retrieval_adapter,
            embedder=self._daemon.embedder,
            vectors=vector_adapter,
            model=embed_model,
        )
        retriever = retrieval.Retriever(
            source=source,
            embedder=self._daemon.embedder,
        )

        knowledge_index = _IndexForKnowledge(index, store)
        kb = knowledge.new(
            KnowledgeOptions(
                brain_id=brain_id,
                store=store,
                index=knowledge_index,
            )
        )

        # Reindex anything already on disk (warm restart, or a brain
        # populated out-of-band by the Go eval runner) before accepting
        # traffic. ``index.rebuild`` calls ``asyncio.run`` internally
        # which is incompatible with a live loop, so walk the store via
        # ``_rebuild_sync`` instead.
        try:
            await _rebuild_sync(index, store)
        except Exception as exc:  # noqa: BLE001
            _log.debug("search: initial rebuild failed for %s: %s", brain_id, exc)

        unsubscribe = await _subscribe_reindex(store, index)

        # Vector backfill runs detached so the first /search request is
        # served immediately via BM25 while remote embed batches populate
        # the vector index in the background. Matches the Go daemon's
        # goroutine in daemon.go:build().
        backfill_task: asyncio.Task[None] | None = None
        if vector_adapter is not None and self._daemon.embedder is not None:
            backfill_task = asyncio.create_task(
                backfill_vectors(
                    brain_id=brain_id,
                    store=store,
                    index=index,
                    embedder=self._daemon.embedder,
                    model=embed_model,
                    logger=_log,
                ),
                name=f"jb-backfill-{brain_id}",
            )

        return BrainResources(
            id=brain_id,
            root=root,
            store=store,
            memory_store=mem_store,
            memory_manager=manager,
            search_index=index,
            retriever=retriever,
            knowledge_base=kb,
            embed_model=embed_model,
            _unsubscribe=unsubscribe,
            _backfill_task=backfill_task,
        )


@dataclass(slots=True)
class Daemon:
    """HTTP daemon state, matching Go's `Daemon`."""

    root: Path
    auth_token: str | None = None
    llm: Provider | None = None
    embedder: Embedder | None = None
    embed_model: str = ""
    brains: BrainManager = field(init=False)

    def __post_init__(self) -> None:
        self.brains = BrainManager(self)

    @classmethod
    async def create(
        cls,
        *,
        root: Path | str | None = None,
        auth_token: str | None = None,
        llm: Provider | None = None,
        embedder: Embedder | None = None,
    ) -> "Daemon":
        resolved_root = Path(root) if root else Path.home() / ".jeffs-brain"
        resolved_root = resolved_root.resolve()
        resolved_root.mkdir(parents=True, exist_ok=True)
        if llm is None:
            try:
                llm = provider_from_env()
            except Exception:  # noqa: BLE001 - fall through to fake
                llm = FakeProvider(["ok"])
        if embedder is None:
            try:
                embedder = embedder_from_env()
            except Exception:  # noqa: BLE001
                embedder = None
        embed_model = resolve_embed_model(embedder)
        return cls(
            root=resolved_root,
            auth_token=auth_token,
            llm=llm,
            embedder=embedder,
            embed_model=embed_model,
        )

    async def close(self) -> None:
        await self.brains.close()
        if self.llm is not None:
            try:
                await self.llm.close()
            except Exception:  # noqa: BLE001
                pass
            self.llm = None
        if self.embedder is not None:
            try:
                await self.embedder.close()
            except Exception:  # noqa: BLE001
                pass
            self.embedder = None


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class _FsMemoryStore:
    """Synchronous filesystem store implementing the ``memory.Store`` protocol.

    Shares the on-disk layout with :class:`PassthroughStore` so the wire
    side and the memory layer observe the same tree without bespoke
    synchronisation. Writes use atomic rename; listing filters out
    ``_``-prefixed entries unless ``include_generated`` is set, matching
    the Go reference.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        validate_path(path)
        return self._root / path

    def read(self, path: str) -> bytes:
        try:
            return self._resolve(path).read_bytes()
        except FileNotFoundError as exc:
            raise MemNotFound(path) from exc
        except IsADirectoryError as exc:
            raise MemNotFound(path) from exc

    def write(self, path: str, content: bytes) -> None:
        abs_path = self._resolve(path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = abs_path.with_suffix(abs_path.suffix + ".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, abs_path)

    def append(self, path: str, content: bytes) -> None:
        abs_path = self._resolve(path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        with abs_path.open("ab") as fh:
            fh.write(content)

    def delete(self, path: str) -> None:
        abs_path = self._resolve(path)
        try:
            abs_path.unlink()
        except FileNotFoundError as exc:
            raise MemNotFound(path) from exc

    def exists(self, path: str) -> bool:
        return self._resolve(path).is_file()

    def stat(self, path: str) -> MemFileInfo:
        abs_path = self._resolve(path)
        try:
            st = abs_path.stat()
        except FileNotFoundError as exc:
            raise MemNotFound(path) from exc
        return MemFileInfo(
            path=path,
            size=st.st_size,
            mod_time=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
            is_dir=abs_path.is_dir(),
        )

    def list(
        self,
        prefix: str,
        opts: MemListOpts | None = None,
    ) -> list[MemFileInfo]:
        opts = opts or MemListOpts()
        base = self._root if not prefix else self._root / prefix
        if prefix:
            validate_path(prefix)
        if not base.exists() or not base.is_dir():
            return []
        out: list[MemFileInfo] = []
        iterator = base.rglob("*") if opts.recursive else base.iterdir()
        for p in iterator:
            name = p.name
            if not opts.include_generated and name.startswith("_"):
                continue
            if opts.glob and not p.is_dir() and not fnmatch.fnmatch(name, opts.glob):
                continue
            rel = p.relative_to(self._root).as_posix()
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            out.append(
                MemFileInfo(
                    path=rel,
                    size=st.st_size,
                    mod_time=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
                    is_dir=p.is_dir(),
                )
            )
        out.sort(key=lambda fi: fi.path)
        return out

    def batch(
        self,
        fn: Callable[[Any], None],
        opts: Any = None,
    ) -> None:
        """Replay mutations synchronously.

        The memory package's batch callback performs a short sequence of
        reads and writes which all resolve eagerly here. The handle
        buffers ops and delegates back to the store on commit.
        """

        class _Batch:
            def __init__(self, parent: "_FsMemoryStore") -> None:
                self.parent = parent
                self.ops: list[tuple[str, Any, ...]] = []
                self._pending_writes: dict[str, bytes] = {}
                self._pending_deletes: set[str] = set()

            def read(self, path: str) -> bytes:
                if path in self._pending_deletes:
                    raise MemNotFound(path)
                if path in self._pending_writes:
                    return self._pending_writes[path]
                return self.parent.read(path)

            def write(self, path: str, content: bytes) -> None:
                self._pending_deletes.discard(path)
                self._pending_writes[path] = bytes(content)

            def append(self, path: str, content: bytes) -> None:
                try:
                    existing = self.read(path)
                except MemNotFound:
                    existing = b""
                self.write(path, existing + bytes(content))

            def delete(self, path: str) -> None:
                self._pending_writes.pop(path, None)
                self._pending_deletes.add(path)

            def exists(self, path: str) -> bool:
                if path in self._pending_deletes:
                    return False
                if path in self._pending_writes:
                    return True
                return self.parent.exists(path)

            def stat(self, path: str) -> MemFileInfo:
                if path in self._pending_deletes:
                    raise MemNotFound(path)
                if path in self._pending_writes:
                    return MemFileInfo(
                        path=path,
                        size=len(self._pending_writes[path]),
                        mod_time=_utcnow(),
                    )
                return self.parent.stat(path)

            def list(
                self, prefix: str, list_opts: MemListOpts | None = None
            ) -> list[MemFileInfo]:
                base = self.parent.list(prefix, list_opts)
                seen = {fi.path: fi for fi in base}
                for p in self._pending_deletes:
                    seen.pop(p, None)
                for p, body in self._pending_writes.items():
                    seen[p] = MemFileInfo(
                        path=p,
                        size=len(body),
                        mod_time=_utcnow(),
                    )
                return sorted(seen.values(), key=lambda fi: fi.path)

        handle = _Batch(self)
        fn(handle)
        for p in handle._pending_deletes:
            try:
                self.delete(p)
            except MemNotFound:
                continue
        for p, body in handle._pending_writes.items():
            self.write(p, body)

    def local_path(self, path: str) -> str | None:
        try:
            return str(self._resolve(path))
        except Exception:  # noqa: BLE001
            return None


class _IndexForRetrieval:
    """Async adapter surfacing :class:`search.Index` to :mod:`retrieval`.

    Matches ``retrieval.SearchIndex``: ``search_bm25(expr, k, scope,
    project)`` and ``all_rows()``. Scores are normalised to a [0, 1]
    range so RRF fusion behaves consistently across legs.
    """

    def __init__(self, index: search.Index) -> None:
        self._index = index

    async def search_bm25(
        self, expr: str, k: int, scope: str, project: str
    ) -> list[IndexedRow]:
        filters: dict[str, Any] = {}
        if scope:
            filters["scope"] = scope
        if project:
            filters["project_slug"] = project
        opts = search.SearchOpts(max_results=k if k > 0 else 20, filters=filters)
        try:
            hits = self._index.search_bm25(expr, top_k=opts.max_results, opts=opts)
        except Exception as exc:  # noqa: BLE001
            _log.debug("search_bm25 failed: %s", exc)
            return []
        rows: list[IndexedRow] = []
        for h in hits:
            # FTS5 bm25 rank is negative (lower is better); invert for
            # display consumers.
            score = 1.0 / (1.0 + abs(h.score)) if h.score else 0.0
            rows.append(
                IndexedRow(
                    path=h.path,
                    title=h.title,
                    summary="",
                    content="",
                    snippet=h.snippet,
                    score=score,
                )
            )
        return rows

    async def all_rows(self) -> list[IndexedRow]:
        try:
            cursor = self._index._conn.execute(  # type: ignore[attr-defined]
                "SELECT path, title, summary, content FROM knowledge_chunks"
            )
        except Exception as exc:  # noqa: BLE001
            _log.debug("all_rows failed: %s", exc)
            return []
        out: list[IndexedRow] = []
        for row in cursor.fetchall():
            out.append(
                IndexedRow(
                    path=row["path"] or "",
                    title=row["title"] or "",
                    summary=row["summary"] or "",
                    content=row["content"] or "",
                )
            )
        return out


class _IndexForKnowledge:
    """Adapter exposing :meth:`update` so ``knowledge.Base`` can trigger
    a rebuild after writes.

    Every ingest persists a raw/documents/*.md file through the
    passthrough store; we just rebuild the whole brain here since the
    working set is small and the SQLite writes are cheap.
    """

    def __init__(self, index: search.Index, store: PassthroughStore) -> None:
        self._index = index
        self._store = store

    async def update(self) -> None:
        await _rebuild_sync(self._index, self._store)


async def _rebuild_sync(index: search.Index, store: PassthroughStore) -> None:
    """Drive ``index.rebuild`` from an async context.

    The underlying ``Index.rebuild`` is synchronous but its walker calls
    the store's async coroutines via ``asyncio.run``. That cannot run
    inside a live event loop, so we drive the walk manually here.
    """
    try:
        entries = await store.list("", recursive=True, include_generated=True)
    except Exception as exc:  # noqa: BLE001
        _log.debug("rebuild list failed: %s", exc)
        return
    chunks: list[search.Chunk] = []
    for entry in entries:
        if entry.is_dir:
            continue
        path = entry.path
        if not path.endswith(".md"):
            continue
        basename = path.rsplit("/", 1)[-1]
        if basename.startswith("_"):
            continue
        try:
            raw_bytes = await store.read(path)
        except Exception:  # noqa: BLE001
            continue
        from ..search.frontmatter import parse_memory_frontmatter, parse_wiki_frontmatter
        from ..search.index import _classify_path  # type: ignore[attr-defined]

        scope, project_slug = _classify_path(path)
        if not scope:
            continue
        raw = raw_bytes.decode("utf-8", errors="replace")
        if scope == "wiki":
            wiki, body = parse_wiki_frontmatter(raw)
            title, summary, tags = wiki.title, wiki.summary, wiki.tags
        else:
            mem, body = parse_memory_frontmatter(raw)
            title, summary, tags = mem.name, mem.description, mem.tags
        chunks.append(
            search.Chunk(
                id=f"{path}#0",
                document_id=path,
                path=path,
                text=body,
                metadata={
                    "title": title,
                    "summary": summary,
                    "tags": tags,
                    "scope": scope,
                    "project_slug": project_slug,
                },
            )
        )
    # Targeted diff so existing embeddings survive a rebuild: only
    # drop chunks that have disappeared from the new set. The FTS
    # virtual table has no stable identity we can diff on across
    # rebuilds, so it is still wiped wholesale; it has no dependents.
    new_ids = {c.id for c in chunks}
    try:
        with index._conn:  # type: ignore[attr-defined]
            conn = index._conn  # type: ignore[attr-defined]
            existing_rows = conn.execute(
                "SELECT chunk_id FROM knowledge_chunks"
            ).fetchall()
            existing_ids = {row["chunk_id"] for row in existing_rows}
            to_drop = existing_ids - new_ids
            if to_drop:
                placeholders = ",".join("?" * len(to_drop))
                ids_list = list(to_drop)
                conn.execute(
                    f"DELETE FROM knowledge_chunks WHERE chunk_id IN ({placeholders})",
                    ids_list,
                )
                conn.execute(
                    f"DELETE FROM knowledge_embeddings WHERE chunk_id IN ({placeholders})",
                    ids_list,
                )
            conn.execute("DELETE FROM knowledge_fts")
        if chunks:
            index.upsert_chunks(chunks)
    except Exception as exc:  # noqa: BLE001
        _log.debug("rebuild upsert failed: %s", exc)


async def _subscribe_reindex(
    store: PassthroughStore,
    index: search.Index,
) -> Callable[[], None]:
    """Attach a best-effort reindex subscriber to the passthrough.

    The subscriber rebuilds the whole index on every mutation. This is
    brute but robust for the small brains the daemon handles; Go does
    the same in ``daemon.go`` via ``idx.Update``.
    """

    async def sink(event: ChangeEvent) -> None:
        try:
            await _rebuild_sync(index, store)
        except Exception as exc:  # noqa: BLE001
            _log.debug("reindex on %s failed: %s", event.kind, exc)

    return await store.subscribe(sink)


class _IndexVectorStore:
    """Adapter bridging :class:`search.Index` to the retrieval
    :class:`VectorStore` protocol.

    The retrieval layer calls :meth:`search` with the query embedding
    and the active model; we forward to ``Index.search_vectors`` with
    the model filter so vectors written under a different embedder
    never poison the ranking.
    """

    def __init__(self, index: search.Index) -> None:
        self._index = index

    async def search(
        self,
        embedding: list[float],
        model: str,
        k: int,
    ) -> list[IndexedRow]:
        if not embedding:
            return []
        try:
            hits = self._index.search_vectors(
                embedding,
                top_k=k if k > 0 else 20,
                model=model,
            )
        except Exception as exc:  # noqa: BLE001
            _log.debug("search_vectors failed: %s", exc)
            return []
        rows: list[IndexedRow] = []
        for h in hits:
            rows.append(
                IndexedRow(
                    path=h.path,
                    title=h.title,
                    summary="",
                    content="",
                    score=float(h.score),
                )
            )
        return rows


